"""Population pipeline: citizenship, likely-voter turnout, state party calibration."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from personas import cces_loader as cces
from personas.db import connect, record_source_version
from personas.sample import _party_probs, rake_party_factors, sample_personas
from personas.schema import PopulationSpec


def _db(tmp_path):
    con = connect(tmp_path / "pop.duckdb")
    record_source_version(con, "acs", "test")
    return con


def _cell(con, *, age="30-44", race="white_nh", edu="bachelors", weight=100.0,
          count=50, citizen=True, state="WA", puma="53001"):
    con.execute(
        "INSERT INTO acs_skeleton (state, puma, age_band, sex, race_eth, education,"
        " income_band, person_weight, record_count, citizen) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [state, puma, age, "female", race, edu, "50_75k", weight, count, citizen],
    )


def _party_row(con, age, edu, race, dem, ind, rep, n=100):
    con.execute("INSERT INTO party_id_distribution VALUES (?,?,?,?,?,?,?)",
                [age, edu, race, dem, ind, rep, n])


def _spec(n, seed=1, state="WA", **kw):
    return PopulationSpec(name="t", state=state, n=n, seed=seed,
                          source_versions={"acs": "test"}, **kw)


# ---------- citizenship (fix 3) ----------

def test_citizen_only_excludes_noncitizens(tmp_path):
    con = _db(tmp_path)
    _cell(con, race="white_nh", citizen=True)
    _cell(con, race="hispanic", citizen=False, puma="53002")
    citizens = sample_personas(con, _spec(n=300))
    assert {p.demographics.race_eth for p in citizens} == {"white_nh"}
    everyone = sample_personas(con, _spec(n=300, citizen_only=False))
    assert {p.demographics.race_eth for p in everyone} == {"white_nh", "hispanic"}


def test_library_built_without_citizenship_refuses_to_sample(tmp_path):
    con = _db(tmp_path)
    _cell(con, citizen=None)   # what a pre-citizenship library looks like
    with pytest.raises(RuntimeError, match="citizenship"):
        sample_personas(con, _spec(n=5))


def test_old_library_gains_citizen_column_on_connect(tmp_path):
    import duckdb
    path = tmp_path / "old.duckdb"
    old = duckdb.connect(str(path))
    old.execute("""CREATE TABLE acs_skeleton (state VARCHAR, puma VARCHAR,
        age_band VARCHAR, sex VARCHAR, race_eth VARCHAR, education VARCHAR,
        income_band VARCHAR, person_weight DOUBLE, record_count INTEGER)""")
    old.close()
    con = connect(path)
    cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='acs_skeleton'"
    ).fetchall()}
    assert "citizen" in cols


# ---------- likely-voter turnout weighting (fix 3) ----------

def test_likely_voters_reweight_pool_by_turnout(tmp_path):
    con = _db(tmp_path)
    _cell(con, age="18-29", weight=100)
    _cell(con, age="65+", weight=100, puma="53002")
    con.execute("INSERT INTO turnout_propensity VALUES ('18-29','bachelors','white_nh',0.2,500)")
    con.execute("INSERT INTO turnout_propensity VALUES ('65+','bachelors','white_nh',0.8,500)")
    adults = sample_personas(con, _spec(n=4000))
    voters = sample_personas(con, _spec(n=4000, likely_voters=True))
    old = lambda ps: sum(p.demographics.age_band == "65+" for p in ps) / len(ps)  # noqa: E731
    assert 0.46 < old(adults) < 0.54        # equal population weight
    assert 0.77 < old(voters) < 0.83        # 0.8 / (0.8 + 0.2)


def test_likely_voters_without_turnout_table_fails_loudly(tmp_path):
    con = _db(tmp_path)
    _cell(con)
    with pytest.raises(RuntimeError, match="turnout"):
        sample_personas(con, _spec(n=5, likely_voters=True))


# ---------- state party calibration (fix 2) ----------

def _two_group_state(con):
    # National conditional leans Republican for both groups; group A is more
    # Democratic than group B.
    _cell(con, edu="graduate", weight=400)
    _cell(con, edu="hs", weight=600, puma="53002")
    _party_row(con, "30-44", "graduate", "white_nh", 0.45, 0.10, 0.45)
    _party_row(con, "30-44", "hs", "white_nh", 0.25, 0.15, 0.60)


def test_party_raking_matches_state_target(tmp_path):
    con = _db(tmp_path)
    _two_group_state(con)
    con.execute("INSERT INTO party_state_targets VALUES ('WA', 0.52, 0.12, 0.36, 1300)")
    personas = sample_personas(con, _spec(n=6000, seed=4))
    c = Counter(p.priors.party_id for p in personas)
    assert abs(c["dem"] / 6000 - 0.52) < 0.03
    assert abs(c["rep"] / 6000 - 0.36) < 0.03


def test_raking_is_exact_and_keeps_the_demographic_gradient(tmp_path):
    con = _db(tmp_path)
    _two_group_state(con)
    from personas.sample import _load_party_distribution
    dist = _load_party_distribution(con)
    groups = {("30-44", "graduate", "white_nh"): 400.0, ("30-44", "hs", "white_nh"): 600.0}
    target = np.array([0.52, 0.12, 0.36])
    f = rake_party_factors(groups, dist, target)
    adj = {k: _party_probs(dist, *k) * f / (_party_probs(dist, *k) * f).sum() for k in groups}
    marginal = sum(groups[k] * adj[k] for k in groups) / 1000.0
    assert np.allclose(marginal, target, atol=1e-8)
    # Graduates stay more Democratic than high-school grads after calibration.
    assert adj[("30-44", "graduate", "white_nh")][0] > adj[("30-44", "hs", "white_nh")][0]


def test_state_without_target_keeps_national_distribution(tmp_path):
    con = _db(tmp_path)
    _two_group_state(con)   # no party_state_targets row
    personas = sample_personas(con, _spec(n=6000, seed=4))
    rep = sum(p.priors.party_id == "rep" for p in personas) / 6000
    assert abs(rep - (0.4 * 0.45 + 0.6 * 0.60)) < 0.03


def test_backoff_party_row_is_used_not_replaced_by_uniform(tmp_path):
    """Rows with a thin race cell carry a valid (age x edu) backoff estimate."""
    con = _db(tmp_path)
    _cell(con, race="asian_nh")
    _party_row(con, "30-44", "bachelors", "asian_nh", 0.90, 0.05, 0.05, n=2)
    personas = sample_personas(con, _spec(n=2000))
    dem = sum(p.priors.party_id == "dem" for p in personas) / 2000
    assert dem > 0.85   # the old sampler drew uniformly here (~0.33)


# ---------- CCES recodes and tables ----------

def test_cces_race_recode_matches_acs_convention():
    assert cces._race_collapse("White", "No") == "white_nh"
    assert cces._race_collapse("White", "Yes") == "hispanic"   # Hispanic origin wins
    assert cces._race_collapse("Hispanic", None) == "hispanic"
    assert cces._race_collapse("Middle Eastern", "No") == "white_nh"
    assert cces._race_collapse("Mixed", "No") == "other_nh"
    assert cces._race_collapse("nonsense", "No") is None


def test_validated_vote_indicator():
    s = pd.Series(["mail", "polling", None, "unknown", ""])
    assert cces._voted(s).tolist() == [True, True, False, True, False]


def _cces_frame(rows):
    cols = ["age_band", "education", "race_eth", "citizen", "voted", "party_id",
            "state", cces.WEIGHT_VAR]
    return pd.DataFrame(rows, columns=cols)


def test_turnout_table_is_weighted_and_backs_off_thin_race_cells():
    rows = []
    # 20 white respondents: weight-2 voters x10, weight-1 non-voters x10 -> 2/3.
    rows += [("45-64", "hs", "white_nh", True, True, "dem", "WA", 2.0)] * 10
    rows += [("45-64", "hs", "white_nh", True, False, "dem", "WA", 1.0)] * 10
    # 3 Black respondents (thin): must back off to the age x edu marginal.
    rows += [("45-64", "hs", "black_nh", True, False, "dem", "WA", 1.0)] * 3
    # A non-citizen voter record must be ignored.
    rows += [("45-64", "hs", "white_nh", False, True, "dem", "WA", 50.0)]
    t = cces._turnout_table(_cces_frame(rows)).set_index("race_eth")
    assert t.loc["white_nh", "p_vote"] == pytest.approx(20 / 30)
    marginal = 20 / 33
    assert t.loc["black_nh", "p_vote"] == pytest.approx(marginal)
    assert t.loc["black_nh", "cell_n"] == 3
    assert set(t.index) == set(cces.ALL_RACES)


def test_party_targets_weighted_citizens_only_with_minimum_n():
    rows = [("30-44", "hs", "white_nh", True, True, "dem", "WA", 3.0)] * 60
    rows += [("30-44", "hs", "white_nh", True, True, "rep", "WA", 1.0)] * 60
    rows += [("30-44", "hs", "white_nh", False, True, "rep", "WA", 99.0)] * 10  # non-citizens
    rows += [("30-44", "hs", "white_nh", True, True, "rep", "WY", 1.0)] * 20   # too few
    t = cces._party_targets(_cces_frame(rows)).set_index("state")
    assert list(t.index) == ["WA"]
    assert t.loc["WA", "p_dem"] == pytest.approx(0.75)
    assert t.loc["WA", "p_rep"] == pytest.approx(0.25)
    assert t.loc["WA", "n_respondents"] == 120
