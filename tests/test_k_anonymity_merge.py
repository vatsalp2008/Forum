"""k-anonymity by merging thin cells statewide, not dropping them (ADR-002)."""

from collections import Counter

import pytest

from forum.prompts import render_persona_backstory
from personas.db import connect, record_source_version
from personas.sample import (
    K_ANONYMITY_MIN,
    _assert_k_anonymous,
    build_sampling_pool,
    sample_personas,
    sampling_coverage,
)
from personas.schema import STATEWIDE_PUMA, PopulationSpec


def _cell(con, puma, race, weight, count, state="WA"):
    con.execute(
        "INSERT INTO acs_skeleton (state, puma, age_band, sex, race_eth, education,"
        " income_band, person_weight, record_count, citizen) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [state, puma, "30-44", "female", race, "bachelors", "50_75k", weight, count, True],
    )


def _db(tmp_path):
    con = connect(tmp_path / "k.duckdb")
    record_source_version(con, "acs", "test")
    return con


def _spec(n, seed=1, state="WA"):
    return PopulationSpec(name="t", state=state, n=n, seed=seed,
                          source_versions={"acs": "test"})


def test_thin_cells_are_merged_statewide_not_dropped(tmp_path):
    con = _db(tmp_path)
    # Same combination in three PUMAs, each below k but 2+2+2 = 6 >= k.
    for puma in ("53001", "53002", "53003"):
        _cell(con, puma, "black_nh", weight=100, count=2)
    pool = build_sampling_pool(con, "WA")
    assert len(pool) == 1
    state, puma, *_, weight, count = pool[0]
    assert puma == STATEWIDE_PUMA
    assert weight == 300 and count == 6


def test_combinations_too_rare_even_statewide_are_refused(tmp_path):
    con = _db(tmp_path)
    _cell(con, "53001", "white_nh", weight=1000, count=50)
    _cell(con, "53001", "black_nh", weight=10, count=2)   # 2 < k even statewide
    pool = build_sampling_pool(con, "WA")
    assert all(row[4] != "black_nh" for row in pool)
    assert sampling_coverage(con, "WA") == pytest.approx(1000 / 1010)


def test_minority_living_only_in_thin_cells_keeps_its_population_share(tmp_path):
    """The bug this fixes: dropping thin cells erased dispersed minorities."""
    con = _db(tmp_path)
    _cell(con, "53001", "white_nh", weight=8000, count=400)
    # 20% of the population, spread so thinly no single PUMA cell reaches k.
    for i in range(20):
        _cell(con, f"530{i:02d}", "black_nh", weight=100, count=3)
    personas = sample_personas(con, _spec(n=4000, seed=3))
    share = Counter(p.demographics.race_eth for p in personas)["black_nh"] / 4000
    assert 0.17 < share < 0.23, share   # true share is 0.20; old code gave 0.0


def test_sampling_is_deterministic(tmp_path):
    con = _db(tmp_path)
    _cell(con, "53001", "white_nh", weight=500, count=40)
    for i in range(4):
        _cell(con, f"5301{i}", "hispanic", weight=50, count=2)
    a = sample_personas(con, _spec(n=30, seed=9))
    b = sample_personas(con, _spec(n=30, seed=9))
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]


def test_privacy_assertion_fires_on_thin_cell():
    thin_row = ("WA", "53001", "30-44", "female", "white_nh", "bachelors",
                "50_75k", 10.0, K_ANONYMITY_MIN - 1)
    with pytest.raises(RuntimeError, match="k-anonymity"):
        _assert_k_anonymous([thin_row])


def test_backstory_names_no_puma_for_merged_cells(tmp_path):
    con = _db(tmp_path)
    for puma in ("53001", "53002", "53003"):
        _cell(con, puma, "asian_nh", weight=100, count=2)
    persona = sample_personas(con, _spec(n=1))[0]
    text = render_persona_backstory(persona)
    assert "area within the state not specified" in text
    assert "PUMA statewide" not in text


def test_persona_ids_unique_when_one_cell_is_drawn_repeatedly(tmp_path):
    con = _db(tmp_path)
    _cell(con, "53001", "white_nh", weight=1000, count=50)   # the only cell
    personas = sample_personas(con, _spec(n=50))
    ids = [p.persona_id for p in personas]
    assert len(set(ids)) == 50, "duplicate persona ids corrupt pre/post vote matching"
