"""National (multi-state) persona sampling."""

import pytest

from personas.db import connect, record_source_version
from personas.sample import NATIONAL, sample_personas
from personas.schema import PopulationSpec


def _seed_cell(con, state, puma, weight, count=50, age="30-44"):
    con.execute(
        "INSERT INTO acs_skeleton VALUES (?,?,?,?,?,?,?,?,?)",
        [state, puma, age, "female", "white_nh", "bachelors", "50_75k", weight, count],
    )


def _db(tmp_path):
    con = connect(tmp_path / "nat.duckdb")
    record_source_version(con, "acs", "test")
    return con


def _spec(state, n=200, seed=1):
    return PopulationSpec(name=f"{state}", state=state, n=n, seed=seed,
                          source_versions={"acs": "test"})


def test_national_samples_across_states(tmp_path):
    con = _db(tmp_path)
    # WA weight 100, CA weight 300 -> national draw ~25% WA / ~75% CA.
    _seed_cell(con, "WA", "53001", weight=100)
    _seed_cell(con, "CA", "06001", weight=300)
    personas = sample_personas(con, _spec(NATIONAL, n=400, seed=7))
    states = {p.demographics.state for p in personas}
    assert states == {"WA", "CA"}, states
    ca = sum(1 for p in personas if p.demographics.state == "CA")
    # Population-weighted: CA should dominate (well above half); wide band for RNG.
    assert 0.6 < ca / len(personas) < 0.9


def test_state_scope_still_filters(tmp_path):
    con = _db(tmp_path)
    _seed_cell(con, "WA", "53001", weight=100)
    _seed_cell(con, "CA", "06001", weight=300)
    personas = sample_personas(con, _spec("WA", n=50))
    assert {p.demographics.state for p in personas} == {"WA"}


def test_national_empty_pool_raises(tmp_path):
    con = _db(tmp_path)  # no cells inserted
    with pytest.raises(RuntimeError):
        sample_personas(con, _spec(NATIONAL))
