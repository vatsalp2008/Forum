"""Issue-prior blending across sources (cell-N weighted)."""

from personas.db import connect
from personas.sample import _lookup_priors


def _seed_priors(con, table, party, edu, age, climate, gun, tax, ideo, n):
    con.execute(
        f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)",
        [party, edu, age, climate, gun, tax, ideo, n],
    )


def test_blend_is_cell_n_weighted(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    # ANES cell: climate=0.4, n=10; CCES cell: climate=0.9, n=30.
    # Weighted mean = (0.4*10 + 0.9*30) / 40 = 0.775.
    _seed_priors(con, "anes_priors", "dem", "bachelors", "30-44", 0.4, 0.5, 0.5, 0.0, 10)
    _seed_priors(con, "cces_priors", "dem", "bachelors", "30-44", 0.9, 0.5, 0.5, 0.0, 30)
    priors = _lookup_priors(con, "dem", "bachelors", "30-44")
    assert abs(priors.p_climate_action_support - 0.775) < 1e-9


def test_below_k_anonymity_source_is_skipped(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    # CCES cell is below k=5 and must be ignored; only ANES contributes.
    _seed_priors(con, "anes_priors", "rep", "hs", "45-64", 0.3, 0.5, 0.5, 0.0, 20)
    _seed_priors(con, "cces_priors", "rep", "hs", "45-64", 0.99, 0.5, 0.5, 0.0, 2)
    priors = _lookup_priors(con, "rep", "hs", "45-64")
    assert abs(priors.p_climate_action_support - 0.3) < 1e-9


def test_no_valid_source_degrades_to_marginal(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    priors = _lookup_priors(con, "ind", "graduate", "65+")
    assert priors.p_climate_action_support == 0.5
    assert priors.ideology_score == 0.0
