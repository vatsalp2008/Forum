"""Multi-group aggregation tests (stub mode)."""

from backtest.run import run_multigroup


def test_multigroup_aggregates_independent_groups(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = run_multigroup("wa_i1631", n_groups=3, group_size=6, seed=1, stub=True)
    assert report.mode == "stub"
    assert report.n_groups == 3
    assert len(report.group_predicted) == 3
    # Pooled estimate and CI are well-formed.
    assert 0.0 <= report.pooled_ci[0] <= report.pooled_predicted <= report.pooled_ci[1] <= 100.0
    # Between-group variance is computed across groups.
    assert report.between_group_stdev >= 0.0
    lo, hi = report.between_group_ci95
    assert lo <= report.between_group_mean <= hi
    # Pooled opinion change matched every persona (3 groups x 6 = 18), which
    # requires unique persona ids across groups.
    assert report.opinion_change["n"] == 18
    assert "Multi-group backtest report" in report.render()
