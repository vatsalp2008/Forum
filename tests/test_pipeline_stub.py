"""End-to-end stub-mode pipeline test.

Verifies the deliberation graph runs to completion without an API key
and produces the expected report format.
"""

from backtest.run import run_one


def test_stub_pipeline_runs_for_i1631(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path / "..")  # run in a tmpdir-ish location
    report = run_one("wa_i1631", n_personas=4, seed=1, stub=True, budget_usd=1.0)
    assert report.measure_id == "wa_i1631"
    assert report.n_personas == 4
    assert 0.0 <= report.mae_weighted <= 100.0
    assert report.cost_usd == 0.0  # stub mode is free


def test_stub_pipeline_runs_for_i1639(monkeypatch, tmp_path):
    report = run_one("wa_i1639", n_personas=4, seed=1, stub=True, budget_usd=1.0)
    assert report.measure_id == "wa_i1639"
    # ground-truth yes for I-1639 is around 59.3
    assert 0.0 <= report.actual_yes_pct <= 100.0
