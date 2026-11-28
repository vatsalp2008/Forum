"""Baselines: does deliberation beat simply polling the personas? (fix 8)"""

import pytest

from backtest.metrics import (
    Baselines,
    SensitivityRow,
    compute_baselines,
    render_baselines,
    render_sensitivity_report,
)
from backtest.run import _stub_personas, run_all, run_one
from forum.state import Vote


def _votes(personas, stance, rnd=0):
    return [Vote(persona_id=p.persona_id, round=rnd, stance=stance, confidence=0.5,
                 rationale="x") for p in personas]


def test_compute_baselines_for_a_topic_with_a_matching_survey_item():
    personas = _stub_personas(10, "WA", 3)
    b = compute_baselines(personas, _votes(personas, 0.6), actual_pct=43.4, topic="climate")
    assert b.pre_vote_pct == pytest.approx(60.0)
    assert b.pre_vote_mae == pytest.approx(16.6)
    assert b.coin_flip_mae == pytest.approx(6.6)
    expected = 100 * sum(p.priors.p_climate_action_support for p in personas) / 10
    assert b.prior_item == "p_climate_action_support"
    assert b.prior_only_pct == pytest.approx(expected)
    assert b.prior_only_mae == pytest.approx(abs(expected - 43.4))


def test_no_prior_only_baseline_without_a_matching_item():
    personas = _stub_personas(4, "WA", 3)
    b = compute_baselines(personas, _votes(personas, 0.5), actual_pct=57.4, topic="labor")
    assert b.prior_only_pct is None and b.prior_only_mae is None


def test_render_states_whether_deliberation_helped():
    b = Baselines(pre_vote_pct=60.0, pre_vote_mae=16.6, coin_flip_mae=6.6)
    helped = "\n".join(render_baselines(b, post_pct=47.0, post_mae=3.6))
    assert "deliberation improved accuracy by 13.0 pts" in helped
    assert "n/a (no matching survey item)" in helped
    hurt = "\n".join(render_baselines(b, post_pct=70.0, post_mae=26.6))
    assert "deliberation made accuracy worse by 10.0 pts" in hurt


def test_stub_report_includes_baselines_section():
    report = run_one("wa_i1639", n_personas=6, seed=2, stub=True)
    assert report.baselines is not None
    text = report.render()
    assert "## Baselines: does deliberation add anything?" in text
    assert "Same personas before deliberation (pre-vote)" in text
    assert "p_gun_restriction_support" in text   # firearms measure has a matching item


def test_backtest_summary_reports_baseline_maes(tmp_path):
    run_all(measure_ids=["wa_i1631", "wa_i502"], n_personas=4, seed=1, stub=True)
    summary = next((tmp_path / "backtest" / "runs").rglob("summary.md")).read_text()
    assert "Pre-vote MAE" in summary and "Coin-flip MAE" in summary
    assert "Mean pre-vote MAE (no deliberation)" in summary
    assert "1 measures with a matching survey item" in summary


def test_sensitivity_report_has_pre_vote_and_coin_flip_columns():
    row = SensitivityRow(measure_id="m", actual_yes_pct=40.0, seeds=[1, 2],
                         predicted_weighted=[45.0, 47.0], mae_weighted=[5.0, 7.0],
                         brier=[0.2, 0.22], pre_mae_weighted=[9.0, 11.0])
    text = render_sensitivity_report([row], n_personas=12)
    assert "Mean pre-vote MAE" in text and "Coin-flip MAE" in text
    assert "| 10.00 | 10.00 |" in text   # mean pre-vote MAE 10.00, coin flip |50-40| = 10.00
