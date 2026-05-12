from forum.state import Vote
from backtest.metrics import (
    brier_score,
    mean_absolute_error,
    opinion_change,
    predicted_yes_pct,
    predicted_yes_share_weighted,
)


def _v(pid, r, stance):
    return Vote(persona_id=pid, round=r, stance=stance, confidence=0.7, rationale="x")


def test_predicted_yes_pct_majority_yes():
    votes = [_v(f"p{i}", 1, 0.8 if i < 7 else 0.2) for i in range(10)]
    assert predicted_yes_pct(votes) == 70.0


def test_predicted_yes_share_weighted_matches_mean():
    votes = [_v(f"p{i}", 1, 0.5) for i in range(4)]
    assert predicted_yes_share_weighted(votes) == 50.0


def test_brier_perfect_prediction_yes_pass():
    assert brier_score(1.0, True) == 0.0


def test_brier_worst_prediction():
    assert brier_score(0.0, True) == 1.0


def test_mae_basic():
    assert mean_absolute_error(60.0, 50.0) == 10.0


def test_opinion_change_reports_flips():
    pre = [_v("a", 0, 0.8), _v("b", 0, 0.2)]
    post = [_v("a", 6, 0.4), _v("b", 6, 0.7)]
    delta = opinion_change(pre, post)
    assert delta["n"] == 2
    assert delta["flip_rate"] == 1.0
    assert delta["mean_abs_delta"] > 0
