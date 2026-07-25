from forum.state import Vote
from backtest.metrics import (
    SEGMENT_MIN_N,
    bootstrap_ci_yes_share,
    brier_score,
    mean_absolute_error,
    opinion_change,
    predicted_yes_pct,
    predicted_yes_share_weighted,
    segment_breakdown,
)
from personas.schema import DemographicSkeleton, IssuePriors, Persona


def _v(pid, r, stance):
    return Vote(persona_id=pid, round=r, stance=stance, confidence=0.7, rationale="x")


def _persona(pid, party, age="30-44"):
    return Persona(
        persona_id=pid,
        demographics=DemographicSkeleton(
            state="WA", puma="00000", age_band=age, sex="female",
            race_eth="white_nh", education="bachelors", income_band="50_75k",
        ),
        priors=IssuePriors(
            party_id=party, p_climate_action_support=0.5,
            p_gun_restriction_support=0.5, p_tax_on_rich_support=0.5,
            ideology_score=0.0,
        ),
        sampling_seed=1,
        source_versions={"acs": "stub", "anes": "stub"},
    )


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


def test_bootstrap_ci_brackets_point_estimate():
    votes = [_v(f"p{i}", 1, s) for i, s in enumerate([0.2, 0.4, 0.6, 0.8, 0.5, 0.7])]
    point = predicted_yes_share_weighted(votes)
    lo, hi = bootstrap_ci_yes_share(votes, n_boot=500, seed=1)
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 100.0


def test_bootstrap_ci_is_deterministic_by_seed():
    votes = [_v(f"p{i}", 1, 0.3 + 0.05 * i) for i in range(8)]
    assert bootstrap_ci_yes_share(votes, seed=7) == bootstrap_ci_yes_share(votes, seed=7)


def test_segment_breakdown_suppresses_small_cells():
    # 6 democrats (reportable), 2 republicans (suppressed under k=5).
    personas = [_persona(f"d{i}", "dem") for i in range(6)] + \
               [_persona(f"r{i}", "rep") for i in range(2)]
    pre = [_v(p.persona_id, 0, 0.4) for p in personas]
    post = [_v(p.persona_id, 6, 0.6) for p in personas]
    seg = segment_breakdown(personas, pre, post)
    assert seg["party_id"]["dem"]["n"] == 6
    assert "pred_yes_pct" in seg["party_id"]["dem"]
    assert seg["party_id"]["rep"]["suppressed"] is True
    assert seg["party_id"]["rep"]["n"] == 2
    assert SEGMENT_MIN_N == 5
