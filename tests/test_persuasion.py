"""Counterfactual persuasion-graph tests (stub mode)."""

from backtest.persuasion import (
    SpeakerInfluence,
    counterfactual_influence,
    render_persuasion_report,
    run_persuasion_graph,
)


def test_persuasion_graph_runs_stub(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = run_persuasion_graph("wa_i1631", n_personas=6, seed=1, stub=True)
    assert rows, "expected at least one speaker with influence"
    for r in rows:
        assert isinstance(r, SpeakerInfluence)
        assert r.n_personas == 6
        assert r.mean_abs_shift >= abs(r.mean_signed_shift)  # |mean| <= mean|·|
    # Sorted by descending magnitude.
    mags = [r.mean_abs_shift for r in rows]
    assert mags == sorted(mags, reverse=True)


def test_render_report_contains_table():
    rows = [SpeakerInfluence("stub-1-0001", 2, 6, 0.05, 0.12)]
    out = render_persuasion_report("wa_i1631", "stub", rows)
    assert "Persuasion graph" in out
    assert "stub-1-0001" in out
    assert "STUB" in out


def test_counterfactual_influence_shapes():
    # A speaker with no statements should not appear; moderator excluded.
    from forum.budget import CostMeter
    from forum.llm import make_llm_client
    from backtest.measure_loader import load_measure
    from backtest.run import _stub_personas
    from forum.graph import run_deliberation

    measure = load_measure("wa_i1631")
    personas = _stub_personas(6, state="WA", seed=3)
    llm = make_llm_client("gemini", meter=CostMeter(cap_usd=1.0), stub=True)
    final = run_deliberation(llm, measure.spec, personas, seed=3)
    baseline, rows = counterfactual_influence(
        llm, measure, personas, final["statements"], seed=3
    )
    assert set(baseline) == {p.persona_id for p in personas}
    assert all(r.speaker_id != "moderator" for r in rows)


def test_noise_floor_is_measured_and_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = run_persuasion_graph("wa_i1631", n_personas=6, seed=1, stub=True)
    floors = {r.noise_floor for r in rows}
    assert len(floors) == 1 and next(iter(floors)) >= 0.0   # one shared placebo floor
    for r in rows:
        assert r.above_noise == (r.mean_abs_shift > r.noise_floor)
    out = render_persuasion_report("wa_i1631", "stub", rows)
    assert "Noise floor (placebo re-vote, nothing removed)" in out
    assert "Above noise?" in out


def test_identical_votes_give_zero_influence_and_zero_noise():
    """With a perfectly consistent voter nothing should register as influence."""
    from backtest.measure_loader import load_measure
    from backtest.run import _stub_personas
    from forum.graph import run_deliberation
    from tests.fakes import ScriptedLLM

    measure = load_measure("wa_i1631")
    personas = _stub_personas(6, "WA", 2)
    llm = ScriptedLLM(stance=0.55)
    final = run_deliberation(llm, measure.spec, personas, seed=2)
    _, rows = counterfactual_influence(llm, measure, personas, final["statements"], seed=2)
    assert all(r.mean_abs_shift == 0.0 and r.noise_floor == 0.0 for r in rows)
    assert not any(r.above_noise for r in rows)
