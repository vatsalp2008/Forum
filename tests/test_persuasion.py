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
