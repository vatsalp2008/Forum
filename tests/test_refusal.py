from datetime import date, timedelta

import pytest

import backtest.run as run_mod
from backtest.measure_loader import GroundTruth, LoadedMeasure
from backtest.run import run_one
from forum.refusal import RefusalError, RefusalResult, check_output, check_request
from forum.state import MeasureSpec


def test_refuses_election_prediction_framing():
    res = check_request("Predict the election outcome for the upcoming senate race.")
    assert res.refused
    assert "refus" in (res.reason or "").lower()


def test_refuses_persuasive_messaging_framing():
    res = check_request("Generate persuasive messaging targeted at suburban women.")
    assert res.refused


def test_refuses_demographic_targeting():
    res = check_request("How do we convert young rural voters to support this policy?")
    assert res.refused


def test_passes_neutral_research_framing():
    res = check_request(
        "Should the state implement a carbon pollution fee on large emitters?"
    )
    assert not res.refused


def test_refuses_pre_election_window():
    near = date.today() + timedelta(days=30)
    res = check_request("This is a neutral framing.", vote_date=near)
    assert res.refused


def test_passes_outside_window():
    far = date.today() + timedelta(days=365)
    res = check_request("This is a neutral framing.", vote_date=far)
    assert not res.refused


def test_output_refusal_catches_messaging_recommendation():
    text = "Here is a messaging recommendation for the campaign: ..."
    assert check_output(text).refused


def test_output_refusal_passes_clean_report():
    text = "Predicted yes-share: 47%. Confidence interval: ±5pts. Notes follow."
    assert not check_output(text).refused


# ---- pipeline enforcement (the "binding" refusal, methodology §6) ----

def _fake_measure(framing: str) -> LoadedMeasure:
    spec = MeasureSpec(
        measure_id="wa_test",
        title="Test",
        framing=framing,
        briefing="A balanced briefing.",
        briefing_sources=[],
        pro_arguments=["pro"],
        con_arguments=["con"],
        n_rounds=1,
    )
    gt = GroundTruth("wa_test", yes_pct=50.0, no_pct=50.0, passed=False, segment_results={})
    return LoadedMeasure(spec=spec, ground_truth=gt, state="WA", year=2018, topic="test")


def test_input_refusal_halts_before_deliberation(monkeypatch, tmp_path):
    """A refused framing must raise before any deliberation runs."""
    monkeypatch.setattr(
        run_mod, "load_measure",
        lambda mid: _fake_measure("How do we convert young rural voters to support this?"),
    )
    # If deliberation were reached, this would raise a different error; assert
    # we get RefusalError, proving we halted at the input gate.
    monkeypatch.setattr(
        run_mod, "run_deliberation",
        lambda *a, **k: pytest.fail("deliberation ran despite refused framing"),
    )
    with pytest.raises(RefusalError):
        run_one("wa_test", n_personas=2, seed=1, stub=True)


def test_output_refusal_blocks_report_write(monkeypatch, tmp_path):
    """A refused output must raise and write no report file."""
    monkeypatch.setattr(run_mod, "load_measure", lambda mid: _fake_measure("A neutral policy question."))
    monkeypatch.setattr(run_mod, "check_output", lambda text: RefusalResult(refused=True, reason="blocked"))
    monkeypatch.setattr(run_mod, "RUNS_DIR", tmp_path)
    with pytest.raises(RefusalError):
        run_one("wa_test", n_personas=2, seed=1, stub=True)
    assert not list(tmp_path.rglob("*.md")), "report was written despite refused output"
