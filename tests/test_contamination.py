"""Contamination probe tests (stub mode)."""

from backtest.contamination import ContaminationResult, run_contamination_probe


def test_flag_thresholds():
    def mk(knows, model_yes, actual=50.0):
        return ContaminationResult(
            measure_id="m", seed=1, actual_yes_pct=actual,
            model_knows=knows, model_yes_pct=model_yes,
            model_confidence=0.9, note="",
        )

    assert mk(True, 52.0).flag == "HIGH"       # 2 pts off
    assert mk(True, 60.0).flag == "MODERATE"   # 10 pts off
    assert mk(True, 80.0).flag == "LOW"        # 30 pts off
    assert mk(False, None).flag == "NONE"
    assert mk(True, None).flag == "NONE"       # claims knowledge but no number
    assert mk(True, 52.0).recall_error == 2.0


def test_stub_probe_reports_no_knowledge():
    results = run_contamination_probe(measure_ids=["wa_i1631"], seeds=[1], stub=True)
    assert len(results) == 1
    r = results[0]
    # Stub must not fabricate prior knowledge.
    assert r.model_knows is False
    assert r.flag == "NONE"
