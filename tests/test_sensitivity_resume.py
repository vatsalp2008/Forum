"""An interrupted sweep resumes from its finished rooms, and never mixes setups."""

import json
from pathlib import Path

import pytest

import backtest.run as run
from backtest.run import ResumeError, completed_rooms, run_sensitivity

SWEEP = dict(measure_ids=["wa_i1631", "wa_i502"], n_personas=4, seeds=[1, 2], stub=True)
RUN_DIR = Path("backtest/runs/sweep")


def _count_room_runs(monkeypatch) -> list:
    calls = []
    real = run.run_one

    def counting(mid, **kw):
        calls.append((mid, kw["seed"]))
        return real(mid, **kw)

    monkeypatch.setattr(run, "run_one", counting)
    return calls


def test_resume_reruns_only_the_missing_room(monkeypatch):
    run_sensitivity(**SWEEP, run_id="sweep")
    uninterrupted = (RUN_DIR / "sensitivity.md").read_text()
    assert len(completed_rooms("sweep")) == 4

    (RUN_DIR / "wa_i502-seed2.json").unlink()        # as if the sweep died in this room
    calls = _count_room_runs(monkeypatch)
    run_sensitivity(**SWEEP, run_id="sweep")

    assert calls == [("wa_i502", 2)]
    assert (RUN_DIR / "sensitivity.md").read_text() == uninterrupted


@pytest.mark.parametrize("field, value", [
    ("mode", "live"),                                  # never mix stub rooms into a live sweep
    ("model_version", "gemini:gemma-4-31b-it"),        # or rooms from another model
    ("n_personas", 12),                                # or another room size
])
def test_resume_refuses_a_room_saved_with_a_different_setup(monkeypatch, field, value):
    run_sensitivity(**SWEEP, run_id="sweep")
    path = RUN_DIR / "wa_i1631-seed1.json"
    saved = json.loads(path.read_text())
    saved["report"][field] = value
    path.write_text(json.dumps(saved))
    (RUN_DIR / "wa_i502-seed2.json").unlink()

    calls = _count_room_runs(monkeypatch)
    with pytest.raises(ResumeError, match=field):
        run_sensitivity(**SWEEP, run_id="sweep")
    assert calls == []                                 # refused before running anything


def test_cli_resume_reports_progress_and_rejects_unknown_runs():
    from typer.testing import CliRunner

    from forum.cli import app

    runner = CliRunner()
    args = ["sensitivity", "--stub", "--measures", "wa_i1631", "--seeds", "1,2", "--n", "4"]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    run_id = first.output.split("Run id ", 1)[1].split(".", 1)[0]

    (Path("backtest/runs") / run_id / "wa_i1631-seed2.json").unlink()
    resumed = runner.invoke(app, [*args, "--resume", run_id])
    assert resumed.exit_code == 0, resumed.output
    assert f"Resuming {run_id}: 1 of 2 rooms already finished." in resumed.output

    missing = runner.invoke(app, [*args, "--resume", "no-such-run"])
    assert missing.exit_code == 2
