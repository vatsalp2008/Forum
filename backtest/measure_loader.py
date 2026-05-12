"""Load measure YAMLs into MeasureSpec + ground-truth records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from forum.state import MeasureSpec

MEASURES_DIR = Path(__file__).resolve().parent / "measures"


@dataclass
class GroundTruth:
    measure_id: str
    yes_pct: float
    no_pct: float
    passed: bool
    segment_results: dict


@dataclass
class LoadedMeasure:
    spec: MeasureSpec
    ground_truth: GroundTruth
    state: str
    year: int
    topic: str


def load_measure(measure_id: str) -> LoadedMeasure:
    candidates = list(MEASURES_DIR.rglob(f"{measure_id.replace('wa_', '')}.yaml"))
    if not candidates:
        candidates = list(MEASURES_DIR.rglob(f"{measure_id}.yaml"))
    if not candidates:
        raise FileNotFoundError(f"Measure {measure_id} not found under {MEASURES_DIR}")
    path = candidates[0]
    with path.open("r") as f:
        data = yaml.safe_load(f)

    spec = MeasureSpec(
        measure_id=data["measure_id"],
        title=data["title"],
        framing=data["framing"].strip(),
        briefing=data["briefing"].strip(),
        briefing_sources=list(data.get("briefing_sources", [])),
        pro_arguments=list(data.get("pro_arguments", [])),
        con_arguments=list(data.get("con_arguments", [])),
        n_rounds=int(data.get("n_rounds", 5)),
    )
    gt_block = data.get("ground_truth", {})
    gt = GroundTruth(
        measure_id=spec.measure_id,
        yes_pct=float(gt_block.get("yes_pct", 50.0)),
        no_pct=float(gt_block.get("no_pct", 50.0)),
        passed=bool(gt_block.get("passed", False)),
        segment_results=dict(data.get("segment_results", {})),
    )
    return LoadedMeasure(
        spec=spec,
        ground_truth=gt,
        state=data["state"],
        year=int(data["year"]),
        topic=data["topic"],
    )


def list_measures() -> list[str]:
    return [p.stem for p in MEASURES_DIR.rglob("*.yaml")]
