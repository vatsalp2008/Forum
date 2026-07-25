"""Contamination probe.

The central validity confound of the backtest (methodology §4, §5.1): the
LLM has substantial training-data knowledge of well-known historical ballot
measures, so a "predicted" yes-share may be memorized rather than an emergent
deliberative result.

This probe measures that leakage directly. It asks the model — as itself, NOT
as a persona and with NO briefing or deliberation — whether it already knows the
measure's certified outcome and, if so, the certified yes-percentage. The
closer the model's cold recall is to the real result, the more a matching
backtest prediction should be discounted.

Interpretation:
    recall_error = |model_recalled_yes_pct - actual_yes_pct|
    HIGH      knows and recall_error <= 5   -> strong prior knowledge; discount backtest
    MODERATE  knows and recall_error <= 15  -> partial prior knowledge
    LOW       knows and recall_error > 15   -> claims knowledge but misremembers
    NONE      does not claim knowledge      -> cleanest case
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backtest.measure_loader import LoadedMeasure, load_measure, list_measures
from forum.budget import DEFAULT_DELIBERATION_BUDGET_USD, CostMeter
from forum.llm import DEFAULT_CITIZEN_MODEL, LLMClient
from forum.prompts import CONTAMINATION_PROBE_PROMPT, PROMPT_VERSION

RUNS_DIR = Path("backtest/runs")


@dataclass
class ContaminationResult:
    measure_id: str
    seed: int
    actual_yes_pct: float
    model_knows: bool
    model_yes_pct: float | None
    model_confidence: float
    note: str

    @property
    def recall_error(self) -> float | None:
        if self.model_knows and self.model_yes_pct is not None:
            return abs(self.model_yes_pct - self.actual_yes_pct)
        return None

    @property
    def flag(self) -> str:
        e = self.recall_error
        if e is None:
            return "NONE"
        if e <= 5:
            return "HIGH"
        if e <= 15:
            return "MODERATE"
        return "LOW"


def probe_measure(
    llm: LLMClient,
    measure: LoadedMeasure,
    seed: int,
    model: str = DEFAULT_CITIZEN_MODEL,
) -> ContaminationResult:
    """Probe the model's cold prior knowledge of one measure's outcome."""
    prompt = CONTAMINATION_PROBE_PROMPT.format(
        title=measure.spec.title,
        state=measure.state,
        year=measure.year,
        framing=measure.spec.framing,
    )
    resp = llm.generate(
        model=model,
        system="You are a factual-recall auditor.",
        user=prompt,
        json_mode=True,
        temperature=0.0,
        seed=seed,
    )
    knows, yes_pct, conf, note = False, None, 0.0, ""
    try:
        obj = json.loads(resp.text)
        knows = bool(obj.get("knows", False))
        raw = obj.get("yes_pct", None)
        yes_pct = float(raw) if raw is not None else None
        conf = float(obj.get("confidence", 0.0))
        note = str(obj.get("note", ""))[:300]
    except (json.JSONDecodeError, ValueError, TypeError):
        note = "[unparseable probe response]"
    return ContaminationResult(
        measure_id=measure.spec.measure_id,
        seed=seed,
        actual_yes_pct=measure.ground_truth.yes_pct,
        model_knows=knows,
        model_yes_pct=yes_pct,
        model_confidence=conf,
        note=note,
    )


def run_contamination_probe(
    measure_ids: Iterable[str] | None = None,
    seeds: Iterable[int] = (1,),
    stub: bool = False,
    model: str = DEFAULT_CITIZEN_MODEL,
    budget_usd: float = DEFAULT_DELIBERATION_BUDGET_USD,
) -> list[ContaminationResult]:
    """Probe each (measure, seed) and write a contamination report."""
    if measure_ids is None:
        measure_ids = list_measures()
    measure_ids = list(measure_ids)
    seeds = list(seeds)

    meter = CostMeter(cap_usd=budget_usd)
    llm = LLMClient(meter=meter, stub=stub)

    results: list[ContaminationResult] = []
    for mid in measure_ids:
        measure = load_measure(mid)
        for seed in seeds:
            results.append(probe_measure(llm, measure, seed=seed, model=model))

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-contam" + ("-stub" if stub else "")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    mode = "stub" if stub else "live"
    (run_dir / "contamination.md").write_text(
        render_contamination_report(results, model=model, mode=mode)
    )
    (run_dir / "contamination.json").write_text(
        json.dumps([r.__dict__ | {"flag": r.flag, "recall_error": r.recall_error}
                    for r in results], indent=2)
    )
    return results


def render_contamination_report(
    results: list[ContaminationResult], model: str, mode: str
) -> str:
    lines = [
        "# Contamination probe report",
        "",
        f"- Mode: {mode.upper()}"
        + ("  ⚠️  STUB — the model was not actually queried" if mode == "stub" else ""),
        f"- Probed model: {model}",
        f"- Prompt version: {PROMPT_VERSION}",
        "",
        "This probe asks the model, cold and out of character, whether it already",
        "knows each measure's certified outcome. Discount any backtest prediction",
        "whose measure shows HIGH contamination.",
        "",
        "| Measure | Seed | Knows? | Model yes% | Actual yes% | Recall err | Conf | Flag |",
        "|---|---:|:--:|---:|---:|---:|---:|:--:|",
    ]
    for r in results:
        my = f"{r.model_yes_pct:.1f}%" if r.model_yes_pct is not None else "—"
        re_ = f"{r.recall_error:.1f}" if r.recall_error is not None else "—"
        lines.append(
            f"| {r.measure_id} | {r.seed} | {'yes' if r.model_knows else 'no'} "
            f"| {my} | {r.actual_yes_pct:.1f}% | {re_} | {r.model_confidence:.2f} | {r.flag} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- **HIGH** (knows, recall error ≤ 5 pts): strong prior knowledge. A backtest",
        "  prediction that matches the real result here may be memorization, not",
        "  deliberation. Treat as contaminated.",
        "- **MODERATE** (knows, ≤ 15 pts): partial prior knowledge; interpret with caution.",
        "- **LOW** (knows but > 15 pts off): claims recall but misremembers; weak signal.",
        "- **NONE** (declines to claim knowledge): cleanest case for a valid backtest.",
    ]
    return "\n".join(lines)
