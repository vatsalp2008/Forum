"""Backtest runner.

Loads a measure, samples personas, runs the deliberation graph, computes
metrics, and writes a report to backtest/runs/<run_id>/<measure_id>.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from backtest.measure_loader import LoadedMeasure, load_measure
from backtest.metrics import (
    MeasureReport,
    SensitivityRow,
    brier_score,
    mean_absolute_error,
    opinion_change,
    predicted_yes_pct,
    predicted_yes_share_weighted,
    render_sensitivity_report,
)
from forum.budget import DEFAULT_DELIBERATION_BUDGET_USD, CostMeter
from forum.graph import run_deliberation
from forum.llm import LLMClient
from forum.refusal import RefusalError, check_output, check_request
from forum.state import Vote
from personas.db import connect, get_source_versions
from personas.sample import sample_personas
from personas.schema import PopulationSpec

RUNS_DIR = Path("backtest/runs")


def run_one(
    measure_id: str,
    n_personas: int = 12,
    seed: int = 42,
    stub: bool = False,
    budget_usd: float = DEFAULT_DELIBERATION_BUDGET_USD,
    run_id: str | None = None,
) -> MeasureReport:
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + ("-stub" if stub else "")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    measure: LoadedMeasure = load_measure(measure_id)

    # Binding refusal layer (methodology §6), input side: halt before any LLM
    # cost if the framing matches a refused category.
    req = check_request(measure.spec.framing)
    if req.refused:
        raise RefusalError(req)

    con = connect()
    source_versions = get_source_versions(con)
    if not source_versions and not stub:
        raise RuntimeError(
            "Persona library has no loaded sources. "
            "Run `forum personas build --state WA` first, or use --stub."
        )

    if stub and not source_versions:
        # Fabricate a tiny in-memory persona pool for stub runs so that
        # the pipeline can be exercised without the ACS/ANES data.
        personas = _stub_personas(n_personas, state=measure.state, seed=seed)
    else:
        spec = PopulationSpec(
            name=f"{measure.state}-adult-citizens",
            state=measure.state,
            n=n_personas,
            seed=seed,
            source_versions=source_versions,
        )
        personas = sample_personas(con, spec)

    meter = CostMeter(cap_usd=budget_usd)
    llm = LLMClient(meter=meter, stub=stub)

    final = run_deliberation(llm, measure.spec, personas, seed=seed)

    pre_votes: list[Vote] = [v for v in final["votes"] if v.round == 0]
    post_votes: list[Vote] = [v for v in final["votes"] if v.round == measure.spec.n_rounds + 1]

    pred_thresh = predicted_yes_pct(post_votes)
    pred_weighted = predicted_yes_share_weighted(post_votes)
    actual = measure.ground_truth.yes_pct
    mae_thresh = mean_absolute_error(pred_thresh, actual)
    mae_weighted = mean_absolute_error(pred_weighted, actual)
    brier = brier_score(pred_weighted / 100.0, measure.ground_truth.passed)
    delta = opinion_change(pre_votes, post_votes)

    notes: list[str] = []
    if stub:
        notes.append("STUB MODE: no LLM calls; predictions are pseudo-random. Not a real result.")
    if not source_versions:
        notes.append("Persona library not loaded; personas were fabricated for pipeline test.")
    flagged = sum(1 for s in final.get("statements", []) if s.flagged_by_critic)
    notes.append(f"Critic flagged {flagged} statements.")

    report = MeasureReport(
        measure_id=measure.spec.measure_id,
        n_personas=len(personas),
        n_rounds=measure.spec.n_rounds,
        predicted_yes_pct_threshold=pred_thresh,
        predicted_yes_pct_weighted=pred_weighted,
        actual_yes_pct=actual,
        mae_threshold=mae_thresh,
        mae_weighted=mae_weighted,
        brier=brier,
        opinion_change=delta,
        cost_usd=meter.spent_usd,
        seed=seed,
        mode=final.get("mode", "stub" if stub else "live"),
        model_version=final.get("model_version", "stub"),
        prompt_version=final.get("prompt_version", "stub"),
        persona_lib_versions=source_versions or {"acs": "stub", "anes": "stub"},
        notes=notes,
    )

    # Binding refusal layer (methodology §6), output side: the Output
    # Synthesizer refuses to emit content that looks like messaging guidance
    # or an election forecast, regardless of how clean the input was.
    report_text = report.render()
    out = check_output(report_text)
    if out.refused:
        raise RefusalError(out)

    # Include the seed in the filename so multiple seeds of the same measure
    # (e.g. a sensitivity sweep sharing one run_id) do not clobber each other.
    stem = f"{measure_id}-seed{seed}"
    (run_dir / f"{stem}.md").write_text(report_text)
    (run_dir / f"{stem}.json").write_text(
        json.dumps(_serialize_run(final, asdict(report)), default=str, indent=2)
    )
    return report


def run_all(
    measure_ids: Iterable[str] | None = None,
    n_personas: int = 12,
    seed: int = 42,
    stub: bool = False,
) -> list[MeasureReport]:
    from backtest.measure_loader import list_measures
    if measure_ids is None:
        measure_ids = list_measures()
    run_id = time.strftime("%Y%m%d-%H%M%S") + ("-stub" if stub else "")
    reports = []
    for mid in measure_ids:
        reports.append(
            run_one(mid, n_personas=n_personas, seed=seed, stub=stub, run_id=run_id)
        )
    _write_summary(run_id, reports)
    return reports


def run_sensitivity(
    measure_ids: Iterable[str] | None = None,
    n_personas: int = 12,
    seeds: Iterable[int] = (1, 2, 3, 4, 5),
    stub: bool = False,
) -> list[SensitivityRow]:
    """Run each (measure, seed) combination; emit per-run reports plus an
    aggregate sensitivity report."""
    from backtest.measure_loader import list_measures, load_measure
    if measure_ids is None:
        measure_ids = list_measures()
    measure_ids = list(measure_ids)
    seeds = list(seeds)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-sens" + ("-stub" if stub else "")

    rows: list[SensitivityRow] = []
    for mid in measure_ids:
        preds: list[float] = []
        maes: list[float] = []
        briers: list[float] = []
        for seed in seeds:
            r = run_one(mid, n_personas=n_personas, seed=seed, stub=stub, run_id=run_id)
            preds.append(r.predicted_yes_pct_weighted)
            maes.append(r.mae_weighted)
            briers.append(r.brier)
        actual = load_measure(mid).ground_truth.yes_pct
        rows.append(SensitivityRow(
            measure_id=mid, actual_yes_pct=actual, seeds=seeds,
            predicted_weighted=preds, mae_weighted=maes, brier=briers,
        ))

    run_dir = RUNS_DIR / run_id
    (run_dir / "sensitivity.md").write_text(render_sensitivity_report(rows, n_personas))
    return rows


def _write_summary(run_id: str, reports: list[MeasureReport]) -> None:
    if not reports:
        return
    run_dir = RUNS_DIR / run_id
    lines = [
        "# Backtest summary",
        "",
        f"Run id: {run_id}",
        "",
        "| Measure | Personas | Predicted (weighted) | Actual | MAE | Brier | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.measure_id} | {r.n_personas} | {r.predicted_yes_pct_weighted:.1f}% "
            f"| {r.actual_yes_pct:.1f}% | {r.mae_weighted:.1f} | {r.brier:.4f} "
            f"| ${r.cost_usd:.4f} |"
        )
    if reports:
        avg_mae = sum(r.mae_weighted for r in reports) / len(reports)
        avg_brier = sum(r.brier for r in reports) / len(reports)
        total_cost = sum(r.cost_usd for r in reports)
        lines += [
            "",
            f"Mean MAE (weighted): {avg_mae:.2f} points",
            f"Mean Brier: {avg_brier:.4f}",
            f"Total cost: ${total_cost:.4f}",
        ]
    (run_dir / "summary.md").write_text("\n".join(lines))


def _serialize_run(final_state: dict, report_dict: dict) -> dict:
    return {
        "report": report_dict,
        "statements": [
            {
                "speaker_id": s.speaker_id,
                "round": s.round,
                "text": s.text,
                "flagged": s.flagged_by_critic,
                "critic_note": s.critic_note,
            }
            for s in final_state.get("statements", [])
        ],
        "votes": [
            {
                "persona_id": v.persona_id,
                "round": v.round,
                "stance": v.stance,
                "confidence": v.confidence,
                "rationale": v.rationale,
            }
            for v in final_state.get("votes", [])
        ],
        "personas": [
            {
                "persona_id": p.persona_id,
                "demographics": p.demographics.model_dump(),
                "priors": p.priors.model_dump(),
            }
            for p in final_state.get("personas", [])
        ],
    }


# ---------- stub helpers ----------

def _stub_personas(n: int, state: str, seed: int) -> list:
    """Fabricate personas for stub-mode pipeline tests. Not for real runs."""
    import random as _r

    from personas.schema import DemographicSkeleton, IssuePriors, Persona

    rng = _r.Random(seed)
    out = []
    age_bands = ["18-29", "30-44", "45-64", "65+"]
    for i in range(n):
        skel = DemographicSkeleton(
            state=state, puma="00000",
            age_band=rng.choice(age_bands),  # type: ignore[arg-type]
            sex=rng.choice(["male", "female"]),  # type: ignore[arg-type]
            race_eth=rng.choice(["white_nh", "black_nh", "hispanic", "asian_nh", "other_nh"]),  # type: ignore[arg-type]
            education=rng.choice(["lt_hs", "hs", "some_college", "bachelors", "graduate"]),  # type: ignore[arg-type]
            income_band=rng.choice(["lt_25k", "25_50k", "50_75k", "75_125k", "125k_plus"]),  # type: ignore[arg-type]
        )
        priors = IssuePriors(
            party_id=rng.choice(["dem", "ind", "rep"]),  # type: ignore[arg-type]
            p_climate_action_support=rng.uniform(0.2, 0.8),
            p_gun_restriction_support=rng.uniform(0.2, 0.8),
            p_tax_on_rich_support=rng.uniform(0.2, 0.8),
            ideology_score=rng.uniform(-1, 1),
        )
        out.append(
            Persona(
                persona_id=f"stub-{i:04d}",
                demographics=skel,
                priors=priors,
                sampling_seed=seed,
                source_versions={"acs": "stub", "anes": "stub"},
            )
        )
    return out
