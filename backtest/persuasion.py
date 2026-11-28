"""Counterfactual persuasion graph (ADR-005; methodology §5.5).

v0 reported "influence" as each agent's *self-attribution* of belief change —
rationalization-prone and not causal. This module measures speaker-level
influence *counterfactually* instead: it re-runs every persona's final vote
with one speaker's statements removed from the deliberation context, and
attributes the resulting stance shift to that speaker.

    influence(speaker s) = mean over personas p of [ stance(p | full)
                                                     − stance(p | context without s) ]

A positive signed influence means the speaker pushed the room toward support;
the magnitude (mean |shift|) measures how much the speaker moved minds either
way. This is a genuine leave-one-out counterfactual on the final vote — not a
full re-deliberation (that would be O(statements) reruns), but measured rather
than self-reported, which is the improvement §5.5 calls for.

Two controls keep the measurement honest:

- **Same context except the speaker.** Baseline and counterfactual votes are
  built by graph.post_vote_context: the moderator's briefing is pinned in both,
  and the transcript differs only by the removed speaker's lines. (Previously
  both used a last-20-statements window: the full context had pushed the
  briefing out, and removing any speaker pulled it back in, so every
  speaker's "influence" partly measured the briefing.)
- **Noise floor.** A placebo re-vote with the full, unchanged context and a
  different seed measures how much stances move from sampling alone. A
  speaker's shift at or below that floor is indistinguishable from noise.

Cost: baseline N votes + N placebo votes + N × (number of speakers)
counterfactual votes. Bounded and fully exercised in stub mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from backtest.measure_loader import LoadedMeasure, load_measure
from backtest.run import RUNS_DIR, _stub_personas
from forum.budget import DEFAULT_DELIBERATION_BUDGET_USD, CostMeter
from forum.determinism import stable_hash
from forum.graph import _vote_for_persona, post_vote_context, run_deliberation
from forum.llm import PROVIDER_GEMINI, make_llm_client
from forum.refusal import RefusalError, check_request
from forum.state import Statement
from personas.db import connect, get_source_versions
from personas.sample import sample_personas
from personas.schema import PopulationSpec

# Seed offset for the placebo (nothing-removed) re-vote that sets the noise floor.
PLACEBO_SEED_OFFSET = 7_919


@dataclass
class SpeakerInfluence:
    speaker_id: str
    n_statements: int
    n_personas: int
    mean_signed_shift: float   # + = pushed toward support
    mean_abs_shift: float      # magnitude of mind-changing, either direction
    noise_floor: float = 0.0   # mean |shift| of a placebo re-vote (nothing removed)

    @property
    def above_noise(self) -> bool:
        """Whether removing this speaker moved votes more than sampling noise does."""
        return self.mean_abs_shift > self.noise_floor


def counterfactual_influence(
    llm,
    measure: LoadedMeasure,
    personas: list,
    statements: list[Statement],
    seed: int,
) -> tuple[dict[str, float], list[SpeakerInfluence]]:
    """Return (baseline stance per persona, per-speaker counterfactual influence)."""
    spec = measure.spec
    post_round = spec.n_rounds + 1
    briefing, full_transcript = post_vote_context(statements)

    def final_vote(persona, transcript: str, vote_seed: int) -> float:
        return _vote_for_persona(
            llm, persona, spec, round_num=post_round, seed=vote_seed,
            briefing=briefing, transcript=transcript,
        ).stance

    # Baseline: each persona's final vote given the full deliberation.
    baseline = {p.persona_id: final_vote(p, full_transcript, seed) for p in personas}

    # Placebo: the identical context with a different sampling seed. Its mean
    # |shift| is the noise floor any real speaker effect must exceed.
    placebo = [abs(baseline[p.persona_id] - final_vote(p, full_transcript, seed + PLACEBO_SEED_OFFSET))
               for p in personas]
    noise_floor = sum(placebo) / len(placebo) if placebo else 0.0

    speakers = sorted({s.speaker_id for s in statements if s.speaker_id != "moderator"})
    rows: list[SpeakerInfluence] = []
    for sp in speakers:
        _, ablated = post_vote_context(statements, exclude_speaker=sp)
        # Distinct seed per speaker so the counterfactual vote is not an exact
        # replay of the baseline call.
        shifts = [baseline[p.persona_id] - final_vote(p, ablated, seed + stable_hash(sp, 100_000))
                  for p in personas]
        n = len(shifts)
        rows.append(SpeakerInfluence(
            speaker_id=sp,
            n_statements=sum(1 for s in statements if s.speaker_id == sp),
            n_personas=n,
            mean_signed_shift=sum(shifts) / n if n else 0.0,
            mean_abs_shift=sum(abs(x) for x in shifts) / n if n else 0.0,
            noise_floor=noise_floor,
        ))
    rows.sort(key=lambda r: r.mean_abs_shift, reverse=True)
    return baseline, rows


def render_persuasion_report(
    measure_id: str, mode: str, rows: list[SpeakerInfluence]
) -> str:
    lines = [
        f"# Persuasion graph: {measure_id}",
        "",
        f"- Mode: {mode.upper()}"
        + ("  ⚠️  STUB — pseudo-random, not a real result" if mode == "stub" else ""),
        "",
        "Counterfactual leave-one-speaker-out influence on the final vote "
        "(methodology §5.5). Measured, not self-reported.",
        "",
        f"- Noise floor (placebo re-vote, nothing removed): mean |shift| "
        f"{rows[0].noise_floor if rows else 0.0:.3f}",
        "",
        "| Speaker | Statements | Mean |shift| | Mean signed shift | Above noise? |",
        "|---|---:|---:|---:|:--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.speaker_id} | {r.n_statements} | {r.mean_abs_shift:.3f} "
            f"| {r.mean_signed_shift:+.3f} | {'yes' if r.above_noise else 'no'} |"
        )
    lines += [
        "",
        "- **Mean |shift|**: how much removing this speaker moves the room, either direction.",
        "- **Mean signed shift**: + means the speaker pushed stances toward support, − toward opposition.",
        "- **Above noise?**: only speakers whose |shift| exceeds the placebo noise floor show a",
        "  measurable effect; the rest cannot be told apart from sampling noise.",
    ]
    return "\n".join(lines)


def run_persuasion_graph(
    measure_id: str,
    n_personas: int = 12,
    seed: int = 42,
    stub: bool = False,
    budget_usd: float = DEFAULT_DELIBERATION_BUDGET_USD,
    provider: str = PROVIDER_GEMINI,
) -> list[SpeakerInfluence]:
    """Run a deliberation, then compute the counterfactual persuasion graph."""
    measure: LoadedMeasure = load_measure(measure_id)
    req = check_request(measure.spec.framing)   # binding refusal, before cost
    if req.refused:
        raise RefusalError(req)

    con = connect()
    source_versions = get_source_versions(con)
    if not source_versions and not stub:
        raise RuntimeError(
            "Persona library has no loaded sources. "
            "Run `forum personas build --state WA --cces` first, or use --stub."
        )
    if stub and not source_versions:
        personas = _stub_personas(n_personas, state=measure.state, seed=seed)
    else:
        personas = sample_personas(con, PopulationSpec(
            name=f"{measure.state}-likely-voters", state=measure.state,
            n=n_personas, seed=seed, source_versions=source_versions,
            likely_voters=True,
        ))

    meter = CostMeter(cap_usd=budget_usd)
    llm = make_llm_client(provider, meter=meter, stub=stub)
    final = run_deliberation(llm, measure.spec, personas, seed=seed)
    statements = final.get("statements", [])
    mode = final.get("mode", "stub" if stub else "live")

    _, rows = counterfactual_influence(llm, measure, personas, statements, seed=seed)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-persuasion" + ("-stub" if stub else "")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{measure_id}-seed{seed}.md").write_text(
        render_persuasion_report(measure.spec.measure_id, mode, rows)
    )
    return rows


def run_many(measure_ids: Iterable[str] | None = None, **kw) -> dict[str, list[SpeakerInfluence]]:
    from backtest.measure_loader import list_measures
    if measure_ids is None:
        measure_ids = list_measures()
    return {mid: run_persuasion_graph(mid, **kw) for mid in measure_ids}
