"""Deliberative Polling protocol implemented as a LangGraph workflow.

Nodes:
    pre_vote      -> initial stance per persona (DP phase 1; no information yet)
    briefing      -> moderator presents balanced briefing (DP phase 2)
    deliberate    -> R rounds of speaker-selection + statements (DP phase 3)
    post_vote     -> final stance per persona (DP phase 4)

What each agent sees:
    - From phase 2 on, the moderator's briefing is *pinned* into every citizen's
      context. It used to sit in the rolling window of recent statements and
      fell out of view: only 6 of 20 speakers and none of the final voters saw it.
    - A speaker sees the last SPEAKER_CONTEXT_STATEMENTS citizen statements
      (conversational recency).
    - A final voter sees the whole citizen transcript, as a real DP participant
      who sat through the discussion would.

Contamination control (methodology §4): no agent sees the measure's title,
ballot number, or year. Curated materials pass through redact_curated, and
model output passes through redact_generated before other agents see it.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph

from forum.determinism import stable_hash
from forum.llm import LLMClient
from forum.prompts import (
    BRIEFING_CONTEXT,
    CITIZEN_DELIBERATE_PROMPT,
    CITIZEN_SYSTEM_PROMPT,
    CRITIC_FACT_CHECK_PROMPT,
    MODERATOR_BRIEFING_PROMPT,
    PROMPT_VERSION,
    TRANSCRIPT_CONTEXT,
    VOTE_PROMPT,
    render_persona_backstory,
)
from forum.redaction import measure_number, redact_curated, redact_generated
from forum.state import DeliberationState, MeasureSpec, Statement, Vote

MODERATOR = "moderator"

# How many recent citizen statements a speaker sees when taking a turn.
SPEAKER_CONTEXT_STATEMENTS = 6


# ---------- what agents see ----------

def briefing_text(statements: list[Statement]) -> str:
    """The moderator's briefing (empty before phase 2)."""
    for s in statements:
        if s.speaker_id == MODERATOR:
            return s.text
    return ""


def format_transcript(
    statements: list[Statement],
    limit: int | None = None,
    exclude_speaker: str | None = None,
) -> str:
    """Citizen statements in order (the briefing is pinned separately).

    limit keeps only the most recent N; exclude_speaker drops one speaker's
    statements (used by the counterfactual persuasion graph).
    """
    lines = [s for s in statements
             if s.speaker_id != MODERATOR and s.speaker_id != exclude_speaker]
    if limit is not None:
        lines = lines[-limit:]
    if not lines:
        return "(no statements yet)"
    out = []
    for s in lines:
        flag = " [flagged by fact-checker]" if s.flagged_by_critic else ""
        out.append(f"- {s.speaker_id} (round {s.round}){flag}: {s.text}")
    return "\n".join(out)


def post_vote_context(
    statements: list[Statement], exclude_speaker: str | None = None
) -> tuple[str, str]:
    """(briefing, transcript) a final voter sees.

    Shared by the post_vote node and the persuasion graph, so a counterfactual
    vote differs from the real final vote *only* by the removed speaker.
    """
    return briefing_text(statements), format_transcript(statements, exclude_speaker=exclude_speaker)


def _citizen_system(persona, briefing: str = "") -> str:
    system = CITIZEN_SYSTEM_PROMPT + "\n\n" + render_persona_backstory(persona)
    if briefing:
        system += "\n\n" + BRIEFING_CONTEXT.format(briefing=briefing)
    return system


def _redaction_keys(measure: MeasureSpec) -> tuple[str | None, int | None]:
    return measure_number(measure.measure_id), measure.year


# ---------- helpers ----------

def _vote_for_persona(
    llm: LLMClient,
    persona,
    measure: MeasureSpec,
    round_num: int,
    seed: int,
    briefing: str = "",
    transcript: str = "",
) -> Vote:
    system = _citizen_system(persona, briefing)
    if transcript:
        system += "\n" + TRANSCRIPT_CONTEXT.format(transcript=transcript)
    number, _ = _redaction_keys(measure)
    resp = llm.generate(
        model=llm.citizen_model,
        system=system,
        user=VOTE_PROMPT.format(framing=redact_curated(measure.framing, number)),
        json_mode=True,
        temperature=0.6,
        seed=seed + stable_hash(persona.persona_id, 100_000),
    )
    try:
        obj = json.loads(resp.text)
        stance = float(obj.get("stance", 0.5))
        confidence = float(obj.get("confidence", 0.5))
        rationale = str(obj.get("rationale", ""))[:600]
    except (json.JSONDecodeError, ValueError, TypeError):
        stance, confidence, rationale = 0.5, 0.3, "[unparseable response]"
    return Vote(
        persona_id=persona.persona_id,
        round=round_num,
        stance=max(0.0, min(1.0, stance)),
        confidence=max(0.0, min(1.0, confidence)),
        rationale=rationale,
    )


# ---------- nodes ----------

def make_pre_vote_node(llm: LLMClient):
    def node(state: DeliberationState) -> DeliberationState:
        measure = state["measure"]
        personas = state["personas"]
        seed = state.get("seed", 0)
        votes = list(state.get("votes", []))
        for p in personas:
            votes.append(_vote_for_persona(llm, p, measure, round_num=0, seed=seed))
        return {**state, "votes": votes, "current_round": 0}
    return node


def make_briefing_node(llm: LLMClient):
    def node(state: DeliberationState) -> DeliberationState:
        measure = state["measure"]
        statements = list(state.get("statements", []))
        seed = state.get("seed", 0)
        number, year = _redaction_keys(measure)
        bullet = lambda items: "\n".join(f"- {redact_curated(x, number)}" for x in items)  # noqa: E731
        resp = llm.generate(
            model=llm.moderator_model,
            system="You are an impartial deliberation moderator.",
            user=MODERATOR_BRIEFING_PROMPT.format(
                framing=redact_curated(measure.framing, number),
                facts=redact_curated(measure.briefing, number),
                sources=bullet(measure.briefing_sources),
                pro=bullet(measure.pro_arguments),
                con=bullet(measure.con_arguments),
            ),
            temperature=0.4,
            seed=seed,
        )
        statements.append(
            Statement(
                speaker_id=MODERATOR,
                round=0,
                text=redact_generated(resp.text or "[empty briefing]", number, year),
                issued_at=datetime.now(timezone.utc),
            )
        )
        return {**state, "statements": statements}
    return node


def make_deliberate_node(llm: LLMClient):
    def node(state: DeliberationState) -> DeliberationState:
        measure = state["measure"]
        personas = state["personas"]
        statements = list(state.get("statements", []))
        seed = state.get("seed", 0)
        current_round = state.get("current_round", 0) + 1
        rng = random.Random(seed + current_round)
        number, year = _redaction_keys(measure)
        briefing = briefing_text(statements)
        facts = redact_curated(measure.briefing, number)

        # Speaker selection: pick K speakers per round; round-robin biased by speaking count.
        speak_counts = {p.persona_id: 0 for p in personas}
        for s in statements:
            if s.speaker_id != MODERATOR:
                speak_counts[s.speaker_id] = speak_counts.get(s.speaker_id, 0) + 1
        eligible_sorted = sorted(personas, key=lambda p: (speak_counts[p.persona_id], p.persona_id))
        speakers_this_round = eligible_sorted[: max(1, len(personas) // 3)]
        rng.shuffle(speakers_this_round)

        for speaker in speakers_this_round:
            user_prompt = CITIZEN_DELIBERATE_PROMPT.format(
                round_num=current_round,
                total_rounds=measure.n_rounds,
                recent_statements=format_transcript(statements, limit=SPEAKER_CONTEXT_STATEMENTS),
            )
            resp = llm.generate(
                model=llm.citizen_model,
                system=_citizen_system(speaker, briefing),
                user=user_prompt,
                temperature=0.7,
                seed=seed + current_round * 1000 + stable_hash(speaker.persona_id, 100_000),
            )
            text = redact_generated(resp.text or "[empty]", number, year)

            # Critic fact-check against the curated briefing facts.
            critic = llm.generate(
                model=llm.critic_model,
                system="You are a fact-checker. Be precise; do not flag normative disagreement.",
                user=CRITIC_FACT_CHECK_PROMPT.format(statement=text, facts=facts),
                json_mode=True,
                temperature=0.0,
                seed=seed,
            )
            try:
                cobj = json.loads(critic.text)
                flagged = bool(cobj.get("flagged"))
                note = str(cobj.get("note", ""))[:300] or None
            except (json.JSONDecodeError, ValueError, TypeError):
                flagged, note = False, None

            statements.append(
                Statement(
                    speaker_id=speaker.persona_id,
                    round=current_round,
                    text=text,
                    issued_at=datetime.now(timezone.utc),
                    flagged_by_critic=flagged,
                    critic_note=note,
                )
            )

        finished = current_round >= measure.n_rounds
        return {**state, "statements": statements, "current_round": current_round, "finished": finished}
    return node


def make_post_vote_node(llm: LLMClient):
    def node(state: DeliberationState) -> DeliberationState:
        measure = state["measure"]
        personas = state["personas"]
        statements = state.get("statements", [])
        votes = list(state.get("votes", []))
        seed = state.get("seed", 0)
        briefing, transcript = post_vote_context(statements)
        for p in personas:
            votes.append(
                _vote_for_persona(
                    llm, p, measure,
                    round_num=measure.n_rounds + 1,
                    seed=seed,
                    briefing=briefing,
                    transcript=transcript,
                )
            )
        return {**state, "votes": votes}
    return node


# ---------- graph wiring ----------

def _route_after_deliberate(state: DeliberationState) -> str:
    return "post_vote" if state.get("finished") else "deliberate"


def build_graph(llm: LLMClient) -> Any:
    builder: StateGraph = StateGraph(DeliberationState)
    builder.add_node("pre_vote", make_pre_vote_node(llm))
    builder.add_node("briefing", make_briefing_node(llm))
    builder.add_node("deliberate", make_deliberate_node(llm))
    builder.add_node("post_vote", make_post_vote_node(llm))

    builder.set_entry_point("pre_vote")
    builder.add_edge("pre_vote", "briefing")
    builder.add_edge("briefing", "deliberate")
    builder.add_conditional_edges("deliberate", _route_after_deliberate, {
        "deliberate": "deliberate",
        "post_vote": "post_vote",
    })
    builder.add_edge("post_vote", END)
    return builder.compile()


def run_deliberation(
    llm: LLMClient,
    measure: MeasureSpec,
    personas: list,
    seed: int,
) -> DeliberationState:
    graph = build_graph(llm)
    initial: DeliberationState = {
        "measure": measure,
        "personas": personas,
        "statements": [],
        "votes": [],
        "current_round": 0,
        "cost_usd": 0.0,
        "finished": False,
        "seed": seed,
        "mode": "stub" if llm.stub else "live",
        "model_version": "stub" if llm.stub else llm.model_version,
        "prompt_version": PROMPT_VERSION,
    }
    final = graph.invoke(initial, config={"recursion_limit": 50})
    final["cost_usd"] = llm.meter.spent_usd
    return final  # type: ignore[return-value]
