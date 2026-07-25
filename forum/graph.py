"""Deliberative Polling protocol implemented as a LangGraph workflow.

Nodes:
    pre_vote      -> initial stance per persona (DP phase 1)
    briefing      -> moderator presents balanced briefing (DP phase 2)
    deliberate    -> R rounds of speaker-selection + statements (DP phase 3)
    post_vote     -> final stance per persona (DP phase 4)
    synthesize    -> output report
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph

from forum.llm import (
    DEFAULT_CITIZEN_MODEL,
    DEFAULT_CRITIC_MODEL,
    DEFAULT_MODERATOR_MODEL,
    LLMClient,
)
from forum.prompts import (
    CITIZEN_DELIBERATE_PROMPT,
    CITIZEN_SYSTEM_PROMPT,
    CRITIC_FACT_CHECK_PROMPT,
    MODERATOR_BRIEFING_PROMPT,
    PROMPT_VERSION,
    VOTE_PROMPT,
    render_persona_backstory,
)
from forum.state import DeliberationState, MeasureSpec, Statement, Vote


# ---------- helpers ----------

def _vote_for_persona(
    llm: LLMClient,
    persona,
    measure: MeasureSpec,
    round_num: int,
    seed: int,
    recent_statements_text: str = "",
) -> Vote:
    system = CITIZEN_SYSTEM_PROMPT + "\n\n" + render_persona_backstory(persona)
    if recent_statements_text:
        system += "\n\nRecent deliberation:\n" + recent_statements_text
    resp = llm.generate(
        model=DEFAULT_CITIZEN_MODEL,
        system=system,
        user=VOTE_PROMPT.format(framing=measure.framing),
        json_mode=True,
        temperature=0.6,
        seed=seed + hash(persona.persona_id) % 100_000,
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


def _format_recent_statements(statements: list[Statement], limit: int = 6) -> str:
    if not statements:
        return "(no statements yet)"
    recent = statements[-limit:]
    lines = []
    for s in recent:
        flag = " [flagged]" if s.flagged_by_critic else ""
        lines.append(f"- {s.speaker_id} (round {s.round}){flag}: {s.text}")
    return "\n".join(lines)


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
        resp = llm.generate(
            model=DEFAULT_MODERATOR_MODEL,
            system="You are an impartial deliberation moderator.",
            user=MODERATOR_BRIEFING_PROMPT.format(
                title=measure.title,
                framing=measure.framing,
                sources="\n".join(f"- {s}" for s in measure.briefing_sources),
                pro="\n".join(f"- {a}" for a in measure.pro_arguments),
                con="\n".join(f"- {a}" for a in measure.con_arguments),
            ),
            temperature=0.4,
            seed=seed,
        )
        statements.append(
            Statement(
                speaker_id="moderator",
                round=0,
                text=resp.text or "[empty briefing]",
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
        votes = list(state.get("votes", []))
        seed = state.get("seed", 0)
        current_round = state.get("current_round", 0) + 1
        rng = random.Random(seed + current_round)

        # Speaker selection: pick K speakers per round; round-robin biased by speaking count.
        speak_counts = {p.persona_id: 0 for p in personas}
        for s in statements:
            if s.speaker_id != "moderator":
                speak_counts[s.speaker_id] = speak_counts.get(s.speaker_id, 0) + 1
        eligible_sorted = sorted(personas, key=lambda p: (speak_counts[p.persona_id], p.persona_id))
        speakers_this_round = eligible_sorted[: max(1, len(personas) // 3)]
        rng.shuffle(speakers_this_round)

        for speaker in speakers_this_round:
            recent_text = _format_recent_statements(statements)
            user_prompt = CITIZEN_DELIBERATE_PROMPT.format(
                round_num=current_round,
                total_rounds=measure.n_rounds,
                recent_statements=recent_text,
            )
            resp = llm.generate(
                model=DEFAULT_CITIZEN_MODEL,
                system=CITIZEN_SYSTEM_PROMPT + "\n\n" + render_persona_backstory(speaker),
                user=user_prompt,
                temperature=0.7,
                seed=seed + current_round * 1000 + hash(speaker.persona_id) % 100_000,
            )
            text = resp.text or "[empty]"

            # Critic fact-check (cheap; on each statement)
            critic = llm.generate(
                model=DEFAULT_CRITIC_MODEL,
                system="You are a fact-checker. Be precise; do not flag normative disagreement.",
                user=CRITIC_FACT_CHECK_PROMPT.format(
                    statement=text,
                    sources="\n".join(measure.briefing_sources),
                ),
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
        recent = _format_recent_statements(statements, limit=20)
        for p in personas:
            votes.append(
                _vote_for_persona(
                    llm, p, measure,
                    round_num=measure.n_rounds + 1,
                    seed=seed,
                    recent_statements_text=recent,
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
        "model_version": (
            "stub" if llm.stub
            else f"{DEFAULT_CITIZEN_MODEL}+{DEFAULT_MODERATOR_MODEL}"
        ),
        "prompt_version": PROMPT_VERSION,
    }
    final = graph.invoke(initial, config={"recursion_limit": 50})
    final["cost_usd"] = llm.meter.spent_usd
    return final  # type: ignore[return-value]
