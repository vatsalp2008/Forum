"""Deliberation state — the LangGraph state object."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from pydantic import BaseModel, Field

from personas.schema import Persona


class Vote(BaseModel):
    """A single persona's stance on the policy question."""

    persona_id: str
    round: int  # 0 = pre-deliberation, N = post-deliberation
    stance: float = Field(ge=0.0, le=1.0)  # 1.0 = strongly support, 0.0 = strongly oppose
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str  # short, model-generated


class Statement(BaseModel):
    """A single statement made during deliberation."""

    speaker_id: str  # persona_id or "moderator"
    round: int
    text: str
    issued_at: datetime
    flagged_by_critic: bool = False
    critic_note: str | None = None


class MeasureSpec(BaseModel):
    """The policy question being deliberated."""

    measure_id: str
    title: str
    framing: str  # neutral framing presented to agents
    briefing: str  # balanced briefing material (used in briefing_node)
    briefing_sources: list[str]  # citations
    pro_arguments: list[str]
    con_arguments: list[str]
    n_rounds: int = 5


class DeliberationState(TypedDict, total=False):
    """LangGraph state. Mutated by node functions."""

    measure: MeasureSpec
    personas: list[Persona]
    statements: list[Statement]
    votes: list[Vote]
    current_round: int
    cost_usd: float
    finished: bool
    seed: int
    model_version: str
    prompt_version: str
