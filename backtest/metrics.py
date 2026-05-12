"""Metrics for backtest reporting.

Definitions follow standard prediction-evaluation conventions used in the
political-forecasting literature (e.g. Tetlock 2005). All percentages are in
the 0..100 scale; all probabilities in the 0..1 scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from forum.state import Vote


def predicted_yes_pct(post_votes: list[Vote], threshold: float = 0.5) -> float:
    """Return predicted yes-percentage from post-deliberation stances.

    A persona's post-deliberation stance > threshold counts as a yes vote.
    """
    if not post_votes:
        return 50.0
    yeses = sum(1 for v in post_votes if v.stance > threshold)
    return 100.0 * yeses / len(post_votes)


def predicted_yes_share_weighted(post_votes: list[Vote]) -> float:
    """Stance-weighted predicted yes-share (uses the continuous stance directly)."""
    if not post_votes:
        return 50.0
    return 100.0 * sum(v.stance for v in post_votes) / len(post_votes)


def mean_absolute_error(predicted_pct: float, actual_pct: float) -> float:
    return abs(predicted_pct - actual_pct)


def brier_score(predicted_yes_prob: float, actual_passed: bool) -> float:
    """Brier score: (p - o)^2 where p in [0,1], o in {0,1}."""
    o = 1.0 if actual_passed else 0.0
    return (predicted_yes_prob - o) ** 2


def opinion_change(pre_votes: list[Vote], post_votes: list[Vote]) -> dict:
    """Aggregate pre/post deltas. Personas matched by persona_id."""
    pre_by = {v.persona_id: v for v in pre_votes}
    deltas: list[float] = []
    flips = 0
    for v in post_votes:
        pre = pre_by.get(v.persona_id)
        if pre is None:
            continue
        deltas.append(v.stance - pre.stance)
        if (pre.stance > 0.5) != (v.stance > 0.5):
            flips += 1
    if not deltas:
        return {"mean_abs_delta": 0.0, "flip_rate": 0.0, "n": 0}
    n = len(deltas)
    mean_abs = sum(abs(d) for d in deltas) / n
    return {"mean_abs_delta": mean_abs, "flip_rate": flips / n, "n": n}


@dataclass
class MeasureReport:
    measure_id: str
    n_personas: int
    n_rounds: int
    predicted_yes_pct_threshold: float
    predicted_yes_pct_weighted: float
    actual_yes_pct: float
    mae_threshold: float
    mae_weighted: float
    brier: float
    opinion_change: dict
    cost_usd: float
    seed: int
    model_version: str
    prompt_version: str
    persona_lib_versions: dict
    notes: list[str]

    def render(self) -> str:
        lines = [
            f"# Backtest report: {self.measure_id}",
            "",
            f"- Personas: {self.n_personas}",
            f"- Rounds: {self.n_rounds}",
            f"- Seed: {self.seed}",
            f"- Models: {self.model_version}",
            f"- Prompt version: {self.prompt_version}",
            f"- Persona library versions: {self.persona_lib_versions}",
            f"- Cost: ${self.cost_usd:.4f}",
            "",
            "## Prediction vs. ground truth",
            "",
            f"- Predicted yes (threshold rule): {self.predicted_yes_pct_threshold:.1f}%",
            f"- Predicted yes (weighted): {self.predicted_yes_pct_weighted:.1f}%",
            f"- Actual yes: {self.actual_yes_pct:.1f}%",
            f"- MAE (threshold): {self.mae_threshold:.1f} pts",
            f"- MAE (weighted): {self.mae_weighted:.1f} pts",
            f"- Brier: {self.brier:.4f}",
            "",
            "## Opinion change (pre vs. post deliberation)",
            "",
            f"- Mean |Δstance|: {self.opinion_change.get('mean_abs_delta', 0):.3f}",
            f"- Flip rate: {self.opinion_change.get('flip_rate', 0):.3f}",
            f"- N matched: {self.opinion_change.get('n', 0)}",
            "",
            "## Notes",
            "",
        ]
        for n in self.notes:
            lines.append(f"- {n}")
        return "\n".join(lines)
