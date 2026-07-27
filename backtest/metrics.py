"""Metrics for backtest reporting.

Definitions follow standard prediction-evaluation conventions used in the
political-forecasting literature (e.g. Tetlock 2005). All percentages are in
the 0..100 scale; all probabilities in the 0..1 scale.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from forum.state import Vote

# Minimum cell size for a per-segment result to be reported (k-anonymity /
# small-N suppression, matching the persona sampler's K_ANONYMITY_MIN).
SEGMENT_MIN_N = 5


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


def bootstrap_ci_yes_share(
    post_votes: list[Vote],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the weighted predicted yes-share.

    Resamples personas (with replacement) to estimate the sampling variance of
    the predicted yes-share given this persona pool. Returns (low, high) on the
    0..100 scale. Degenerate pools return a point interval.
    """
    if not post_votes:
        return (50.0, 50.0)
    if len(post_votes) == 1:
        v = 100.0 * post_votes[0].stance
        return (v, v)
    rng = random.Random(seed)
    n = len(post_votes)
    stances = [v.stance for v in post_votes]
    shares: list[float] = []
    for _ in range(n_boot):
        sample = [stances[rng.randrange(n)] for _ in range(n)]
        shares.append(100.0 * sum(sample) / n)
    shares.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (shares[lo_idx], shares[hi_idx])


def segment_breakdown(
    personas: list,
    pre_votes: list[Vote],
    post_votes: list[Vote],
    k_min: int = SEGMENT_MIN_N,
) -> dict:
    """Predicted per-demographic-segment breakdown from the deliberation.

    Groups personas along a fixed set of axes and, for each cell, reports the
    weighted predicted yes-share and mean |Δstance|. Cells with fewer than
    k_min personas are suppressed (reported as suppressed, not with numbers) to
    honor the k-anonymity promise in methodology §2/§5.

    This is a *predicted* breakdown (methodology §5). Comparison against
    certified per-segment returns is done separately, only when ground-truth
    segment_results are populated for the measure.
    """
    def axis_value(p, axis: str) -> str:
        if axis == "party_id":
            return p.priors.party_id
        return getattr(p.demographics, axis)

    axes = ["party_id", "age_band", "education", "race_eth"]
    post_by = {v.persona_id: v for v in post_votes}
    out: dict = {}
    for axis in axes:
        cells: dict[str, list] = {}
        for p in personas:
            cells.setdefault(axis_value(p, axis), []).append(p)
        axis_out: dict = {}
        for value, members in sorted(cells.items()):
            ids = {p.persona_id for p in members}
            cell_post = [v for v in post_votes if v.persona_id in ids]
            cell_pre = [v for v in pre_votes if v.persona_id in ids]
            n = len(cell_post)
            if n < k_min:
                axis_out[value] = {"n": n, "suppressed": True}
                continue
            delta = opinion_change(cell_pre, cell_post)
            axis_out[value] = {
                "n": n,
                "pred_yes_pct": predicted_yes_share_weighted(cell_post),
                "mean_abs_delta": delta["mean_abs_delta"],
            }
        out[axis] = axis_out
    return out


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
class SensitivityRow:
    measure_id: str
    actual_yes_pct: float
    seeds: list[int]
    predicted_weighted: list[float]
    mae_weighted: list[float]
    brier: list[float]

    @property
    def mean_predicted(self) -> float:
        return sum(self.predicted_weighted) / len(self.predicted_weighted)

    @property
    def stdev_predicted(self) -> float:
        m = self.mean_predicted
        var = sum((x - m) ** 2 for x in self.predicted_weighted) / max(1, len(self.predicted_weighted) - 1)
        return var ** 0.5

    @property
    def min_predicted(self) -> float:
        return min(self.predicted_weighted)

    @property
    def max_predicted(self) -> float:
        return max(self.predicted_weighted)

    @property
    def mean_mae(self) -> float:
        return sum(self.mae_weighted) / len(self.mae_weighted)

    @property
    def mean_brier(self) -> float:
        return sum(self.brier) / len(self.brier)

    @property
    def ci95_predicted(self) -> tuple[float, float]:
        """Normal-approximation 95% CI of the predicted yes-share across seeds."""
        half = 1.96 * self.stdev_predicted
        return (self.mean_predicted - half, self.mean_predicted + half)


def render_sensitivity_report(rows: list[SensitivityRow], n_personas: int) -> str:
    """Render an aggregate sensitivity report across measures and seeds."""
    lines = [
        "# Sensitivity report",
        "",
        f"N personas per run: {n_personas}",
        f"Seeds per measure: {len(rows[0].seeds) if rows else 0}",
        "",
        "## Aggregate across seeds",
        "",
        "| Measure | Actual | Mean Predicted | 95% CI | Std Dev | Min | Max | Spread | Mean MAE | Mean Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        spread = r.max_predicted - r.min_predicted
        ci_lo, ci_hi = r.ci95_predicted
        lines.append(
            f"| {r.measure_id} | {r.actual_yes_pct:.1f}% "
            f"| {r.mean_predicted:.1f}% | {ci_lo:.1f}–{ci_hi:.1f} | {r.stdev_predicted:.2f} "
            f"| {r.min_predicted:.1f}% | {r.max_predicted:.1f}% "
            f"| {spread:.1f} pts | {r.mean_mae:.2f} | {r.mean_brier:.4f} |"
        )
    lines += ["", "## Per-seed detail", "", "| Measure | Seed | Predicted | MAE | Brier |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        for seed, pred, mae, brier in zip(r.seeds, r.predicted_weighted, r.mae_weighted, r.brier):
            lines.append(f"| {r.measure_id} | {seed} | {pred:.1f}% | {mae:.2f} | {brier:.4f} |")
    lines += [
        "",
        "## Interpretation guide",
        "",
        "- **Low std dev (<2 pts) AND low MAE (<3 pts)**: result is stable. Either real signal OR contamination — run `forum contamination-probe` to disambiguate.",
        "- **Low std dev AND high MAE**: methodology is biased but reproducible. Investigate per-segment errors.",
        "- **High std dev (>5 pts)**: result is noise-dominated. Increase N or rethink methodology.",
        "- **Wide spread relative to MAE**: a single-seed report would be misleading. Always report variance.",
    ]
    return "\n".join(lines)


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
    mode: str
    model_version: str
    prompt_version: str
    persona_lib_versions: dict
    notes: list[str]
    ci_low: float = 0.0
    ci_high: float = 0.0
    segments: dict = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"# Backtest report: {self.measure_id}",
            "",
            f"- Mode: {self.mode.upper()}"
            + ("  ⚠️  STUB — pseudo-random, not a real result" if self.mode == "stub" else ""),
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
            f"- Predicted yes (weighted): {self.predicted_yes_pct_weighted:.1f}% "
            f"(95% CI {self.ci_low:.1f}–{self.ci_high:.1f}, bootstrap over personas)",
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
        ]
        lines += self._render_segments()
        lines += ["## Notes", ""]
        for n in self.notes:
            lines.append(f"- {n}")
        return "\n".join(lines)

    def _render_segments(self) -> list[str]:
        if not self.segments:
            return []
        lines = [
            "## Predicted per-segment breakdown",
            "",
            f"Cells with fewer than {SEGMENT_MIN_N} personas are suppressed "
            "(k-anonymity). Predicted, not certified — see methodology §5.",
            "",
        ]
        for axis, cells in self.segments.items():
            lines += [f"### {axis}", "", "| Segment | N | Predicted yes | Mean |Δstance| |",
                      "|---|---:|---:|---:|"]
            for value, cell in cells.items():
                if cell.get("suppressed"):
                    lines.append(f"| {value} | {cell['n']} | suppressed (N<{SEGMENT_MIN_N}) | — |")
                else:
                    lines.append(
                        f"| {value} | {cell['n']} | {cell['pred_yes_pct']:.1f}% "
                        f"| {cell['mean_abs_delta']:.3f} |"
                    )
            lines.append("")
        return lines


@dataclass
class MultiGroupReport:
    """Aggregate of several independent deliberating groups on one measure.

    A single group of ~12 is noisy; running G independent groups and pooling
    them stabilizes the point estimate and yields a between-group variance —
    a second, complementary uncertainty measure alongside the within-group
    bootstrap CI (methodology §5, "sensitivity range across persona seeds").
    """

    measure_id: str
    mode: str
    n_groups: int
    group_size: int
    seed: int
    actual_yes_pct: float
    n_rounds: int
    group_predicted: list[float]     # per-group weighted predicted yes-share
    pooled_predicted: float          # pooled over all personas
    pooled_ci: tuple[float, float]   # bootstrap CI over pooled personas
    cost_usd: float
    model_version: str
    prompt_version: str
    persona_lib_versions: dict
    opinion_change: dict
    segments: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def between_group_mean(self) -> float:
        return sum(self.group_predicted) / len(self.group_predicted)

    @property
    def between_group_stdev(self) -> float:
        m = self.between_group_mean
        n = len(self.group_predicted)
        var = sum((x - m) ** 2 for x in self.group_predicted) / max(1, n - 1)
        return var ** 0.5

    @property
    def between_group_ci95(self) -> tuple[float, float]:
        half = 1.96 * self.between_group_stdev
        return (self.between_group_mean - half, self.between_group_mean + half)

    @property
    def mae_weighted(self) -> float:
        return abs(self.pooled_predicted - self.actual_yes_pct)

    def render(self) -> str:
        bg_lo, bg_hi = self.between_group_ci95
        lines = [
            f"# Multi-group backtest report: {self.measure_id}",
            "",
            f"- Mode: {self.mode.upper()}"
            + ("  ⚠️  STUB — pseudo-random, not a real result" if self.mode == "stub" else ""),
            f"- Groups: {self.n_groups} × {self.group_size} personas",
            f"- Rounds: {self.n_rounds}",
            f"- Seed (base): {self.seed}",
            f"- Models: {self.model_version}",
            f"- Prompt version: {self.prompt_version}",
            f"- Persona library versions: {self.persona_lib_versions}",
            f"- Cost: ${self.cost_usd:.4f}",
            "",
            "## Prediction vs. ground truth",
            "",
            f"- Pooled predicted yes (weighted): {self.pooled_predicted:.1f}% "
            f"(95% CI {self.pooled_ci[0]:.1f}–{self.pooled_ci[1]:.1f}, bootstrap over personas)",
            f"- Between-group mean: {self.between_group_mean:.1f}% "
            f"(95% CI {bg_lo:.1f}–{bg_hi:.1f}, σ={self.between_group_stdev:.2f} across {self.n_groups} groups)",
            f"- Actual yes: {self.actual_yes_pct:.1f}%",
            f"- MAE (pooled weighted): {self.mae_weighted:.1f} pts",
            "",
            "## Per-group predictions",
            "",
            "| Group | Predicted yes |",
            "|---:|---:|",
        ]
        for i, p in enumerate(self.group_predicted):
            lines.append(f"| {i} | {p:.1f}% |")
        lines += [
            "",
            "## Opinion change (pooled, pre vs. post)",
            "",
            f"- Mean |Δstance|: {self.opinion_change.get('mean_abs_delta', 0):.3f}",
            f"- Flip rate: {self.opinion_change.get('flip_rate', 0):.3f}",
            f"- N matched: {self.opinion_change.get('n', 0)}",
            "",
        ]
        lines += MeasureReport._render_segments(self)  # reuse segment renderer
        lines += ["## Notes", ""]
        for n in self.notes:
            lines.append(f"- {n}")
        return "\n".join(lines)
