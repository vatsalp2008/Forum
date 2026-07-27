"""Cost tracking and budget enforcement.

Approximate Gemini pricing per million tokens (free-tier-eligible models).
Update as Google changes pricing; the numbers here are for estimation,
not billing.
"""

from __future__ import annotations

from dataclasses import dataclass

# Approximate USD per 1M tokens. Verify against current Google pricing.
PRICING = {
    # Approximate USD per 1M tokens. Gemma models are free on AI Studio's
    # free tier; we use small nominal numbers so the meter still tracks
    # relative usage. Verify against current Google pricing.
    "gemma-3-1b-it":       {"input": 0.0, "output": 0.0},
    "gemma-3-4b-it":       {"input": 0.0, "output": 0.0},
    "gemma-3-12b-it":      {"input": 0.0, "output": 0.0},
    "gemma-3-27b-it":      {"input": 0.0, "output": 0.0},
    "gemma-4-26b-a4b-it":  {"input": 0.0, "output": 0.0},
    "gemma-4-31b-it":      {"input": 0.0, "output": 0.0},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash":      {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro":        {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash":      {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    # Anthropic / Claude (USD per 1M tokens). Verify against current pricing.
    "claude-opus-5":         {"input": 5.00, "output": 25.00},
    "claude-opus-4-8":       {"input": 5.00, "output": 25.00},
    "claude-sonnet-5":       {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5":      {"input": 1.00, "output": 5.00},
}

DEFAULT_DELIBERATION_BUDGET_USD = 5.0
DEFAULT_BACKTEST_RUN_BUDGET_USD = 15.0


@dataclass
class BudgetExceeded(Exception):
    spent_usd: float
    cap_usd: float

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"BudgetExceeded: spent ${self.spent_usd:.4f}, cap ${self.cap_usd:.4f}"
        )


class CostMeter:
    def __init__(self, cap_usd: float = DEFAULT_DELIBERATION_BUDGET_USD) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.calls: list[tuple[str, int, int, float]] = []  # (model, in_tok, out_tok, usd)

    def charge(self, model: str, input_tokens: int, output_tokens: int) -> float:
        rates = PRICING.get(model)
        if rates is None:
            raise ValueError(f"Unknown model {model!r} for pricing.")
        cost = (
            input_tokens * rates["input"] / 1_000_000
            + output_tokens * rates["output"] / 1_000_000
        )
        self.spent_usd += cost
        self.calls.append((model, input_tokens, output_tokens, cost))
        if self.spent_usd > self.cap_usd:
            raise BudgetExceeded(self.spent_usd, self.cap_usd)
        return cost

    def summary(self) -> dict:
        by_model: dict[str, dict] = {}
        for model, in_t, out_t, usd in self.calls:
            d = by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
            d["calls"] += 1
            d["in"] += in_t
            d["out"] += out_t
            d["usd"] += usd
        return {"total_usd": self.spent_usd, "by_model": by_model}
