"""Output-side refusal: the binding refusal layer.

Per docs/methodology.md §6, FORUM refuses certain output types regardless of
input framing. This module defines the patterns and a check function used by
the Output Synthesizer before any deliberation result is emitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

# Substrings that, if present in the framing or the request, should trigger refusal.
REFUSAL_PATTERNS = [
    r"\bpredict\b[^.]{0,40}\b(election|vote|ballot|measure|outcome)\b",
    r"\bforecast\b[^.]{0,40}\b(election|vote|ballot|measure)\b",
    r"\b(persuasive|persuade) (messaging|message|messages|argument)\b",
    r"\bgenerate\b[^.]{0,40}\bmessaging\b",
    r"\bhow (do|can) (we|you|i) (convert|persuade|win over)\b",
    r"\btarget(ed|ing)?\b[^.]{0,20}\b(demographic|group|voters?|audience)\b",
    r"\bvoter (suppression|targeting|microtargeting)\b",
    r"\bcampaign (messaging|strategy|tactics|targeting)\b",
    r"\boptimize\b[^.]{0,30}\b(messaging|message|persuasion)\b",
]

# Window before a named live election in which prediction-style framings
# are refused regardless of input wording.
PRE_ELECTION_REFUSAL_WINDOW_DAYS = 60


@dataclass
class RefusalResult:
    refused: bool
    reason: str | None = None


class RefusalError(RuntimeError):
    """Raised when the pipeline refuses a request or an output.

    Per docs/methodology.md §6 the refusal layer is *binding*: a refused
    framing halts before any LLM cost, and a refused output blocks the report
    from being written. Carries the originating RefusalResult for callers.
    """

    def __init__(self, result: RefusalResult) -> None:
        self.result = result
        super().__init__(result.reason or "refused")


def check_request(framing: str, vote_date: date | None = None) -> RefusalResult:
    """Apply input-side and date-based refusal checks.

    framing: the policy framing or request text
    vote_date: date of the named live vote, if applicable
    """
    lowered = framing.lower()
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, lowered):
            return RefusalResult(
                refused=True,
                reason=(
                    "Request matches a category FORUM refuses (election prediction, "
                    "persuasive-messaging generation, or demographic targeting). "
                    "See docs/methodology.md §6 and docs/aup.md."
                ),
            )

    if vote_date is not None:
        days_to_vote = (vote_date - date.today()).days
        if 0 <= days_to_vote <= PRE_ELECTION_REFUSAL_WINDOW_DAYS:
            return RefusalResult(
                refused=True,
                reason=(
                    f"FORUM refuses deliberations on named live measures within "
                    f"{PRE_ELECTION_REFUSAL_WINDOW_DAYS} days of the vote "
                    f"(found: {days_to_vote} days). See docs/methodology.md §6."
                ),
            )

    return RefusalResult(refused=False)


def check_output(report_text: str) -> RefusalResult:
    """Output-side refusal: catches generated content that looks like
    persuasive messaging or election forecasting even if the input was clean."""
    lowered = report_text.lower()
    output_patterns = [
        r"\bmessaging recommendation\b",
        r"\bsuggested talking points\b",
        r"\bcampaign should\b",
        r"\bto persuade (this|that|these|those|target) (group|demographic|voters?)\b",
        r"\bprobability of (passage|passing) (is|of)\b",
    ]
    for pattern in output_patterns:
        if re.search(pattern, lowered):
            return RefusalResult(
                refused=True,
                reason=(
                    "Output contains content the system refuses to emit "
                    "(messaging guidance, election forecast, or targeting). "
                    "See docs/methodology.md §6."
                ),
            )
    return RefusalResult(refused=False)
