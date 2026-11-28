"""Contamination control: strip ballot-measure identifiers from agent-visible text.

The backtest's central confound (methodology §4) is that the model may
*recall* a historical measure's outcome instead of deliberating. The single
strongest recall cue is the measure's identity: its initiative number
("Initiative 1631", "I-1631") and the year it was on the ballot. None of these
may reach an agent. Two entry points:

- ``redact_curated``: for curated measure materials (framing, briefing facts,
  source citations, pro/con arguments). Removes every ballot identifier and
  every year; nothing in a curated policy description needs either.
- ``redact_generated``: for model output shown to *other* agents (the
  moderator's briefing, citizen statements). Removes ballot identifiers plus
  this measure's own number and year, but keeps other numbers, so a citizen can
  still mention the I-5 freeway or a 2030 emissions target.

The contamination probe (backtest/contamination.py) deliberately names the
measure and year; it measures recall and never feeds a deliberation.
"""

from __future__ import annotations

import re

_ID_WORDS = re.compile(
    r"\b(?:initiative|referendum|proposition|prop\.?|ballot\s+measure|measure|question|amendment)"
    r"(?:\s+measure)?\s+(?:no\.?\s*|number\s+|#\s*)?\d+[a-z]?\b",
    re.IGNORECASE,
)
# Short codes such as I-1631 / R-74. Only applied to curated text: in model
# output "I-405" is far more likely to be a freeway than a ballot measure.
_ID_CODES = re.compile(r"\b(?:I|R|EI|SJR|HJR)-\s?\d+\b")
_YEARS = re.compile(r"\b(?:19|20)\d{2}\b")
_PLACEHOLDER = "the measure"


def measure_number(measure_id: str | None) -> str | None:
    """Trailing ballot number of a measure id ("wa_i1631" -> "1631").

    Only numbers of 3+ digits are returned: bare 1-2 digit numbers are too
    common in ordinary text to redact safely.
    """
    m = re.search(r"(\d{3,})$", measure_id or "")
    return m.group(1) if m else None


def _tidy(text: str) -> str:
    t = re.sub(r"\bthe\s+the measure\b", _PLACEHOLDER, text, flags=re.IGNORECASE)
    t = re.sub(r"\(\s*[,;]?\s*\)", "", t)        # "( )" or "(, )" left behind
    t = re.sub(r"\(\s*[,;]\s*", "(", t)
    t = re.sub(r"\s*[,;]\s*\)", ")", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"[ \t]+([,.;:)?!])", r"\1", t)
    t = re.sub(r",\s*$", "", t.strip())
    return t.strip()


def redact_curated(text: str, number: str | None = None) -> str:
    """Strip every ballot identifier and every year from curated material."""
    t = _ID_WORDS.sub(_PLACEHOLDER, text)
    t = _ID_CODES.sub(_PLACEHOLDER, t)
    if number:
        t = re.sub(rf"\b{re.escape(number)}\b", "", t)
    t = _YEARS.sub("", t)
    return _tidy(t)


def redact_generated(text: str, number: str | None = None, year: int | None = None) -> str:
    """Strip ballot identifiers and this measure's own number/year from model output."""
    t = _ID_WORDS.sub(_PLACEHOLDER, text)
    if number:
        t = re.sub(rf"\b(?:[IR]-?\s?)?{re.escape(number)}\b", _PLACEHOLDER, t)
    if year:
        t = re.sub(rf"\b{year}\b", "", t)
    return _tidy(t)
