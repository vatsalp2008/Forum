"""Thin LLM client wrapper.

For v0 we use Google Gemini via google-generativeai. The wrapper supports
a "stub" mode that returns deterministic canned responses, used by the
backtest harness in stub mode and by tests. Stub mode lets the entire
pipeline run end-to-end without an API key.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from forum.budget import CostMeter

# gemini-2.5-flash-lite: fast, free-tier-eligible, and supports native JSON
# mode (see JSON_MODE_SUPPORTED_MODELS) which makes vote parsing reliable.
# The retired gemma-3-*-it names and the flaky free-tier gemma-4-* models
# were dropped as defaults; verify availability for your key with
# `genai.list_models()` if a live run 404s on the model name.
DEFAULT_CITIZEN_MODEL = "gemini-2.5-flash-lite"
DEFAULT_MODERATOR_MODEL = "gemini-2.5-flash-lite"
DEFAULT_CRITIC_MODEL = "gemini-2.5-flash-lite"

# Gemma models do not support the JSON-mode mime type that Gemini does.
# We instead instruct via prompt and parse robustly (see strip_code_fences).
JSON_MODE_SUPPORTED_MODELS = {
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro",
    "gemini-2.0-flash", "gemini-2.0-flash-lite",
}


class LLMError(RuntimeError):
    """Raised when a live LLM call fails after all retries.

    We deliberately do NOT degrade to a stub response in live mode: a research
    artifact must never silently emit fabricated data into a "real" run. Callers
    that want stub data must construct the client with stub=True explicitly.
    """


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


def _approx_tokens(text: str) -> int:
    """Cheap token approximation (4 chars per token)."""
    return max(1, len(text) // 4)


def strip_code_fences(text: str) -> str:
    """Strip surrounding ```json ... ``` or ``` ... ``` fences if present.

    Gemma and other open-weight models often wrap JSON in markdown fences
    even when asked not to. This strips them before json.loads.
    """
    s = text.strip()
    if s.startswith("```"):
        # remove opening fence (with optional language tag)
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        else:
            s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


# Free-tier defaults. Requests-per-minute limits are low, so we space live
# calls out and back off generously on 429/503/504. Override via env for
# paid tiers: FORUM_LLM_MIN_INTERVAL_S, FORUM_LLM_MAX_RETRIES.
DEFAULT_MIN_CALL_INTERVAL_S = 4.0
DEFAULT_MAX_RETRIES = 5


def _retry_delay_seconds(err: Exception) -> float | None:
    """Extract a server-suggested retry delay (e.g. 'retry_delay { seconds: 2 }')."""
    m = re.search(r"seconds:\s*(\d+)", str(err))
    return float(m.group(1)) if m else None


class LLMClient:
    def __init__(
        self,
        meter: CostMeter,
        stub: bool = False,
        api_key: str | None = None,
        min_interval_s: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.meter = meter
        self.stub = stub
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client: Any = None
        self.min_interval_s = (
            min_interval_s if min_interval_s is not None
            else float(os.environ.get("FORUM_LLM_MIN_INTERVAL_S", DEFAULT_MIN_CALL_INTERVAL_S))
        )
        self.max_retries = (
            max_retries if max_retries is not None
            else int(os.environ.get("FORUM_LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES))
        )
        self._last_call_ts: float = 0.0
        if not self.stub:
            self._init_client()

    def _init_client(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set and stub=False. "
                "Either set GEMINI_API_KEY or pass stub=True."
            )
        # Late import so the module loads without google-generativeai installed.
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=self._api_key)
        self._client = genai

    def _throttle(self) -> None:
        """Space out live calls to stay under free-tier requests-per-minute."""
        if self.min_interval_s <= 0:
            return
        wait = self.min_interval_s - (time.monotonic() - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()

    def generate(
        self,
        model: str,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> LLMResponse:
        if self.stub:
            return _stub_generate(system, user, json_mode, seed=seed)

        full_prompt = system + "\n\n" + user
        in_tok = _approx_tokens(full_prompt)
        m = self._client.GenerativeModel(model)
        kwargs: dict[str, Any] = {"temperature": temperature}
        if json_mode and model in JSON_MODE_SUPPORTED_MODELS:
            kwargs["response_mime_type"] = "application/json"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = m.generate_content(full_prompt, generation_config=kwargs)
                last_err = None
                break
            except Exception as e:
                last_err = e
                # Honor a server-suggested retry delay (rate limits carry one);
                # otherwise exponential backoff. Free-tier RPM limits need
                # seconds, not milliseconds.
                suggested = _retry_delay_seconds(e)
                backoff = suggested if suggested is not None else 2.0 * (2 ** attempt)
                time.sleep(backoff)
        if last_err is not None:
            # Live mode (we return early above when self.stub). Quota /
            # rate-limit / network errors abort the run loudly rather than
            # silently fabricating data. Use stub=True for an offline run.
            raise LLMError(
                f"LLM call to {model!r} failed after {self.max_retries} attempts: "
                f"{type(last_err).__name__}: {last_err}"
            ) from last_err
        text = (resp.text or "").strip()
        if json_mode:
            text = strip_code_fences(text)
        out_tok = _approx_tokens(text)
        self.meter.charge(model, in_tok, out_tok)
        return LLMResponse(text=text, input_tokens=in_tok, output_tokens=out_tok)


# ---------- Stub mode ----------

def _stub_generate(
    system: str, user: str, json_mode: bool, seed: int | None = None
) -> LLMResponse:
    """Deterministic stub for testing the pipeline without an API key.

    Mixes seed and prompt content so different prompts produce different
    canned responses while remaining reproducible.
    """
    base = (seed or 0) ^ (hash(system + user) & 0xFFFFFFFF)
    rng = random.Random(base)
    if json_mode:
        # Heuristic: detect which JSON schema is expected.
        if '"stance"' in user or "stance" in system:
            obj = {
                "stance": round(rng.uniform(0.2, 0.8), 2),
                "confidence": round(rng.uniform(0.4, 0.9), 2),
                "rationale": "stub rationale (no LLM in this run).",
            }
        elif '"knows"' in user:
            # Contamination probe: stub honestly reports no prior knowledge.
            obj = {"knows": False, "yes_pct": None, "confidence": 0.0,
                   "note": "stub probe; no LLM in this run."}
        elif '"flagged"' in user:
            obj = {"flagged": False, "note": ""}
        else:
            obj = {"stub": True}
        text = json.dumps(obj)
    else:
        text = "[stub statement; no LLM in this run]"
    return LLMResponse(text=text, input_tokens=_approx_tokens(system + user), output_tokens=_approx_tokens(text))
