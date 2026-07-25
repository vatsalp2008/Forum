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
import time
from dataclasses import dataclass
from typing import Any

from forum.budget import CostMeter

DEFAULT_CITIZEN_MODEL = "gemma-3-12b-it"
DEFAULT_MODERATOR_MODEL = "gemma-3-27b-it"
DEFAULT_CRITIC_MODEL = "gemma-3-12b-it"

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


class LLMClient:
    def __init__(
        self,
        meter: CostMeter,
        stub: bool = False,
        api_key: str | None = None,
    ) -> None:
        self.meter = meter
        self.stub = stub
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client: Any = None
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
        for attempt in range(3):
            try:
                resp = m.generate_content(full_prompt, generation_config=kwargs)
                last_err = None
                break
            except Exception as e:
                last_err = e
                # brief backoff for transient rate-limit / 5xx
                time.sleep(0.5 * (attempt + 1))
        if last_err is not None:
            # Live mode (we return early above when self.stub). Quota /
            # rate-limit / network errors abort the run loudly rather than
            # silently fabricating data. Use stub=True for an offline run.
            raise LLMError(
                f"LLM call to {model!r} failed after 3 attempts: "
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
        elif '"flagged"' in user:
            obj = {"flagged": False, "note": ""}
        else:
            obj = {"stub": True}
        text = json.dumps(obj)
    else:
        text = "[stub statement; no LLM in this run]"
    return LLMResponse(text=text, input_tokens=_approx_tokens(system + user), output_tokens=_approx_tokens(text))
