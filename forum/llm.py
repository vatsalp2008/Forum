"""Thin, provider-aware LLM client wrapper.

FORUM supports two model families so runs can be audited for cross-model
robustness (methodology §5.6):

- "gemini"    — Google Gemini via google-generativeai (the v0/v1 default)
- "anthropic" — Claude via the anthropic SDK (v2 cross-model family)

Both providers share one interface: a `generate(...)` method, a `.meter`, a
`.stub` flag, and `.citizen_model` / `.moderator_model` / `.critic_model`
attributes the graph reads (so the graph is provider-agnostic). A "stub" mode
returns deterministic canned responses so the whole pipeline runs end-to-end
without any API key.
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

# ---- Gemini defaults ----
# gemini-2.5-flash-lite: fast, free-tier-eligible, native JSON mode. Verify
# availability for your key with `genai.list_models()` if a live run 404s.
DEFAULT_CITIZEN_MODEL = "gemini-2.5-flash-lite"
DEFAULT_MODERATOR_MODEL = "gemini-2.5-flash-lite"
DEFAULT_CRITIC_MODEL = "gemini-2.5-flash-lite"

# Gemma models do not support the JSON-mode mime type that Gemini does.
JSON_MODE_SUPPORTED_MODELS = {
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro",
    "gemini-2.0-flash", "gemini-2.0-flash-lite",
}

# ---- Anthropic / Claude defaults ----
# Default to the most capable model; override per role via FORUM_ANTHROPIC_MODEL
# (e.g. "claude-haiku-4-5" for a cheap, quota-friendly sweep). Sampling params
# are NOT sent (they are rejected on Opus 5 / Sonnet 5). Must be priced in
# forum/budget.py PRICING or the meter raises.
DEFAULT_ANTHROPIC_MODEL = os.environ.get("FORUM_ANTHROPIC_MODEL", "claude-opus-5")
ANTHROPIC_MAX_TOKENS = 4096

PROVIDER_GEMINI = "gemini"
PROVIDER_ANTHROPIC = "anthropic"


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

    Open-weight models (and occasionally Claude) wrap JSON in markdown fences
    even when asked not to. This strips them before json.loads.
    """
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1:] if nl != -1 else s[3:]
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


class _LLMBase:
    """Shared throttle + retry + fail-loud skeleton for all providers."""

    provider = "base"

    def __init__(
        self,
        meter: CostMeter,
        stub: bool = False,
        api_key: str | None = None,
        min_interval_s: float | None = None,
        max_retries: int | None = None,
        citizen_model: str = DEFAULT_CITIZEN_MODEL,
        moderator_model: str = DEFAULT_MODERATOR_MODEL,
        critic_model: str = DEFAULT_CRITIC_MODEL,
    ) -> None:
        self.meter = meter
        self.stub = stub
        self._api_key = api_key
        self._client: Any = None
        self.citizen_model = citizen_model
        self.moderator_model = moderator_model
        self.critic_model = critic_model
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

    @property
    def model_version(self) -> str:
        """Stable stamp of the models used, for report reproducibility."""
        models = sorted({self.citizen_model, self.moderator_model, self.critic_model})
        return f"{self.provider}:" + "+".join(models)

    def _init_client(self) -> None:  # pragma: no cover - provider specific
        raise NotImplementedError

    def _raw_generate(
        self, model: str, system: str, user: str, json_mode: bool, temperature: float
    ) -> tuple[str, int, int]:  # pragma: no cover - provider specific
        """Return (text, input_tokens, output_tokens); raise on failure."""
        raise NotImplementedError

    def _throttle(self) -> None:
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

        last_err: Exception | None = None
        text, in_tok, out_tok = "", 0, 0
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                text, in_tok, out_tok = self._raw_generate(
                    model, system, user, json_mode, temperature
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                # Honor a server-suggested retry delay (rate limits carry one);
                # otherwise exponential backoff. Free-tier limits need seconds.
                suggested = _retry_delay_seconds(e)
                time.sleep(suggested if suggested is not None else 2.0 * (2 ** attempt))
        if last_err is not None:
            # Live mode aborts loudly rather than fabricating data (stub=True
            # is the only path that returns canned output).
            raise LLMError(
                f"{self.provider} LLM call to {model!r} failed after "
                f"{self.max_retries} attempts: {type(last_err).__name__}: {last_err}"
            ) from last_err
        if json_mode:
            text = strip_code_fences(text)
        self.meter.charge(model, in_tok, out_tok)
        return LLMResponse(text=text, input_tokens=in_tok, output_tokens=out_tok)


class LLMClient(_LLMBase):
    """Google Gemini client (default provider)."""

    provider = PROVIDER_GEMINI

    def __init__(self, meter: CostMeter, stub: bool = False, api_key: str | None = None, **kw):
        super().__init__(meter, stub=stub, api_key=api_key or os.environ.get("GEMINI_API_KEY"), **kw)

    def _init_client(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set and stub=False. "
                "Either set GEMINI_API_KEY or pass stub=True."
            )
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=self._api_key)
        self._client = genai

    def _raw_generate(self, model, system, user, json_mode, temperature):
        full_prompt = system + "\n\n" + user
        m = self._client.GenerativeModel(model)
        kwargs: dict[str, Any] = {"temperature": temperature}
        if json_mode and model in JSON_MODE_SUPPORTED_MODELS:
            kwargs["response_mime_type"] = "application/json"
        resp = m.generate_content(full_prompt, generation_config=kwargs)
        text = (resp.text or "").strip()
        return text, _approx_tokens(full_prompt), _approx_tokens(text)


class AnthropicLLMClient(_LLMBase):
    """Anthropic Claude client (cross-model family, methodology §5.6)."""

    provider = PROVIDER_ANTHROPIC

    def __init__(
        self,
        meter: CostMeter,
        stub: bool = False,
        api_key: str | None = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        **kw,
    ):
        # One Claude model fills all three roles by default; override via `model`.
        kw.setdefault("citizen_model", model)
        kw.setdefault("moderator_model", model)
        kw.setdefault("critic_model", model)
        super().__init__(
            meter, stub=stub, api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), **kw
        )

    def _init_client(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set and stub=False. "
                "Either set ANTHROPIC_API_KEY or pass stub=True."
            )
        import anthropic  # late import so the module loads without the SDK

        self._client = anthropic.Anthropic(api_key=self._api_key)

    def _raw_generate(self, model, system, user, json_mode, temperature):
        # Sampling params are omitted: temperature/top_p/top_k are rejected on
        # Opus 5 / Sonnet 5 / Fable 5. JSON is requested via the prompt (the
        # templates already say "Output only the JSON object") and fences are
        # stripped by the shared generate().
        resp = self._client.messages.create(
            model=model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
        return text, resp.usage.input_tokens, resp.usage.output_tokens


def make_llm_client(
    provider: str,
    meter: CostMeter,
    stub: bool = False,
    **kw,
) -> _LLMBase:
    """Construct the client for a provider ("gemini" | "anthropic")."""
    if provider == PROVIDER_GEMINI:
        return LLMClient(meter=meter, stub=stub, **kw)
    if provider == PROVIDER_ANTHROPIC:
        return AnthropicLLMClient(meter=meter, stub=stub, **kw)
    raise ValueError(f"Unknown provider {provider!r}. Use 'gemini' or 'anthropic'.")


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
        if '"stance"' in user or "stance" in system:
            obj = {
                "stance": round(rng.uniform(0.2, 0.8), 2),
                "confidence": round(rng.uniform(0.4, 0.9), 2),
                "rationale": "stub rationale (no LLM in this run).",
            }
        elif '"knows"' in user:
            obj = {"knows": False, "yes_pct": None, "confidence": 0.0,
                   "note": "stub probe; no LLM in this run."}
        elif '"flagged"' in user:
            obj = {"flagged": False, "note": ""}
        else:
            obj = {"stub": True}
        text = json.dumps(obj)
    else:
        text = "[stub statement; no LLM in this run]"
    return LLMResponse(
        text=text,
        input_tokens=_approx_tokens(system + user),
        output_tokens=_approx_tokens(text),
    )
