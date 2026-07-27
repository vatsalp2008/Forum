"""Provider routing + Anthropic client contract (no network)."""

import pytest

from backtest.run import run_one
from forum.budget import CostMeter
from forum.llm import (
    AnthropicLLMClient,
    LLMClient,
    LLMError,
    _is_retryable,
    make_llm_client,
)


def _err(status):
    e = RuntimeError(f"status {status}")
    e.status_code = status
    return e


def test_is_retryable_classification():
    assert _is_retryable(_err(429)) is True
    assert _is_retryable(_err(503)) is True
    assert _is_retryable(_err(400)) is False   # no credit / bad request
    assert _is_retryable(_err(404)) is False   # bad model id
    assert _is_retryable(RuntimeError("network reset")) is True  # unknown -> transient


def test_factory_routes_by_provider():
    meter = CostMeter(cap_usd=1.0)
    assert isinstance(make_llm_client("gemini", meter=meter, stub=True), LLMClient)
    assert isinstance(make_llm_client("anthropic", meter=meter, stub=True), AnthropicLLMClient)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        make_llm_client("openai", meter=CostMeter(cap_usd=1.0), stub=True)


def test_anthropic_stub_role_models_and_version():
    llm = AnthropicLLMClient(meter=CostMeter(cap_usd=1.0), stub=True, model="claude-haiku-4-5")
    assert llm.citizen_model == llm.moderator_model == llm.critic_model == "claude-haiku-4-5"
    assert llm.model_version == "anthropic:claude-haiku-4-5"
    # Stub still returns canned JSON regardless of provider.
    resp = llm.generate(model=llm.citizen_model, system="stance", user="stance", json_mode=True)
    assert '"stance"' in resp.text


class _BoomMessages:
    def create(self, *a, **k):
        raise RuntimeError("simulated anthropic 429")


def test_anthropic_live_failure_raises_not_fabricates(monkeypatch):
    monkeypatch.setattr("forum.llm.time.sleep", lambda *_: None)
    llm = AnthropicLLMClient(
        meter=CostMeter(cap_usd=1.0), stub=False, api_key="fake",
        min_interval_s=0.0, max_retries=2,
    )
    llm._client = type("_C", (), {"messages": _BoomMessages()})()
    with pytest.raises(LLMError):
        llm.generate(model="claude-opus-5", system="s", user="u", json_mode=True)


def test_non_retryable_error_fails_fast(monkeypatch):
    """A permanent 400 must abort after ONE attempt, not burn all retries."""
    monkeypatch.setattr("forum.llm.time.sleep", lambda *_: None)
    calls = {"n": 0}

    class _Msgs:
        def create(self, *a, **k):
            calls["n"] += 1
            raise _err(400)

    llm = AnthropicLLMClient(
        meter=CostMeter(cap_usd=1.0), stub=False, api_key="fake",
        min_interval_s=0.0, max_retries=5,
    )
    llm._client = type("_C", (), {"messages": _Msgs()})()
    with pytest.raises(LLMError):
        llm.generate(model="claude-opus-5", system="s", user="u")
    assert calls["n"] == 1


def test_anthropic_stub_backtest_stamps_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = run_one("wa_i1631", n_personas=4, seed=1, stub=True, provider="anthropic")
    assert report.mode == "stub"  # stub mode is provider-agnostic
