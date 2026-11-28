"""FORUM_GEMINI_MODEL picks the model for every role, and the probe follows it."""

from pathlib import Path

import pytest

from backtest.contamination import run_contamination_probe
from forum.budget import CostMeter
from forum.llm import AnthropicLLMClient, LLMClient


def test_forum_gemini_model_sets_every_role(monkeypatch):
    monkeypatch.setenv("FORUM_GEMINI_MODEL", "gemma-4-31b-it")
    llm = LLMClient(meter=CostMeter(), stub=True)
    assert llm.citizen_model == llm.moderator_model == llm.critic_model == "gemma-4-31b-it"
    assert llm.model_version == "gemini:gemma-4-31b-it"


def test_default_models_without_the_override():
    llm = LLMClient(meter=CostMeter(), stub=True)
    assert llm.model_version == "gemini:gemini-2.5-flash-lite"


def test_explicit_role_model_beats_the_override(monkeypatch):
    monkeypatch.setenv("FORUM_GEMINI_MODEL", "gemma-4-31b-it")
    llm = LLMClient(meter=CostMeter(), stub=True, critic_model="gemini-2.5-flash-lite")
    assert (llm.citizen_model, llm.critic_model) == ("gemma-4-31b-it", "gemini-2.5-flash-lite")


def test_unknown_model_fails_before_any_call(monkeypatch):
    monkeypatch.setenv("FORUM_GEMINI_MODEL", "gemma-4-31b-typo")
    with pytest.raises(ValueError, match="PRICING"):
        LLMClient(meter=CostMeter(), stub=True)
    with pytest.raises(ValueError, match="PRICING"):
        AnthropicLLMClient(meter=CostMeter(), stub=True, model="claude-typo")


def test_contamination_probe_audits_the_configured_model(monkeypatch):
    monkeypatch.setenv("FORUM_GEMINI_MODEL", "gemma-4-26b-a4b-it")
    run_contamination_probe(measure_ids=["wa_i1631"], seeds=[1], stub=True)
    report = next(Path("backtest/runs").rglob("contamination.md")).read_text()
    assert "- Probed model: gemma-4-26b-a4b-it" in report
