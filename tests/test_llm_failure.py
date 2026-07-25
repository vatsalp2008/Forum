"""Live-mode LLM failures must abort loudly, never fabricate data.

A research artifact must never silently emit stub data into a "real" run.
These tests pin that contract: in live mode a failing LLM call raises
LLMError; only an explicit stub client returns canned data. We also check
that reports self-identify their mode.
"""

import pytest

from backtest.run import run_one
from forum.budget import CostMeter
from forum.llm import LLMClient, LLMError


class _BoomModel:
    def generate_content(self, *args, **kwargs):
        raise RuntimeError("simulated quota exhaustion")


def _live_client_that_fails() -> LLMClient:
    meter = CostMeter(cap_usd=1.0)
    llm = LLMClient(meter=meter, stub=False, api_key="fake-key")
    # Replace the google-generativeai client with one that always errors.
    llm._client = type("_G", (), {"GenerativeModel": lambda self, m: _BoomModel()})()
    return llm


def test_live_llm_failure_raises_not_fabricates():
    llm = _live_client_that_fails()
    with pytest.raises(LLMError):
        llm.generate(model="gemma-3-12b-it", system="s", user="u", json_mode=True)


def test_stub_client_still_returns_canned_data():
    llm = LLMClient(meter=CostMeter(cap_usd=1.0), stub=True)
    resp = llm.generate(model="gemma-3-12b-it", system="stance", user="stance", json_mode=True)
    assert '"stance"' in resp.text


def test_stub_report_is_stamped_stub():
    report = run_one("wa_i1631", n_personas=4, seed=1, stub=True, budget_usd=1.0)
    assert report.mode == "stub"
    assert "STUB" in report.render()
