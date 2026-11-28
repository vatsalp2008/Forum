"""The Gemini provider calls the REST API directly and keeps model thinking out.

Gemma 4 is a thinking model: its reply carries the model's private reasoning
as parts marked "thought": true, ahead of the answer. The old SDK glued those
parts onto the answer, so a fact check came back as an essay and a citizen's
statement would have carried the model's reasoning into the transcript that
the other personas read.
"""

import json

import httpx
import pytest

from forum.budget import CostMeter
from forum.llm import (
    GeminiHTTPError,
    LLMClient,
    LLMError,
    UnusableReplyError,
    _retry_delay_seconds,
    gemini_answer_text,
)

THINKING = "*   Role: Fact-checker.\n    *   Task: Review one statement.\n    Not flagged."


def _reply(*parts, finish="STOP", usage=None):
    return {
        "candidates": [{"content": {"role": "model", "parts": list(parts)}, "finishReason": finish}],
        "usageMetadata": usage or {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }


@pytest.fixture
def gemini(monkeypatch):
    """Build a live LLMClient whose HTTP traffic goes to `handler`, not Google."""
    monkeypatch.setattr("forum.llm.time.sleep", lambda *_: None)
    real_client = httpx.Client
    requests: list[httpx.Request] = []

    def make(handler, **kw):
        def recording(request):
            requests.append(request)
            return handler(request)

        monkeypatch.setattr(httpx, "Client", lambda **ckw: real_client(
            transport=httpx.MockTransport(recording), **ckw))
        llm = LLMClient(meter=CostMeter(cap_usd=1.0), stub=False, api_key="test-key",
                        min_interval_s=0.0, **kw)
        return llm, requests

    return make


def test_thinking_parts_never_reach_the_answer(gemini):
    answer = '{"flagged": false, "note": ""}'
    llm, _ = gemini(lambda r: httpx.Response(200, json=_reply(
        {"text": THINKING, "thought": True}, {"text": answer})))
    resp = llm.generate(model="gemma-4-31b-it", system="s", user="u", json_mode=True)
    assert resp.text == answer


def test_request_shape_and_key_handling(gemini):
    llm, requests = gemini(lambda r: httpx.Response(200, json=_reply({"text": "{}"})))
    llm.generate(model="gemini-2.5-flash-lite", system="SYS", user="USER", json_mode=True,
                 temperature=0.3)
    llm.generate(model="gemma-4-31b-it", system="SYS", user="USER", json_mode=True)

    flash, gemma = requests
    assert flash.url.path.endswith("/models/gemini-2.5-flash-lite:generateContent")
    assert flash.headers["x-goog-api-key"] == "test-key"
    assert "test-key" not in str(flash.url)                       # never in the URL
    body = json.loads(flash.content)
    assert body["contents"][0]["parts"][0]["text"] == "SYS\n\nUSER"
    assert body["generationConfig"] == {"temperature": 0.3, "responseMimeType": "application/json"}
    assert "responseMimeType" not in json.loads(gemma.content)["generationConfig"]   # Gemma: no JSON mode


def test_meter_charges_reported_tokens_including_thinking(gemini):
    usage = {"promptTokenCount": 1000, "candidatesTokenCount": 100, "thoughtsTokenCount": 50}
    llm, _ = gemini(lambda r: httpx.Response(200, json=_reply({"text": "ok"}, usage=usage)))
    llm.generate(model="gemini-2.5-flash-lite", system="s", user="u")
    assert llm.meter.spent_usd == pytest.approx((1000 * 0.10 + 150 * 0.40) / 1e6)


def test_overloaded_model_is_retried(gemini):
    replies = iter([httpx.Response(503, json={"error": {"status": "UNAVAILABLE"}}),
                    httpx.Response(200, json=_reply({"text": "fine"}))])
    llm, requests = gemini(lambda r: next(replies))
    assert llm.generate(model="gemma-4-31b-it", system="s", user="u").text == "fine"
    assert len(requests) == 2


def test_bad_request_fails_loudly_without_retrying(gemini):
    llm, requests = gemini(lambda r: httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT"}}),
                           max_retries=4)
    with pytest.raises(LLMError, match="HTTP 400"):
        llm.generate(model="gemma-4-31b-it", system="s", user="u")
    assert len(requests) == 1


def test_cut_off_reply_is_retried_then_fails_loudly(gemini):
    llm, requests = gemini(lambda r: httpx.Response(200, json=_reply(
        {"text": "half an ans"}, finish="MAX_TOKENS")), max_retries=3)
    with pytest.raises(LLMError, match="MAX_TOKENS"):
        llm.generate(model="gemma-4-31b-it", system="s", user="u")
    assert len(requests) == 3


def test_reply_with_only_thinking_has_no_answer():
    with pytest.raises(UnusableReplyError):
        gemini_answer_text(_reply({"text": THINKING, "thought": True}))
    with pytest.raises(UnusableReplyError):
        gemini_answer_text({"promptFeedback": {"blockReason": "SAFETY"}})


def test_retry_delay_is_read_from_rest_and_grpc_errors():
    rest = GeminiHTTPError(429, json.dumps({"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "23s"}]}}, indent=2))
    assert _retry_delay_seconds(rest) == 23.0
    assert _retry_delay_seconds(RuntimeError("retry_delay { seconds: 2 }")) == 2.0
    assert _retry_delay_seconds(GeminiHTTPError(503, "{}")) is None
