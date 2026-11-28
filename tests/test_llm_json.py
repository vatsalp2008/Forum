"""Unusable model replies must stop the run, never become made-up data.

Before this, an unreadable vote silently became a neutral 0.5 and an unreadable
fact check became "not flagged". Neutral votes drag the yes-share toward 50%,
which flatters FORUM on close measures. Gemma has no native JSON mode, so it
wraps JSON in prose and fences more often than Gemini.
"""

import pytest

from backtest.measure_loader import load_measure
from backtest.run import _stub_personas
from forum.graph import run_deliberation
from forum.llm import LLMError, LLMResponse, extract_json_object, generate_json, json_bool
from tests.fakes import ScriptedLLM


@pytest.mark.parametrize("text, expected", [
    ('{"stance": 0.7}', {"stance": 0.7}),
    ('```json\n{"stance": 0.7}\n```', {"stance": 0.7}),
    ('Here is my vote:\n```json\n{"stance": 0.7}\n```\nThanks!', {"stance": 0.7}),
    ('Sure. {"flagged": false, "note": ""} Hope that helps.', {"flagged": False, "note": ""}),
])
def test_extract_json_object_tolerates_fences_and_prose(text, expected):
    assert extract_json_object(text) == expected


@pytest.mark.parametrize("text", ["", "I would vote yes.", "[0.7]", "42", "{stance: 0.7}"])
def test_extract_json_object_rejects_anything_but_an_object(text):
    assert extract_json_object(text) is None


def test_json_bool_does_not_invert_quoted_booleans():
    assert json_bool(False) is False
    assert json_bool("false") is False          # bool("false") would be True
    assert json_bool(" TRUE ") is True
    with pytest.raises(ValueError):
        json_bool("maybe")
    with pytest.raises(ValueError):
        json_bool(None)


class _Replies:
    """Returns canned replies in order and records the seed of each call."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.seeds = []

    def generate(self, model, system, user, json_mode=False, temperature=0.7, seed=None):
        assert json_mode
        self.seeds.append(seed)
        return LLMResponse(text=self._replies.pop(0), input_tokens=1, output_tokens=1)


def test_generate_json_asks_once_more_after_an_unusable_reply():
    llm = _Replies("I'd lean yes.", '{"stance": 0.8}')
    assert generate_json(llm, model="m", system="s", user="u", seed=10) == {"stance": 0.8}
    assert llm.seeds == [10, 11]


def test_generate_json_raises_after_two_unusable_replies():
    llm = _Replies("garbage", '{"confidence": 0.9}')   # second lacks the stance
    with pytest.raises(LLMError, match="unusable JSON reply from 'm' after 2 attempts"):
        generate_json(llm, model="m", system="s", user="u", seed=1,
                      parse=lambda o: float(o["stance"]))


class _Fake(ScriptedLLM):
    """ScriptedLLM with replaceable vote and fact-check replies."""

    def __init__(self, vote_text=None, critic_text=None, **kw):
        super().__init__(**kw)
        self._vote_text, self._critic_text = vote_text, critic_text

    def generate(self, model, system, user, json_mode=False, temperature=0.7, seed=None):
        resp = super().generate(model, system, user, json_mode, temperature, seed)
        role = self.calls[-1][0]
        if role == "vote" and self._vote_text is not None:
            return LLMResponse(text=self._vote_text, input_tokens=1, output_tokens=1)
        if role == "critic" and self._critic_text is not None:
            return LLMResponse(text=self._critic_text, input_tokens=1, output_tokens=1)
        return resp


def _deliberate(llm):
    measure = load_measure("wa_i1631")
    return run_deliberation(llm, measure.spec, _stub_personas(6, "WA", 1), seed=1)


def test_unreadable_vote_stops_the_room_instead_of_voting_neutral():
    llm = _Fake(vote_text="I'd probably vote yes, honestly.")
    with pytest.raises(LLMError):
        _deliberate(llm)
    assert [role for role, _, _ in llm.calls] == ["vote", "vote"]   # asked twice, then stopped


def test_vote_without_a_stance_is_unusable():
    with pytest.raises(LLMError):
        _deliberate(_Fake(vote_text='{"confidence": 0.9, "rationale": "no stance given"}'))


def test_vote_wrapped_in_prose_is_read_correctly():
    final = _deliberate(_Fake(vote_text='My vote: {"stance": 0.85, "confidence": 0.6, "rationale": "r"}'))
    assert {v.stance for v in final["votes"]} == {0.85}


@pytest.mark.parametrize("critic_text, flagged", [
    ('{"flagged": "false", "note": ""}', False),       # bool("false") used to flag it
    ('```json\n{"flagged": "true", "note": "wrong figure"}\n```', True),
])
def test_fact_check_reads_quoted_booleans_and_fences(critic_text, flagged):
    final = _deliberate(_Fake(critic_text=critic_text))
    citizen = [s for s in final["statements"] if s.speaker_id != "moderator"]
    assert citizen and all(s.flagged_by_critic is flagged for s in citizen)


def test_unreadable_fact_check_stops_the_room_instead_of_passing_the_statement():
    with pytest.raises(LLMError):
        _deliberate(_Fake(critic_text="Looks accurate to me."))
