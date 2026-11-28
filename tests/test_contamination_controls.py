"""No agent may see a measure's title, ballot number, or year (fix 7)."""

import re

import pytest

from backtest.measure_loader import list_measures, load_measure
from backtest.run import _stub_personas
from forum.graph import run_deliberation
from forum.redaction import measure_number, redact_curated, redact_generated
from tests.fakes import ScriptedLLM


@pytest.mark.parametrize("measure_id", sorted(list_measures()))
def test_no_agent_prompt_identifies_the_measure(measure_id):
    m = load_measure(measure_id)
    llm = ScriptedLLM()
    run_deliberation(llm, m.spec, _stub_personas(6, m.state, 1), seed=1)
    number = measure_number(m.spec.measure_id)
    assert number, "every backtest measure id should end in its ballot number"
    for _role, system, user in llm.calls:
        prompt = system + "\n" + user
        assert not re.search(rf"\b{number}\b", prompt), f"ballot number leaked: {prompt[:200]}"
        assert str(m.year) not in prompt, f"ballot year leaked: {prompt[:200]}"
        assert m.spec.title not in prompt


def test_moderator_that_recalls_the_initiative_is_scrubbed():
    m = load_measure("wa_i1631")
    llm = ScriptedLLM(moderator_text="Initiative 1631 (I-1631), on the 2018 ballot, would tax carbon.")
    final = run_deliberation(llm, m.spec, _stub_personas(6, "WA", 1), seed=1)
    briefing = final["statements"][0].text
    assert "1631" not in briefing and "2018" not in briefing
    assert all("1631" not in p for p in llm.prompts("speaker"))


def test_citizen_statement_naming_the_initiative_is_scrubbed():
    m = load_measure("wa_i1631")
    llm = ScriptedLLM(statement_text=lambda n: f"Remember I-1631? My commute on I-5 is long ({n}).")
    final = run_deliberation(llm, m.spec, _stub_personas(6, "WA", 1), seed=1)
    texts = [s.text for s in final["statements"] if s.speaker_id != "moderator"]
    assert all("1631" not in t for t in texts)
    assert all("I-5" in t for t in texts)   # ordinary references survive


def test_redaction_rules():
    src = "Washington State Voters' Pamphlet, Initiative Measure No. 1631 (full text, 2018)"
    assert redact_curated(src, "1631") == "Washington State Voters' Pamphlet, the measure (full text)"
    assert redact_curated("Washington Office of Financial Management, I-1433 fiscal impact statement") \
        == "Washington Office of Financial Management, the measure fiscal impact statement"
    # Model output keeps freeways and future targets but loses this measure's identity.
    out = redact_generated("I-1631 failed in 2018; I-405 traffic; targets for 2030.", "1631", 2018)
    assert "1631" not in out and "2018" not in out
    assert "I-405" in out and "2030" in out
    assert measure_number("wa_i1631") == "1631"
    assert measure_number("ca_prop64") is None   # 2-digit numbers are too ambiguous to strip
