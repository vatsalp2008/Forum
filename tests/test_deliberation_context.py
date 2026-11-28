"""What each agent sees: pinned briefing (fix 4) and clean counterfactuals (fix 6)."""

from backtest.measure_loader import load_measure
from backtest.run import _stub_personas
from forum.graph import format_transcript, post_vote_context, run_deliberation
from tests.fakes import ScriptedLLM


def _run(n_personas=6):
    measure = load_measure("wa_i1631")
    llm = ScriptedLLM()
    final = run_deliberation(llm, measure.spec, _stub_personas(n_personas, "WA", 1), seed=1)
    return llm, final


def test_pre_vote_happens_before_any_information():
    llm, _ = _run()
    pre_votes = [p for r, s, u in llm.calls[:6] for p in [s + u]]
    assert all("BRIEFING-TEXT" not in p for p in pre_votes)


def test_every_speaker_sees_the_briefing():
    """It used to scroll out of a 6-statement window: only 6 of 20 speakers saw it."""
    llm, final = _run()
    speaker_prompts = llm.prompts("speaker")
    assert len(speaker_prompts) == 10   # 2 speakers x 5 rounds for 6 personas
    assert all("BRIEFING-TEXT" in p for p in speaker_prompts)


def test_final_vote_sees_briefing_and_the_whole_discussion():
    llm, final = _run()
    final_votes = llm.prompts("vote")[6:]          # first 6 votes are the pre-vote
    n_statements = sum(1 for s in final["statements"] if s.speaker_id != "moderator")
    for p in final_votes:
        assert "BRIEFING-TEXT" in p
        assert sum(f"STATEMENT-{i:02d}" in p for i in range(1, n_statements + 1)) == n_statements


def test_removing_a_speaker_changes_only_that_speakers_lines():
    """The persuasion confound: ablation used to pull the briefing back into view."""
    _, final = _run(n_personas=12)
    statements = final["statements"]
    briefing, full = post_vote_context(statements)
    speakers = {s.speaker_id for s in statements if s.speaker_id != "moderator"}
    assert len(speakers) == 12
    for sp in speakers:
        b_ablated, ablated = post_vote_context(statements, exclude_speaker=sp)
        assert b_ablated == briefing == "BRIEFING-TEXT"
        kept = [line for line in full.splitlines() if not line.startswith(f"- {sp} ")]
        assert ablated.splitlines() == kept


def test_speaker_window_is_recent_citizen_statements_only():
    _, final = _run()
    window = format_transcript(final["statements"], limit=6)
    assert "BRIEFING-TEXT" not in window
    assert len(window.splitlines()) == 6
