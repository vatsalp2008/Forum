from datetime import date, timedelta

from forum.refusal import check_output, check_request


def test_refuses_election_prediction_framing():
    res = check_request("Predict the election outcome for the upcoming senate race.")
    assert res.refused
    assert "refus" in (res.reason or "").lower()


def test_refuses_persuasive_messaging_framing():
    res = check_request("Generate persuasive messaging targeted at suburban women.")
    assert res.refused


def test_refuses_demographic_targeting():
    res = check_request("How do we convert young rural voters to support this policy?")
    assert res.refused


def test_passes_neutral_research_framing():
    res = check_request(
        "Should the state implement a carbon pollution fee on large emitters?"
    )
    assert not res.refused


def test_refuses_pre_election_window():
    near = date.today() + timedelta(days=30)
    res = check_request("This is a neutral framing.", vote_date=near)
    assert res.refused


def test_passes_outside_window():
    far = date.today() + timedelta(days=365)
    res = check_request("This is a neutral framing.", vote_date=far)
    assert not res.refused


def test_output_refusal_catches_messaging_recommendation():
    text = "Here is a messaging recommendation for the campaign: ..."
    assert check_output(text).refused


def test_output_refusal_passes_clean_report():
    text = "Predicted yes-share: 47%. Confidence interval: ±5pts. Notes follow."
    assert not check_output(text).refused
