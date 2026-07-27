"""CCES label-recode unit tests (no data file needed)."""

import pandas as pd

from personas.cces_loader import (
    _age_collapse_from_birthyear,
    _education_collapse,
    _ideology_score,
    _party_id_collapse,
    _support_share,
)


def test_party_label_collapse():
    assert _party_id_collapse("Strong Democrat") == "dem"
    assert _party_id_collapse("Lean Republican") == "rep"
    assert _party_id_collapse("Independent") == "ind"
    assert _party_id_collapse("Not sure") is None


def test_education_label_collapse():
    assert _education_collapse("No HS") == "lt_hs"
    assert _education_collapse("2-year") == "some_college"
    assert _education_collapse("Post-grad") == "graduate"
    assert _education_collapse("nonsense") is None


def test_age_from_birthyear():
    assert _age_collapse_from_birthyear(1990) == "18-29"   # 2018-1990 = 28
    assert _age_collapse_from_birthyear(1980) == "30-44"   # 38
    assert _age_collapse_from_birthyear(1940) == "65+"     # 78
    assert _age_collapse_from_birthyear(float("nan")) is None


def test_support_share_labels():
    s = pd.Series(["Support", "Support", "Oppose", "Not sure", None])
    assert _support_share(s) == 2 / 3          # Not sure/None excluded
    f = pd.Series(["For", "Against", "Against"])
    assert _support_share(f) == 1 / 3
    assert _support_share(pd.Series(["Not sure"])) == 0.5   # no valid -> neutral


def test_ideology_score_labels():
    assert _ideology_score(pd.Series(["Very liberal"])) == -1.0
    assert _ideology_score(pd.Series(["Very conservative"])) == 1.0
    assert _ideology_score(pd.Series(["Moderate", "Not sure"])) == 0.0
