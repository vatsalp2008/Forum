"""CCES/CES priors loader: derives issue-position priors from CCES 2018.

Acquisition:
    Download the CCES 2018 Common Content from the Harvard Dataverse
    (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZSBZ7K).
    Place the file at
    personas/data/raw/cces/CCES18_Common_OUTPUT_vv_topost.dta (or .csv).
    The .csv is preferred here (much smaller); only the needed columns are read.

Like anes_loader, this produces aggregated cross-tab priors stored in the
persona library; raw microdata is never shipped or queryable through FORUM.
Priors land in the `cces_priors` table and are blended with `anes_priors` at
sample time (cell-N weighted; see personas/sample.py).

IMPORTANT — verify the variable map below against the CCES 2018 Guide before
trusting a build. Variable names/codings differ across CCES years. This loader
FAILS LOUDLY (KeyError with guidance) if a mapped column is absent rather than
guessing — it will never silently fabricate priors.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np  # noqa: F401  (parity with anes_loader; may be used in recodes)
import pandas as pd

from personas.db import record_source_version

CCES_RAW_DTA = Path("personas/data/raw/cces/CCES18_Common_OUTPUT_vv_topost.dta")
CCES_RAW_CSV = Path("personas/data/raw/cces/CCES18_Common_OUTPUT_vv_topost.csv")
CCES_VERSION = "2018_cc"

# CCES 2018 Common Content variables, verified against "CCES Guide 2018.pdf":
#   pid7       7-pt party ID: 1 Strong Dem ... 7 Strong Rep (8 Not sure -> dropped)
#   educ       1 No HS, 2 HS grad, 3 Some college, 4 2-year, 5 4-year, 6 Post-grad
#   birthyr    birth year (age = survey year - birthyr)
#   ideo5      1 Very liberal ... 5 Very conservative, 6 Not sure (dropped)
#   CC18_415a  Give EPA power to regulate CO2 emissions: 1 Support, 2 Oppose (climate)
#   CC18_320a  Background checks for all gun sales:      1 For, 2 Against   (guns)
#   CC18_414B  Ballot measure: increase taxes on incomes over $1M by 4% for
#              schools/roads: 1 For, 2 Against                             (tax on rich)
PARTY_ID_VAR = "pid7"
EDUCATION_VAR = "educ"
BIRTHYEAR_VAR = "birthyr"
IDEOLOGY_VAR = "ideo5"
CLIMATE_VAR = "CC18_415a"
GUNS_VAR = "CC18_320a"
TAXES_VAR = "CC18_414B"
SURVEY_YEAR = 2018


# The CCES CSV stores text value labels (not numeric codes). We read labels
# from both the CSV and the .dta (convert_categoricals=True) so the recodes
# below are label-based and source-independent. birthyr is numeric.
_PARTY_LABELS = {
    "Strong Democrat": "dem", "Not very strong Democrat": "dem", "Lean Democrat": "dem",
    "Independent": "ind",
    "Lean Republican": "rep", "Not very strong Republican": "rep", "Strong Republican": "rep",
}
_EDUCATION_LABELS = {
    "No HS": "lt_hs", "High school graduate": "hs", "Some college": "some_college",
    "2-year": "some_college", "4-year": "bachelors", "Post-grad": "graduate",
}
_IDEOLOGY_SCORES = {
    "Very liberal": -1.0, "Liberal": -0.5, "Moderate": 0.0,
    "Conservative": 0.5, "Very conservative": 1.0,
}
_SUPPORT_LABELS = {"Support", "For"}
_OPPOSE_LABELS = {"Oppose", "Against"}


def _party_id_collapse(v) -> str | None:
    return _PARTY_LABELS.get(v)   # "Not sure"/unknown -> None


def _education_collapse(v) -> str | None:
    return _EDUCATION_LABELS.get(v)


def _age_collapse_from_birthyear(v: float) -> str | None:
    if pd.isna(v):
        return None
    age = SURVEY_YEAR - int(v)
    if age < 18:
        return None
    if age <= 29:
        return "18-29"
    if age <= 44:
        return "30-44"
    if age <= 64:
        return "45-64"
    return "65+"


def _support_share(series: pd.Series) -> float:
    """Support-vs-oppose items. Share who Support among valid responses."""
    valid = series[series.isin(_SUPPORT_LABELS | _OPPOSE_LABELS)]
    if len(valid) == 0:
        return 0.5
    return float(valid.isin(_SUPPORT_LABELS).mean())


def _ideology_score(series: pd.Series) -> float:
    """ideo5 labels -> [-1, +1]; drop 'Not sure'. Return cell mean."""
    scores = series.map(_IDEOLOGY_SCORES).dropna()
    if len(scores) == 0:
        return 0.0
    return float(scores.mean())


def _read_cces(usecols: list[str] | None = None) -> pd.DataFrame:
    # Prefer the CSV (much smaller than the .dta) and read only needed columns.
    if CCES_RAW_CSV.exists():
        return pd.read_csv(CCES_RAW_CSV, usecols=usecols, low_memory=False)
    if CCES_RAW_DTA.exists():
        # convert_categoricals=True yields the same text labels as the CSV.
        return pd.read_stata(CCES_RAW_DTA, convert_categoricals=True, columns=usecols)
    raise FileNotFoundError(
        f"CCES file not found at {CCES_RAW_CSV} or {CCES_RAW_DTA}. Download the "
        f"CCES 2018 Common Content from the Harvard Dataverse and place it there. "
        f"See personas/cces_loader.py docstring."
    )


def load_cces(con: duckdb.DuckDBPyConnection) -> int:
    """Load CCES priors into cces_priors. Returns n_prior_cells."""
    required = [PARTY_ID_VAR, EDUCATION_VAR, BIRTHYEAR_VAR, IDEOLOGY_VAR,
                CLIMATE_VAR, GUNS_VAR, TAXES_VAR]
    try:
        df = _read_cces(usecols=required)
    except ValueError as e:
        # pandas raises ValueError when a usecols entry is absent.
        raise KeyError(
            f"CCES columns not found ({e}). Verify the variable map in "
            f"personas/cces_loader.py against the CCES 2018 Guide. "
            f"Refusing to build rather than guess."
        ) from e

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"CCES columns not found: {missing}. Verify the variable map in "
            f"personas/cces_loader.py against the CCES 2018 Guide (names/codings "
            f"vary by year). Refusing to build rather than guess."
        )

    # Only birthyr is numeric; the rest are text value labels handled by the
    # label-based recodes below.
    df[BIRTHYEAR_VAR] = pd.to_numeric(df[BIRTHYEAR_VAR], errors="coerce")

    df["party_id"] = df[PARTY_ID_VAR].apply(_party_id_collapse)
    df["education"] = df[EDUCATION_VAR].apply(_education_collapse)
    df["age_band"] = df[BIRTHYEAR_VAR].apply(_age_collapse_from_birthyear)
    df = df.dropna(subset=["party_id", "education", "age_band"])

    rows: list[dict] = []
    for (pid, edu, age), grp in df.groupby(["party_id", "education", "age_band"]):
        if len(grp) < 5:
            continue  # k-anonymity on prior cells
        rows.append({
            "party_id": pid,
            "education": edu,
            "age_band": age,
            "p_climate_action_support": _support_share(grp[CLIMATE_VAR]),
            "p_gun_restriction_support": _support_share(grp[GUNS_VAR]),
            "p_tax_on_rich_support": _support_share(grp[TAXES_VAR]),
            "ideology_score": _ideology_score(grp[IDEOLOGY_VAR]),
            "cell_n": int(len(grp)),
        })
    priors_df = pd.DataFrame(rows)

    con.execute("DELETE FROM cces_priors")
    if len(priors_df):
        con.register("cces_df", priors_df)
        con.execute("INSERT INTO cces_priors SELECT * FROM cces_df")
        con.unregister("cces_df")

    record_source_version(con, "cces", CCES_VERSION)
    return len(priors_df)
