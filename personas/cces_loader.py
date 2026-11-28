"""CCES/CES priors loader: derives issue-position priors from CCES 2018.

Acquisition:
    Download the CCES 2018 Common Content from the Harvard Dataverse
    (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZSBZ7K).
    Place the file at
    personas/data/raw/cces/CCES18_Common_OUTPUT_vv_topost.dta (or .csv).
    The .csv is preferred here (much smaller); only the needed columns are read.

Like anes_loader, this produces aggregated cross-tab tables stored in the
persona library; raw microdata is never shipped or queryable through FORUM.
Three tables are built:

- `cces_priors` — issue priors, blended with `anes_priors` at sample time
  (cell-N weighted; see personas/sample.py).
- `turnout_propensity` — P(voted | age x education x race) among adult
  citizens, from the 2018 *validated* vote (voter-file match). Weights the
  sampling pool toward the electorate rather than all adults.
- `party_state_targets` — each state's party-ID shares among adult citizens
  (survey-weighted). The sampler calibrates the national P(party | demo) to
  these so a state's persona pool has that state's partisan balance.

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
#   race       White, Black, Hispanic, Asian, Native American, Mixed, Other,
#              Middle Eastern
#   hispanic   Hispanic origin (asked of non-Hispanic-race respondents): Yes/No
#   cit1       U.S. citizen: Yes/No
#   inputstate state of residence (full state name)
#   commonweight  survey weight (all respondents)
#   CL_2018gvm validated 2018 general-election vote method (polling, mail,
#              earlyVote, absentee, unknown); blank = no record of voting
#              (unmatched respondents count as non-voters, the standard
#              validated-turnout convention)
PARTY_ID_VAR = "pid7"
EDUCATION_VAR = "educ"
BIRTHYEAR_VAR = "birthyr"
IDEOLOGY_VAR = "ideo5"
CLIMATE_VAR = "CC18_415a"
GUNS_VAR = "CC18_320a"
TAXES_VAR = "CC18_414B"
RACE_VAR = "race"
HISPANIC_VAR = "hispanic"
CITIZEN_VAR = "cit1"
STATE_VAR = "inputstate"
WEIGHT_VAR = "commonweight"
VOTED_2018_VAR = "CL_2018gvm"
SURVEY_YEAR = 2018

# Minimum respondents for a directly estimated turnout cell (else back off to
# the age x education marginal), and for a state's party target to be used.
TURNOUT_CELL_MIN = 15
PARTY_TARGET_MIN = 100

ALL_RACES = ("white_nh", "black_nh", "hispanic", "asian_nh", "other_nh")


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
# Mirrors the ACS recode: any Hispanic origin -> hispanic; Middle Eastern ->
# white_nh because the Census (whose RAC1P the ACS skeleton uses) classifies
# MENA respondents as White.
_RACE_LABELS = {
    "White": "white_nh", "Black": "black_nh", "Asian": "asian_nh",
    "Middle Eastern": "white_nh",
    "Mixed": "other_nh", "Other": "other_nh", "Native American": "other_nh",
}
_STATE_NAME_TO_USPS = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
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


def _race_collapse(race, hispanic) -> str | None:
    if race == "Hispanic" or hispanic == "Yes":
        return "hispanic"
    return _RACE_LABELS.get(race)


def _voted(series: pd.Series) -> pd.Series:
    """Validated-vote indicator: any recorded vote method counts as voting."""
    return series.notna() & (series.astype(str).str.strip() != "")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total = float(weights.sum())
    if total <= 0:
        return float(values.mean())
    return float((values * weights).sum() / total)


def _turnout_table(df: pd.DataFrame) -> pd.DataFrame:
    """P(voted | age x education x race) among adult citizens (weighted).

    Expects recoded columns age_band, education, race_eth, citizen, voted and
    the weight column. Race cells with fewer than TURNOUT_CELL_MIN respondents
    back off to their (age x education) marginal, as party ID does in ANES.
    """
    d = df[df["citizen"]].dropna(subset=["age_band", "education", "race_eth"])
    w = d[WEIGHT_VAR].astype(float)
    v = d["voted"].astype(float)
    marginals = {
        key: _weighted_mean(v[grp.index], w[grp.index])
        for key, grp in d.groupby(["age_band", "education"])
        if len(grp) >= TURNOUT_CELL_MIN
    }
    cells = {
        key: (_weighted_mean(v[grp.index], w[grp.index]), len(grp))
        for key, grp in d.groupby(["age_band", "education", "race_eth"])
    }
    rows: list[dict] = []
    for (age, edu), marginal in marginals.items():
        for race in ALL_RACES:
            cell = cells.get((age, edu, race))
            if cell is not None and cell[1] >= TURNOUT_CELL_MIN:
                p_vote, n = cell
            else:
                p_vote, n = marginal, (cell[1] if cell is not None else 0)
            rows.append({"age_band": age, "education": edu, "race_eth": race,
                         "p_vote": p_vote, "cell_n": int(n)})
    return pd.DataFrame(rows, columns=["age_band", "education", "race_eth", "p_vote", "cell_n"])


def _party_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted party-ID shares among each state's adult citizens.

    States with fewer than PARTY_TARGET_MIN respondents are omitted (their
    personas keep the uncalibrated national party distribution).
    """
    d = df[df["citizen"]].dropna(subset=["party_id", "state"])
    rows: list[dict] = []
    for state, grp in d.groupby("state"):
        if len(grp) < PARTY_TARGET_MIN:
            continue
        w = grp[WEIGHT_VAR].astype(float)
        total = float(w.sum())
        shares = {k: float(w[grp["party_id"] == k].sum() / total) for k in ("dem", "ind", "rep")}
        rows.append({"state": state, "p_dem": shares["dem"], "p_ind": shares["ind"],
                     "p_rep": shares["rep"], "n_respondents": int(len(grp))})
    return pd.DataFrame(rows, columns=["state", "p_dem", "p_ind", "p_rep", "n_respondents"])


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


def load_cces(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Load CCES into cces_priors, turnout_propensity and party_state_targets.

    Returns the number of rows written to each table.
    """
    required = [PARTY_ID_VAR, EDUCATION_VAR, BIRTHYEAR_VAR, IDEOLOGY_VAR,
                CLIMATE_VAR, GUNS_VAR, TAXES_VAR, RACE_VAR, HISPANIC_VAR,
                CITIZEN_VAR, STATE_VAR, WEIGHT_VAR, VOTED_2018_VAR]
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

    df[WEIGHT_VAR] = pd.to_numeric(df[WEIGHT_VAR], errors="coerce").fillna(0.0)

    df["party_id"] = df[PARTY_ID_VAR].apply(_party_id_collapse)
    df["education"] = df[EDUCATION_VAR].apply(_education_collapse)
    df["age_band"] = df[BIRTHYEAR_VAR].apply(_age_collapse_from_birthyear)
    df["race_eth"] = [_race_collapse(r, h) for r, h in zip(df[RACE_VAR], df[HISPANIC_VAR], strict=True)]
    df["citizen"] = df[CITIZEN_VAR] == "Yes"
    df["state"] = df[STATE_VAR].map(_STATE_NAME_TO_USPS)
    df["voted"] = _voted(df[VOTED_2018_VAR])

    turnout_df = _turnout_table(df)
    targets_df = _party_targets(df)
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

    _replace_table(con, "cces_priors", priors_df)
    _replace_table(con, "turnout_propensity", turnout_df)
    _replace_table(con, "party_state_targets", targets_df)

    record_source_version(con, "cces", CCES_VERSION)
    return {"priors": len(priors_df), "turnout": len(turnout_df),
            "party_targets": len(targets_df)}


def _replace_table(con: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame) -> None:
    con.execute(f"DELETE FROM {table}")
    if len(frame):
        con.register("cces_df", frame)
        cols = ", ".join(frame.columns)
        con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM cces_df")
        con.unregister("cces_df")
