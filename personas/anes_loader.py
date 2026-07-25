"""ANES priors loader: derives issue-position priors from ANES 2020 microdata.

Acquisition:
    Register at https://electionstudies.org and download the ANES 2020
    Time Series Study Full Release. Place the .dta file at
    personas/data/raw/anes/anes_2020_timeseries.dta

The ANES data agreement permits use and aggregation but restricts
redistribution of microdata. This loader produces aggregated cross-tab
priors that are stored in the persona library; the raw microdata is never
shipped or queryable through FORUM.

The variable names below are the ANES 2020 Time Series codebook variables.
The user should verify against the actual codebook delivered with their
data download — variable names occasionally shift between releases.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from personas.db import record_source_version

ANES_RAW_PATH = Path("personas/data/raw/anes/anes_2020_timeseries.dta")
ANES_VERSION = "2020_ts"

# ANES 2020 variables verified against
# anes_timeseries_2020_userguidecodebook_20220210.pdf:
#
#   V201231x  PRE: SUMMARY: PARTY ID                       (7-pt: 1 Strong Dem ... 7 Strong Rep)
#   V201511x  PRE: SUMMARY: RESPONDENT 5 CATEGORY EDUCATION (1 lt_hs ... 5 graduate)
#   V201507x  PRE: SUMMARY: RESPONDENT AGE                 (numeric, top-coded at 80)
#   V201200   PRE: 7PT LIB-CONS SELF-PLACEMENT             (1 Ext Lib ... 7 Ext Cons; 99 = haven't thought)
#   V202334   POST: FAVOR/OPPOSE GREENHOUSE EMISSIONS REG   (1 Favor, 2 Oppose, 3 Neither)
#   V202339   POST: FAVOR/OPPOSE BACKGROUND CHECKS FOR GUNS (1 Favor, 2 Oppose, 3 Neither)
#   V202325   POST: FAVOR/OPPOSE TAX ON MILLIONAIRES        (1 Favor, 2 Oppose, 3 Neither)
#   V201549x  PRE: SUMMARY: R self-identified race/ethnicity (1 White NH ... 6 Multiple NH)
PARTY_ID_VAR = "V201231x"
EDUCATION_VAR = "V201511x"
AGE_VAR = "V201507x"
IDEOLOGY_VAR = "V201200"
CLIMATE_VAR = "V202334"
GUNS_VAR = "V202339"
TAXES_VAR = "V202325"
RACE_VAR = "V201549x"

# Minimum respondents for a party-ID cell to be estimated directly. Race-
# conditioned (age x education x race) cells that fall below this back off to
# the (age x education) marginal rather than being fabricated or dropped.
PARTY_CELL_MIN = 15


def _party_id_collapse(v: float) -> str | None:
    """Collapse 7-pt summary V201231x to {dem, ind, rep}.

    1 Strong Dem, 2 Not very strong Dem, 3 Independent-Dem,
    4 Independent,
    5 Independent-Rep, 6 Not very strong Rep, 7 Strong Rep.
    """
    if pd.isna(v) or v < 1 or v > 7:
        return None
    if v <= 3:
        return "dem"
    if v == 4:
        return "ind"
    return "rep"


def _education_collapse(v: float) -> str | None:
    """Collapse V201511x (5-cat summary) to FORUM's schema.

    1 Less than HS, 2 HS credential, 3 Some post-HS no bachelor's,
    4 Bachelor's, 5 Graduate.
    """
    if pd.isna(v) or v < 1 or v > 5:
        return None
    v = int(v)
    return {1: "lt_hs", 2: "hs", 3: "some_college", 4: "bachelors", 5: "graduate"}[v]


def _race_collapse(v: float) -> str | None:
    """Collapse V201549x (self-identified race/ethnicity summary) to schema.

    1 White NH, 2 Black NH, 3 Hispanic, 4 Asian/NHPI NH,
    5 Native American/other NH, 6 Multiple races NH. 5 and 6 fold into
    other_nh (both too small to stand alone under k-anonymity).
    """
    if pd.isna(v) or v < 1 or v > 6:
        return None
    return {
        1: "white_nh", 2: "black_nh", 3: "hispanic",
        4: "asian_nh", 5: "other_nh", 6: "other_nh",
    }[int(v)]


def _party_dist(grp: pd.DataFrame) -> dict[str, float]:
    counts = grp["party_id"].value_counts(normalize=True)
    return {
        "p_dem": float(counts.get("dem", 0.0)),
        "p_ind": float(counts.get("ind", 0.0)),
        "p_rep": float(counts.get("rep", 0.0)),
    }


def _age_collapse(v: float) -> str | None:
    if pd.isna(v) or v < 18:
        return None
    v = int(v)
    if v <= 29:
        return "18-29"
    if v <= 44:
        return "30-44"
    if v <= 64:
        return "45-64"
    return "65+"


def _favor_share(series: pd.Series) -> float:
    """For V202325/V202334/V202339: 1=Favor, 2=Oppose, 3=Neither.
    Returns share who Favor among valid (1-3) responses.
    """
    s = series.dropna()
    s = s[(s >= 1) & (s <= 3)]
    if len(s) == 0:
        return 0.5
    return float((s == 1).mean())


def _ideology_score(series: pd.Series) -> float:
    """Map 7-point ideology to [-1, +1] score; return cell mean."""
    s = series.dropna()
    s = s[(s >= 1) & (s <= 7)]
    if len(s) == 0:
        return 0.0
    return float(((s - 4) / 3.0).mean())


def load_anes(con: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """Load ANES priors. Returns (n_prior_cells, n_partyid_cells)."""
    if not ANES_RAW_PATH.exists():
        raise FileNotFoundError(
            f"ANES file not found at {ANES_RAW_PATH}. "
            f"Register at https://electionstudies.org, download the 2020 Time Series "
            f"Study (Stata .dta), and place there. See personas/anes_loader.py docstring."
        )

    df = pd.read_stata(ANES_RAW_PATH, convert_categoricals=False)

    df["party_id"] = df[PARTY_ID_VAR].apply(_party_id_collapse)
    df["education"] = df[EDUCATION_VAR].apply(_education_collapse)
    df["age_band"] = df[AGE_VAR].apply(_age_collapse)
    df["race_eth"] = df[RACE_VAR].apply(_race_collapse)
    df = df.dropna(subset=["party_id", "education", "age_band"])

    # Issue priors keyed by (party_id, education, age_band)
    rows: list[dict] = []
    for (pid, edu, age), grp in df.groupby(["party_id", "education", "age_band"]):
        if len(grp) < 5:
            continue  # k-anonymity on the prior cells too
        rows.append({
            "party_id": pid,
            "education": edu,
            "age_band": age,
            "p_climate_action_support": _favor_share(grp[CLIMATE_VAR]),
            "p_gun_restriction_support": _favor_share(grp[GUNS_VAR]),
            "p_tax_on_rich_support": _favor_share(grp[TAXES_VAR]),
            "ideology_score": _ideology_score(grp[IDEOLOGY_VAR]),
            "cell_n": int(len(grp)),
        })
    priors_df = pd.DataFrame(rows)

    con.execute("DELETE FROM anes_priors")
    if len(priors_df):
        con.register("priors_df", priors_df)
        con.execute("INSERT INTO anes_priors SELECT * FROM priors_df")
        con.unregister("priors_df")

    # Conditional party-ID distribution P(party_id | age_band x education x race_eth).
    # Race-conditioned where the (age x education x race) cell has enough
    # respondents; otherwise a documented hierarchical backoff to the
    # (age x education) marginal. We never fabricate race variation that the
    # data does not support, and we never blanket-marginalize when race-level
    # data IS available. cell_n records the true race-cell count, so thin
    # backoff cells stay visible downstream.
    all_races = ("white_nh", "black_nh", "hispanic", "asian_nh", "other_nh")
    df_race = df.dropna(subset=["race_eth"])
    marginals = {
        key: _party_dist(grp)
        for key, grp in df.groupby(["age_band", "education"])
        if len(grp) >= PARTY_CELL_MIN
    }
    race_cells = {
        key: (_party_dist(grp), len(grp))
        for key, grp in df_race.groupby(["age_band", "education", "race_eth"])
    }
    party_rows: list[dict] = []
    for (age, edu) in marginals:
        for race in all_races:
            cell = race_cells.get((age, edu, race))
            if cell is not None and cell[1] >= PARTY_CELL_MIN:
                dist, n = cell                       # race-conditioned estimate
            else:
                dist = marginals[(age, edu)]         # backoff to age x education
                n = cell[1] if cell is not None else 0
            party_rows.append({
                "age_band": age, "education": edu, "race_eth": race,
                **dist, "cell_n": int(n),
            })
    party_df = pd.DataFrame(party_rows)

    con.execute("DELETE FROM party_id_distribution")
    if len(party_df):
        con.register("party_df", party_df)
        con.execute("INSERT INTO party_id_distribution SELECT * FROM party_df")
        con.unregister("party_df")

    record_source_version(con, "anes", ANES_VERSION)
    return len(priors_df), len(party_df)
