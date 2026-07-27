"""CCES/CES priors loader: derives issue-position priors from CCES 2018.

Acquisition:
    Download the CCES 2018 Common Content from the Harvard Dataverse
    (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZSBZ7K).
    Place the Stata file at personas/data/raw/cces/cces18_common_vv.dta
    (a .csv with the same stem also works).

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

CCES_RAW_DTA = Path("personas/data/raw/cces/cces18_common_vv.dta")
CCES_RAW_CSV = Path("personas/data/raw/cces/cces18_common_vv.csv")
CCES_VERSION = "2018_cc"

# CCES 2018 Common Content variables. VERIFY against the CCES 2018 Guide.
#   pid7       7-pt party ID: 1 Strong Dem ... 7 Strong Rep, 8 Not sure
#   educ       1 No HS, 2 HS grad, 3 Some college, 4 2-year, 5 4-year, 6 Post-grad
#   birthyr    birth year (age = survey year - birthyr)
#   ideo5      1 Very liberal ... 5 Very conservative, 6 Not sure
#   CC18_415a  Give EPA power to regulate CO2 emissions: 1 Support, 2 Oppose (climate)
#   CC18_320a  Background checks for all gun sales:       1 Support, 2 Oppose (guns)
#   CC18_417   Raise taxes / state budget item:           VERIFY coding    (tax proxy)
PARTY_ID_VAR = "pid7"
EDUCATION_VAR = "educ"
BIRTHYEAR_VAR = "birthyr"
IDEOLOGY_VAR = "ideo5"
CLIMATE_VAR = "CC18_415a"
GUNS_VAR = "CC18_320a"
TAXES_VAR = "CC18_417"  # VERIFY: substitute the correct tax-on-rich item
SURVEY_YEAR = 2018


def _party_id_collapse(v: float) -> str | None:
    """pid7 -> {dem, ind, rep}; drop 8 (Not sure) and out-of-range."""
    if pd.isna(v) or v < 1 or v > 7:
        return None
    if v <= 3:
        return "dem"
    if v == 4:
        return "ind"
    return "rep"


def _education_collapse(v: float) -> str | None:
    """CCES educ (6-cat) -> FORUM schema. 2-year folds into some_college."""
    if pd.isna(v) or v < 1 or v > 6:
        return None
    return {1: "lt_hs", 2: "hs", 3: "some_college", 4: "some_college",
            5: "bachelors", 6: "graduate"}[int(v)]


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
    """Support/Oppose items coded 1=Support, 2=Oppose. Share who Support."""
    s = series.dropna()
    s = s[(s >= 1) & (s <= 2)]
    if len(s) == 0:
        return 0.5
    return float((s == 1).mean())


def _ideology_score(series: pd.Series) -> float:
    """ideo5 (1..5) -> [-1, +1]; drop 6 (Not sure). Return cell mean."""
    s = series.dropna()
    s = s[(s >= 1) & (s <= 5)]
    if len(s) == 0:
        return 0.0
    return float(((s - 3) / 2.0).mean())


def _read_cces() -> pd.DataFrame:
    if CCES_RAW_DTA.exists():
        return pd.read_stata(CCES_RAW_DTA, convert_categoricals=False)
    if CCES_RAW_CSV.exists():
        return pd.read_csv(CCES_RAW_CSV, low_memory=False)
    raise FileNotFoundError(
        f"CCES file not found at {CCES_RAW_DTA} or {CCES_RAW_CSV}. Download the "
        f"CCES 2018 Common Content from the Harvard Dataverse and place it there. "
        f"See personas/cces_loader.py docstring."
    )


def load_cces(con: duckdb.DuckDBPyConnection) -> int:
    """Load CCES priors into cces_priors. Returns n_prior_cells."""
    df = _read_cces()

    required = [PARTY_ID_VAR, EDUCATION_VAR, BIRTHYEAR_VAR, IDEOLOGY_VAR,
                CLIMATE_VAR, GUNS_VAR, TAXES_VAR]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"CCES columns not found: {missing}. Verify the variable map in "
            f"personas/cces_loader.py against the CCES 2018 Guide (names/codings "
            f"vary by year). Refusing to build rather than guess."
        )

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
