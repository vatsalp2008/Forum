"""ACS PUMS loader: ingests Census PUMS microdata into the persona DB.

Acquisition:
    Download the ACS 5-year 2018-2022 PUMS person-level CSVs from
    https://www.census.gov/programs-surveys/acs/microdata.html and place
    them at personas/data/raw/acs/psam_p<state>.csv (e.g., psam_p53.csv for WA).

This loader does NOT download data automatically (the user is responsible
for the acquisition step, which requires accepting Census terms).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from personas.db import record_source_version

ACS_RAW_DIR = Path("personas/data/raw/acs")
ACS_VERSION = "5yr_2018_2022"

# State FIPS code mapping for the v0 scope (WA only). Add more as needed.
STATE_FIPS = {"WA": "53"}


# ---------- Re-coding ACS variables into the FORUM coarsened schema ----------

def _agep_to_band(agep: int) -> str | None:
    if agep < 18:
        return None
    if agep <= 29:
        return "18-29"
    if agep <= 44:
        return "30-44"
    if agep <= 64:
        return "45-64"
    return "65+"


def _sex_recode(sex: int) -> str:
    return "male" if sex == 1 else "female"


def _race_eth_recode(rac1p: int, hisp: int) -> str:
    if hisp != 1:
        return "hispanic"
    if rac1p == 1:
        return "white_nh"
    if rac1p == 2:
        return "black_nh"
    if rac1p == 6:
        return "asian_nh"
    return "other_nh"


def _schl_to_education(schl: int) -> str:
    if schl <= 15:
        return "lt_hs"
    if schl <= 17:
        return "hs"
    if schl <= 20:
        return "some_college"
    if schl == 21:
        return "bachelors"
    return "graduate"


def _pincp_to_income_band(pincp: float) -> str:
    if pincp < 25_000:
        return "lt_25k"
    if pincp < 50_000:
        return "25_50k"
    if pincp < 75_000:
        return "50_75k"
    if pincp < 125_000:
        return "75_125k"
    return "125k_plus"


# ---------- Loader ----------

def _coalesce_puma(row) -> str | None:
    """5-year files have both PUMA10 and PUMA20 columns; pick the populated one."""
    p20 = row.get("PUMA20")
    p10 = row.get("PUMA10")
    if pd.notna(p20) and p20 != -9 and p20 != 0:
        return str(int(p20)).zfill(5)
    if pd.notna(p10) and p10 != -9 and p10 != 0:
        return str(int(p10)).zfill(5)
    return None


def load_state(con: duckdb.DuckDBPyConnection, state: str = "WA") -> int:
    """Load ACS PUMS for a single state into acs_skeleton.

    Returns the number of cells inserted (post-aggregation).
    """
    fips = STATE_FIPS.get(state)
    if fips is None:
        raise ValueError(f"State {state} not in STATE_FIPS; add it.")

    csv_path = ACS_RAW_DIR / f"psam_p{fips}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ACS PUMS file not found at {csv_path}. "
            f"Download from https://www.census.gov/programs-surveys/acs/microdata.html "
            f"and place there. See personas/acs_loader.py docstring."
        )

    cols = ["ST", "PUMA10", "PUMA20", "AGEP", "SEX", "RAC1P", "HISP", "SCHL", "PINCP", "PWGTP"]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    df = df.dropna(subset=["AGEP", "SEX", "RAC1P", "SCHL"])

    df["age_band"] = df["AGEP"].astype(int).apply(_agep_to_band)
    df = df[df["age_band"].notna()].copy()  # drop minors

    df["sex"] = df["SEX"].astype(int).apply(_sex_recode)
    df["race_eth"] = df.apply(
        lambda r: _race_eth_recode(int(r["RAC1P"]), int(r["HISP"])), axis=1
    )
    df["education"] = df["SCHL"].fillna(0).astype(int).apply(_schl_to_education)
    df["income_band"] = df["PINCP"].fillna(0).astype(float).apply(_pincp_to_income_band)
    df["state"] = state
    df["puma"] = df.apply(_coalesce_puma, axis=1)
    df = df[df["puma"].notna()].copy()

    grouped = (
        df.groupby(
            [
                "state", "puma", "age_band", "sex", "race_eth",
                "education", "income_band",
            ],
            as_index=False,
        )
        .agg(person_weight=("PWGTP", "sum"), record_count=("PWGTP", "count"))
    )

    con.execute("DELETE FROM acs_skeleton WHERE state = ?", [state])
    con.register("grouped_df", grouped)
    con.execute("INSERT INTO acs_skeleton SELECT * FROM grouped_df")
    con.unregister("grouped_df")

    record_source_version(con, "acs", ACS_VERSION)
    return len(grouped)
