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
import numpy as np
import pandas as pd

from personas.db import record_source_version

ACS_RAW_DIR = Path("personas/data/raw/acs")
ACS_VERSION = "5yr_2018_2022"

# Full FIPS -> USPS map (50 states + DC; territories like PR excluded).
FIPS_TO_USPS = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}
# USPS -> 2-digit zero-padded FIPS, used to build per-state filenames
# (psam_p<FIPS>.csv, e.g. WA -> psam_p53.csv, CA -> psam_p06.csv).
STATE_FIPS = {usps: f"{fips:02d}" for fips, usps in FIPS_TO_USPS.items()}

# ACS PUMS person columns FORUM reads.
_PUMS_COLS = ["ST", "PUMA10", "PUMA20", "AGEP", "SEX", "RAC1P", "HISP", "SCHL", "PINCP", "PWGTP"]


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


# ---------- National (multi-state) loader ----------

def _recode_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized recode of a raw PUMS chunk into FORUM cells.

    Equivalent to the scalar recodes above but fast enough for the ~16M-row
    national files. Returns rows with a valid state/age/puma; drops the rest.
    """
    age = df["AGEP"].to_numpy()
    schl = df["SCHL"].fillna(0).to_numpy(dtype="int64")
    inc = df["PINCP"].fillna(0).to_numpy(dtype="float64")
    rac = df["RAC1P"].to_numpy()
    hisp_ne1 = df["HISP"].to_numpy() != 1

    p20 = pd.to_numeric(df["PUMA20"], errors="coerce")
    p10 = pd.to_numeric(df["PUMA10"], errors="coerce")
    p20_ok = p20.notna() & (p20 != -9) & (p20 != 0)
    p10_ok = p10.notna() & (p10 != -9) & (p10 != 0)
    puma_num = p20.where(p20_ok, p10.where(p10_ok))

    out = pd.DataFrame({
        "state": df["ST"].map(FIPS_TO_USPS),
        "puma": puma_num,
        "age_band": np.select(
            [age < 18, age <= 29, age <= 44, age <= 64],
            [None, "18-29", "30-44", "45-64"], default="65+",
        ),
        "sex": np.where(df["SEX"].to_numpy() == 1, "male", "female"),
        "race_eth": np.select(
            [hisp_ne1, rac == 1, rac == 2, rac == 6],
            ["hispanic", "white_nh", "black_nh", "asian_nh"], default="other_nh",
        ),
        "education": np.select(
            [schl <= 15, schl <= 17, schl <= 20, schl == 21],
            ["lt_hs", "hs", "some_college", "bachelors"], default="graduate",
        ),
        "income_band": np.select(
            [inc < 25_000, inc < 50_000, inc < 75_000, inc < 125_000],
            ["lt_25k", "25_50k", "50_75k", "75_125k"], default="125k_plus",
        ),
        "PWGTP": df["PWGTP"].to_numpy(),
    })
    out = out.dropna(subset=["state", "puma", "age_band"])
    out["puma"] = out["puma"].astype("int64").astype(str).str.zfill(5)
    return out


def _find_national_files() -> list[Path]:
    for d in (ACS_RAW_DIR / "csv_pus", ACS_RAW_DIR):
        files = sorted(d.glob("psam_pus*.csv"))
        if files:
            return files
    return []


def load_national(con: duckdb.DuckDBPyConnection, chunksize: int = 2_000_000) -> int:
    """Load the national ACS PUMS (psam_pus[a-d].csv) for all 50 states + DC.

    Reads in chunks to bound memory, recodes vectorized, aggregates to cells,
    and replaces the entire acs_skeleton (national supersedes any per-state
    load). Returns the number of cells inserted.
    """
    files = _find_national_files()
    if not files:
        raise FileNotFoundError(
            f"National PUMS files (psam_pus*.csv) not found under {ACS_RAW_DIR}. "
            f"Download csv_pus.zip from "
            f"https://www2.census.gov/programs-surveys/acs/data/pums/2022/5-Year/ "
            f"and unzip into personas/data/raw/acs/csv_pus/."
        )

    keys = ["state", "puma", "age_band", "sex", "race_eth", "education", "income_band"]
    partials: list[pd.DataFrame] = []
    for path in files:
        for chunk in pd.read_csv(path, usecols=_PUMS_COLS, chunksize=chunksize):
            recoded = _recode_vectorized(chunk)
            partials.append(
                recoded.groupby(keys, as_index=False).agg(
                    person_weight=("PWGTP", "sum"), record_count=("PWGTP", "count")
                )
            )

    combined = (
        pd.concat(partials, ignore_index=True)
        .groupby(keys, as_index=False)
        .agg(person_weight=("person_weight", "sum"), record_count=("record_count", "sum"))
    )

    con.execute("DELETE FROM acs_skeleton")
    con.register("national_df", combined)
    con.execute("INSERT INTO acs_skeleton SELECT * FROM national_df")
    con.unregister("national_df")

    record_source_version(con, "acs", ACS_VERSION)
    return len(combined)
