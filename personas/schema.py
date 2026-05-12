"""Persona record schema and population specification.

A Persona is a frozen, hashable record. Determinism requires that the same population
spec + seed + source versions produces the same Persona records bit-for-bit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AgeBand = Literal["18-29", "30-44", "45-64", "65+"]
Sex = Literal["male", "female"]
RaceEthnicity = Literal[
    "white_nh",        # White, non-Hispanic
    "black_nh",        # Black, non-Hispanic
    "hispanic",
    "asian_nh",
    "other_nh",        # AIAN / NHPI / multiracial / other, non-Hispanic
]
Education = Literal[
    "lt_hs",           # Less than high school
    "hs",              # High school graduate
    "some_college",    # Some college, including associate's
    "bachelors",       # Bachelor's degree
    "graduate",        # Graduate or professional degree
]
IncomeBand = Literal[
    "lt_25k",
    "25_50k",
    "50_75k",
    "75_125k",
    "125k_plus",
]
PartyID = Literal["dem", "ind", "rep"]


class DemographicSkeleton(BaseModel):
    """Joint demographic attributes drawn from ACS PUMS."""

    model_config = {"frozen": True}

    age_band: AgeBand
    sex: Sex
    race_eth: RaceEthnicity
    education: Education
    income_band: IncomeBand
    puma: str  # 5-digit PUMA code (state-prefixed)
    state: str  # 2-letter state code


class IssuePriors(BaseModel):
    """Issue-position priors derived from ANES cross-tabs.

    Each value is a probability in [0, 1] of the corresponding stance.
    Conditioned on (party_id x education x age_band) — the canonical ANES
    coarsening that maintains adequate cell sizes.
    """

    model_config = {"frozen": True}

    party_id: PartyID
    p_climate_action_support: float = Field(ge=0.0, le=1.0)
    p_gun_restriction_support: float = Field(ge=0.0, le=1.0)
    p_tax_on_rich_support: float = Field(ge=0.0, le=1.0)
    ideology_score: float = Field(ge=-1.0, le=1.0)  # -1 liberal, +1 conservative


class Persona(BaseModel):
    """A single synthetic persona."""

    model_config = {"frozen": True}

    persona_id: str  # deterministic hash of (skeleton + seed + source versions)
    demographics: DemographicSkeleton
    priors: IssuePriors
    sampling_seed: int
    source_versions: dict[str, str]  # e.g., {"acs": "5yr_2018_2022", "anes": "2020_ts"}


class PopulationSpec(BaseModel):
    """Specification of a population to sample from."""

    model_config = {"frozen": True}

    name: str
    state: str
    citizen_only: bool = True
    adult_only: bool = True
    n: int  # number of personas to draw
    seed: int
    source_versions: dict[str, str]
