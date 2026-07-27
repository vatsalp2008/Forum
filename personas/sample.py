"""Stratified persona sampling with k-anonymity enforcement.

The sampler refuses to draw from cells with record_count < K_ANONYMITY_MIN.
Cells below the threshold are dropped from the sampling pool; their weight
is redistributed proportionally to the remaining valid cells.
"""

from __future__ import annotations

import hashlib
import json

import duckdb
import numpy as np

from personas.db import get_source_versions
from personas.schema import (
    DemographicSkeleton,
    IssuePriors,
    Persona,
    PopulationSpec,
)

K_ANONYMITY_MIN = 5


def _persona_id(skeleton: DemographicSkeleton, seed: int, source_versions: dict) -> str:
    payload = json.dumps(
        {
            "skeleton": skeleton.model_dump(),
            "seed": seed,
            "sources": source_versions,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _draw_party_id(
    con: duckdb.DuckDBPyConnection,
    age_band: str,
    education: str,
    race_eth: str,
    rng: np.random.Generator,
) -> str:
    row = con.execute(
        """
        SELECT p_dem, p_ind, p_rep, cell_n FROM party_id_distribution
        WHERE age_band = ? AND education = ? AND race_eth = ?
        """,
        [age_band, education, race_eth],
    ).fetchone()
    if row is None or row[3] < K_ANONYMITY_MIN:
        # fall back to uniform; flag in source as "fallback_uniform"
        return str(rng.choice(["dem", "ind", "rep"]))
    p_dem, p_ind, p_rep, _ = row
    return str(rng.choice(["dem", "ind", "rep"], p=[p_dem, p_ind, p_rep]))


# Issue-prior sources, blended at sample time. Add new survey tables here;
# each must share anes_priors' schema (see personas/db.py).
PRIOR_SOURCE_TABLES = ("anes_priors", "cces_priors")


def _lookup_priors(
    con: duckdb.DuckDBPyConnection,
    party_id: str,
    education: str,
    age_band: str,
) -> IssuePriors:
    """Blend issue priors across all sources for this cell (cell-N weighted).

    Each source contributes its (climate, gun, tax, ideology) values weighted
    by its cell sample size, so larger surveys pull the estimate proportionally.
    Sources whose cell is missing or below k-anonymity are skipped. If no source
    has a valid cell, degrade to marginal 0.5 priors.
    """
    contributions: list[tuple[tuple[float, float, float, float], int]] = []
    for table in PRIOR_SOURCE_TABLES:
        row = con.execute(
            f"""
            SELECT p_climate_action_support, p_gun_restriction_support,
                   p_tax_on_rich_support, ideology_score, cell_n
            FROM {table}
            WHERE party_id = ? AND education = ? AND age_band = ?
            """,
            [party_id, education, age_band],
        ).fetchone()
        if row is not None and row[4] >= K_ANONYMITY_MIN:
            contributions.append((row[:4], int(row[4])))

    if not contributions:
        # graceful degradation: marginal-only priors.
        return IssuePriors(
            party_id=party_id,  # type: ignore[arg-type]
            p_climate_action_support=0.5,
            p_gun_restriction_support=0.5,
            p_tax_on_rich_support=0.5,
            ideology_score=0.0,
        )

    total_n = sum(n for _, n in contributions)
    blended = [
        sum(vals[i] * n for vals, n in contributions) / total_n
        for i in range(4)
    ]
    return IssuePriors(
        party_id=party_id,  # type: ignore[arg-type]
        p_climate_action_support=blended[0],
        p_gun_restriction_support=blended[1],
        p_tax_on_rich_support=blended[2],
        ideology_score=blended[3],
    )


def sample_personas(
    con: duckdb.DuckDBPyConnection, spec: PopulationSpec
) -> list[Persona]:
    """Draw N personas under the population spec.

    Determinism: identical (spec, source_versions) => identical persona list.
    Privacy: no cell with record_count < K_ANONYMITY_MIN is sampled from.
    """
    source_versions = get_source_versions(con)
    if not source_versions:
        raise RuntimeError("No source versions recorded. Load ACS+ANES first.")

    # Pool: skeleton cells from the requested state with k-anonymity enforced.
    pool = con.execute(
        """
        SELECT state, puma, age_band, sex, race_eth, education,
               income_band, person_weight, record_count
        FROM acs_skeleton
        WHERE state = ? AND record_count >= ?
        """,
        [spec.state, K_ANONYMITY_MIN],
    ).fetchall()

    if not pool:
        raise RuntimeError(
            f"No ACS cells with k-anonymity >= {K_ANONYMITY_MIN} for state={spec.state}. "
            f"Load ACS data first."
        )

    weights = np.array([row[7] for row in pool], dtype=float)
    weights = weights / weights.sum()

    rng = np.random.default_rng(spec.seed)
    indices = rng.choice(len(pool), size=spec.n, replace=True, p=weights)

    personas: list[Persona] = []
    for i in indices:
        row = pool[i]
        skeleton = DemographicSkeleton(
            state=row[0], puma=row[1], age_band=row[2], sex=row[3],  # type: ignore[arg-type]
            race_eth=row[4], education=row[5], income_band=row[6],   # type: ignore[arg-type]
        )
        party_id = _draw_party_id(
            con, skeleton.age_band, skeleton.education, skeleton.race_eth, rng
        )
        priors = _lookup_priors(con, party_id, skeleton.education, skeleton.age_band)
        pid = _persona_id(skeleton, spec.seed, source_versions)
        personas.append(
            Persona(
                persona_id=pid,
                demographics=skeleton,
                priors=priors,
                sampling_seed=spec.seed,
                source_versions=source_versions,
            )
        )
    return personas
