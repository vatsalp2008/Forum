"""Stratified persona sampling with k-anonymity enforcement.

Population pipeline (ADR-002; methodology §3):

1. **Scope**: one state, or NATIONAL (every loaded state, population-weighted).
2. **Citizens only** (the default): ACS CIT != 5. Ballot measures are decided
   by citizens, so non-citizen residents are not part of the population.
3. **k-anonymity by merging**: the sampler never draws from a cell with fewer
   than K_ANONYMITY_MIN source records. Thin cells are *merged*, not dropped:
   every PUMA-level cell below the threshold is pooled with the other thin
   cells sharing its demographic combination into one statewide cell
   (location generalized to STATEWIDE_PUMA). Only combinations still below the
   threshold after pooling are refused.
   Why merge: dropping thin cells is not demographically neutral. Rare
   combinations are disproportionately minority residents, so dropping them
   removed 77% of Black and 47% of Hispanic Washingtonians but 8% of white
   ones (persona pool 81% white vs 69% true). Merging recovers all but ~0.06%
   of the population while keeping k>=5 on every combination a persona reveals.
4. **Likely voters** (optional; on for backtests): each cell's weight is
   multiplied by its validated turnout propensity (CCES 2018), so the room
   resembles the electorate instead of all adults.
5. **Party ID**: drawn from the national ANES P(party | age x education x
   race), raked per state so each state's adult-citizen pool matches that
   state's own party balance (CCES). Uncalibrated, a Washington pool came out
   more Republican than Democratic.
6. **Issue priors**: ANES + CCES cross-tabs for (party x education x age),
   blended by cell size.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import duckdb
import numpy as np

from personas.db import get_source_versions
from personas.schema import (
    STATEWIDE_PUMA,
    DemographicSkeleton,
    IssuePriors,
    Persona,
    PopulationSpec,
)

K_ANONYMITY_MIN = 5

# Sentinel population scope: sample across ALL loaded states, population-weighted.
# ACS person weights (person_weight) are national population weights, so weighted
# sampling over the union of states is population-proportional without any
# per-state renormalization.
NATIONAL = "US"

PARTIES = ("dem", "ind", "rep")

# Column order of a sampling-pool row.
_POOL_COLS = ("state", "puma", "age_band", "sex", "race_eth", "education",
              "income_band", "person_weight", "record_count")
# Indices into a pool row.
_STATE, _AGE, _RACE, _EDU, _WEIGHT, _COUNT = 0, 2, 4, 5, 7, 8


def _is_national(state: str) -> bool:
    return state.upper() in (NATIONAL, "*")


def _scope(state: str, citizen_only: bool) -> tuple[str, list]:
    """SQL WHERE clause + params restricting acs_skeleton to the population."""
    clauses: list[str] = []
    params: list = []
    if not _is_national(state):
        clauses.append("state = ?")
        params.append(state)
    if citizen_only:
        clauses.append("citizen")
    return (" AND ".join(clauses) or "TRUE"), params


def _require_citizenship_recorded(con: duckdb.DuckDBPyConnection, state: str) -> None:
    """Fail loudly on libraries built before citizenship was recorded."""
    where, params = _scope(state, citizen_only=False)
    n_null = con.execute(
        f"SELECT COUNT(*) FROM acs_skeleton WHERE {where} AND citizen IS NULL", params
    ).fetchone()[0]
    if n_null:
        raise RuntimeError(
            f"This persona library was built before citizenship was recorded "
            f"({n_null:,} ACS cells lack it), so a citizens-only population cannot "
            f"be drawn. Rebuild it: `forum personas build --state <ST> --cces` "
            f"(or `--national --cces`)."
        )


def build_sampling_pool(
    con: duckdb.DuckDBPyConnection, state: str, citizen_only: bool = True
) -> list[tuple]:
    """Return the k-anonymous sampling cells for a state (or NATIONAL).

    Cells with >= K_ANONYMITY_MIN records are kept at PUMA level. Thinner
    cells are pooled per (state x demographic combination) into a statewide
    cell, which is kept only if the pool reaches the threshold. Citizenship is
    a filter, not a revealed attribute, so cells are aggregated over it. The
    result is ordered so that seeded sampling over it is deterministic.
    """
    if citizen_only:
        _require_citizenship_recorded(con, state)
    where, params = _scope(state, citizen_only)
    sql = f"""
    WITH base AS (
        SELECT state, puma, age_band, sex, race_eth, education, income_band,
               SUM(person_weight) AS person_weight, SUM(record_count) AS record_count
        FROM acs_skeleton
        WHERE {where}
        GROUP BY state, puma, age_band, sex, race_eth, education, income_band
    ),
    merged AS (
        SELECT state, '{STATEWIDE_PUMA}' AS puma, age_band, sex, race_eth, education, income_band,
               SUM(person_weight) AS person_weight, SUM(record_count) AS record_count
        FROM base
        WHERE record_count < ?
        GROUP BY state, age_band, sex, race_eth, education, income_band
    )
    SELECT {", ".join(_POOL_COLS)} FROM base WHERE record_count >= ?
    UNION ALL
    SELECT {", ".join(_POOL_COLS)} FROM merged WHERE record_count >= ?
    ORDER BY state, puma, age_band, sex, race_eth, education, income_band
    """
    k = K_ANONYMITY_MIN
    pool = con.execute(sql, params + [k, k, k]).fetchall()
    _assert_k_anonymous(pool)
    return pool


def _assert_k_anonymous(pool: list[tuple]) -> None:
    """Hard privacy guarantee (ADR-002): refuse to sample from any thin cell."""
    thin = [row for row in pool if row[_COUNT] < K_ANONYMITY_MIN]
    if thin:
        raise RuntimeError(
            f"k-anonymity violation: {len(thin)} sampling cell(s) have fewer than "
            f"{K_ANONYMITY_MIN} source records. Refusing to sample."
        )


def sampling_coverage(
    con: duckdb.DuckDBPyConnection, state: str, citizen_only: bool = True
) -> float:
    """Share of the population's weight that is sampleable after merging.

    The remainder is the weight of demographic combinations too rare to
    sample even statewide (refused to protect privacy).
    """
    pool = build_sampling_pool(con, state, citizen_only)
    where, params = _scope(state, citizen_only)
    total = con.execute(
        f"SELECT SUM(person_weight) FROM acs_skeleton WHERE {where}", params
    ).fetchone()[0]
    if not total:
        return 0.0
    return float(sum(row[_WEIGHT] for row in pool) / total)


def _persona_id(
    skeleton: DemographicSkeleton, seed: int, source_versions: dict, draw_index: int
) -> str:
    # draw_index makes ids unique within a sample: cells are drawn with
    # replacement, and two draws from one cell are two different personas
    # (votes, speaking turns, and pre/post matching are all keyed by id).
    payload = json.dumps(
        {
            "skeleton": skeleton.model_dump(),
            "seed": seed,
            "draw": draw_index,
            "sources": source_versions,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------- party ID: national conditional, calibrated per state ----------

def _load_party_distribution(con: duckdb.DuckDBPyConnection) -> dict[tuple, np.ndarray]:
    rows = con.execute(
        "SELECT age_band, education, race_eth, p_dem, p_ind, p_rep FROM party_id_distribution"
    ).fetchall()
    return {(a, e, r): np.array([d, i, p], dtype=float) for a, e, r, d, i, p in rows}


def _load_party_targets(con: duckdb.DuckDBPyConnection) -> dict[str, np.ndarray]:
    rows = con.execute(
        "SELECT state, p_dem, p_ind, p_rep FROM party_state_targets"
    ).fetchall()
    return {s: np.array([d, i, r], dtype=float) for s, d, i, r in rows}


def _party_probs(dist: dict, age: str, edu: str, race: str) -> np.ndarray:
    """National P(party | age, education, race).

    Every stored row rests on >= PARTY_CELL_MIN respondents: anes_loader backs
    thin race cells off to their (age x education) marginal and records the
    race-cell count in cell_n for transparency only. So a row is used whenever
    it exists. (The sampler used to treat cell_n < 5 as missing and draw party
    uniformly, discarding the valid backoff estimate.) Only a demographic with
    no row at all falls back to uniform.
    """
    q = dist.get((age, edu, race))
    if q is None or q.sum() <= 0:
        return np.full(3, 1.0 / 3.0)
    return q / q.sum()


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    sums = m.sum(axis=1, keepdims=True)
    return np.divide(m, sums, out=np.full_like(m, 1.0 / m.shape[1]), where=sums > 0)


def rake_party_factors(
    group_weights: dict[tuple, float],
    dist: dict,
    target: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Per-party multipliers that make a state's weighted party mix hit target.

    Iterative proportional fitting on the (demographic group x party) table
    W_g * P(party | g): each group's party probabilities stay normalized while
    the party columns are scaled until the population-weighted party shares
    equal the state target. Demographic gradients from the national survey are
    preserved; only the state's overall partisan level shifts.

    group_weights maps (age_band, education, race_eth) -> population weight.
    """
    keys = list(group_weights)
    if not keys:
        return np.ones(3)
    weights = np.array([group_weights[k] for k in keys], dtype=float)
    q = np.vstack([_party_probs(dist, *k) for k in keys])
    target = np.asarray(target, dtype=float) / np.sum(target)
    f = np.ones(3)
    for _ in range(max_iter):
        adj = _normalize_rows(q * f)
        m = (weights[:, None] * adj).sum(axis=0) / weights.sum()
        if np.max(np.abs(m - target)) < tol:
            break
        f = f * np.divide(target, m, out=np.ones(3), where=m > 0)
        f = f / f.sum()
    return f


def _state_party_factors(
    pool: list[tuple], indices: np.ndarray, dist: dict, targets: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Raking factors for every drawn state that has a party target.

    Calibrated on the (adult-citizen) pool weights *before* turnout weighting,
    because the targets describe a state's adult citizens, not its voters.
    """
    states = {pool[i][_STATE] for i in indices} & set(targets)
    groups: dict[str, dict] = {s: defaultdict(float) for s in states}
    for row in pool:
        if row[_STATE] in groups:
            groups[row[_STATE]][(row[_AGE], row[_EDU], row[_RACE])] += row[_WEIGHT]
    return {s: rake_party_factors(groups[s], dist, targets[s]) for s in states}


def _draw_party_id(
    dist: dict,
    factors: np.ndarray | None,
    age_band: str,
    education: str,
    race_eth: str,
    rng: np.random.Generator,
) -> str:
    p = _party_probs(dist, age_band, education, race_eth)
    if factors is not None:
        p = p * factors
        p = p / p.sum() if p.sum() > 0 else np.full(3, 1.0 / 3.0)
    return str(rng.choice(PARTIES, p=p))


# ---------- turnout ----------

def _load_turnout(con: duckdb.DuckDBPyConnection) -> dict[tuple, float]:
    rows = con.execute(
        "SELECT age_band, education, race_eth, p_vote FROM turnout_propensity"
    ).fetchall()
    return {(a, e, r): float(p) for a, e, r, p in rows}


def _turnout_multipliers(con: duckdb.DuckDBPyConnection, pool: list[tuple]) -> np.ndarray:
    turnout = _load_turnout(con)
    if not turnout:
        raise RuntimeError(
            "likely_voters=True needs the CCES validated-turnout table, which is empty. "
            "Run `forum personas build --cces` (requires the CCES 2018 file)."
        )
    fallback = float(np.mean(list(turnout.values())))
    return np.array(
        [turnout.get((r[_AGE], r[_EDU], r[_RACE]), fallback) for r in pool], dtype=float
    )


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

    national = _is_national(spec.state)
    pool = build_sampling_pool(con, spec.state, citizen_only=spec.citizen_only)
    if not pool:
        scope = "any loaded state" if national else f"state={spec.state}"
        raise RuntimeError(
            f"No ACS cells with k-anonymity >= {K_ANONYMITY_MIN} for {scope}. "
            f"Load ACS data first."
        )

    base_weights = np.array([row[_WEIGHT] for row in pool], dtype=float)
    draw_weights = base_weights
    if spec.likely_voters:
        draw_weights = base_weights * _turnout_multipliers(con, pool)

    rng = np.random.default_rng(spec.seed)
    indices = rng.choice(
        len(pool), size=spec.n, replace=True, p=draw_weights / draw_weights.sum()
    )

    dist = _load_party_distribution(con)
    factors = _state_party_factors(pool, indices, dist, _load_party_targets(con))

    personas: list[Persona] = []
    for draw_index, i in enumerate(indices):
        row = pool[i]
        skeleton = DemographicSkeleton(
            state=row[0], puma=row[1], age_band=row[2], sex=row[3],  # type: ignore[arg-type]
            race_eth=row[4], education=row[5], income_band=row[6],   # type: ignore[arg-type]
        )
        party_id = _draw_party_id(
            dist, factors.get(skeleton.state),
            skeleton.age_band, skeleton.education, skeleton.race_eth, rng,
        )
        priors = _lookup_priors(con, party_id, skeleton.education, skeleton.age_band)
        pid = _persona_id(skeleton, spec.seed, source_versions, draw_index)
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
