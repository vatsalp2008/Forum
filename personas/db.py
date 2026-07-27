"""DuckDB connection and schema initialization for the persona library."""

from __future__ import annotations

from pathlib import Path

import duckdb

DB_PATH = Path("personas/data/persona_library.duckdb")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS acs_skeleton (
    state           VARCHAR NOT NULL,
    puma            VARCHAR NOT NULL,
    age_band        VARCHAR NOT NULL,
    sex             VARCHAR NOT NULL,
    race_eth        VARCHAR NOT NULL,
    education       VARCHAR NOT NULL,
    income_band     VARCHAR NOT NULL,
    person_weight   DOUBLE  NOT NULL,
    record_count    INTEGER NOT NULL  -- raw record count in this cell
);

CREATE INDEX IF NOT EXISTS idx_acs_state ON acs_skeleton(state);

CREATE TABLE IF NOT EXISTS anes_priors (
    party_id        VARCHAR NOT NULL,
    education       VARCHAR NOT NULL,
    age_band        VARCHAR NOT NULL,
    p_climate_action_support  DOUBLE NOT NULL,
    p_gun_restriction_support DOUBLE NOT NULL,
    p_tax_on_rich_support     DOUBLE NOT NULL,
    ideology_score            DOUBLE NOT NULL,
    cell_n                    INTEGER NOT NULL,
    PRIMARY KEY (party_id, education, age_band)
);

CREATE TABLE IF NOT EXISTS cces_priors (
    -- Same schema/keys as anes_priors; a second issue-prior source (CES/CCES)
    -- that is blended with anes_priors at sample time (cell-N weighted).
    party_id        VARCHAR NOT NULL,
    education       VARCHAR NOT NULL,
    age_band        VARCHAR NOT NULL,
    p_climate_action_support  DOUBLE NOT NULL,
    p_gun_restriction_support DOUBLE NOT NULL,
    p_tax_on_rich_support     DOUBLE NOT NULL,
    ideology_score            DOUBLE NOT NULL,
    cell_n                    INTEGER NOT NULL,
    PRIMARY KEY (party_id, education, age_band)
);

CREATE TABLE IF NOT EXISTS party_id_distribution (
    -- conditional distribution P(party_id | demographic skeleton)
    age_band        VARCHAR NOT NULL,
    education       VARCHAR NOT NULL,
    race_eth        VARCHAR NOT NULL,
    p_dem           DOUBLE NOT NULL,
    p_ind           DOUBLE NOT NULL,
    p_rep           DOUBLE NOT NULL,
    cell_n          INTEGER NOT NULL,
    PRIMARY KEY (age_band, education, race_eth)
);

CREATE TABLE IF NOT EXISTS source_versions (
    source_name   VARCHAR PRIMARY KEY,
    version       VARCHAR NOT NULL,
    loaded_at     TIMESTAMP NOT NULL
);
"""


def connect(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open (or create) the persona library database."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA_SQL)
    return con


def record_source_version(
    con: duckdb.DuckDBPyConnection, source_name: str, version: str
) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    con.execute(
        """
        INSERT INTO source_versions(source_name, version, loaded_at)
        VALUES (?, ?, ?)
        ON CONFLICT (source_name) DO UPDATE
            SET version = excluded.version, loaded_at = excluded.loaded_at
        """,
        [source_name, version, now],
    )


def get_source_versions(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    rows = con.execute("SELECT source_name, version FROM source_versions").fetchall()
    return {name: version for name, version in rows}
