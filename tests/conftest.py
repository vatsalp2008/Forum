"""Shared test fixtures.

Every test runs from a fresh temporary working directory. FORUM resolves its
persona library (personas/data/persona_library.duckdb) and run artifacts
(backtest/runs/) relative to the working directory, so without this the suite
would silently read the developer's real persona library and litter
backtest/runs/ with stub reports. Tests that need a library build their own in
tmp_path.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
