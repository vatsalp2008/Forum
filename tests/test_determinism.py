"""Same seed => same result, across separate Python processes.

Python salts str hashes per process (PYTHONHASHSEED), so any seed derived from
the built-in hash() silently changes between runs. These tests pin the
methodology §4 reproducibility claim for everything on FORUM's side.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

from forum.determinism import stable_hash

REPO = Path(__file__).resolve().parent.parent

_SCRIPT = """
from forum.budget import CostMeter
from forum.llm import make_llm_client
from forum.graph import run_deliberation
from backtest.measure_loader import load_measure
from backtest.run import _stub_personas
m = load_measure("wa_i1631")
personas = _stub_personas(6, "WA", seed=42)
llm = make_llm_client("gemini", CostMeter(1.0), stub=True)
final = run_deliberation(llm, m.spec, personas, seed=42)
print([(v.persona_id, v.round, v.stance) for v in final["votes"]])
"""


def _run_with_hashseed(hashseed: str, cwd: Path) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(REPO)}
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=cwd, env=env, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_same_seed_same_votes_across_processes(tmp_path):
    results = {_run_with_hashseed(h, tmp_path) for h in ("1", "2", "3")}
    assert len(results) == 1, f"votes differ across processes: {results}"


def test_stable_hash_is_deterministic_and_bounded():
    assert stable_hash("persona-abc") == stable_hash("persona-abc")
    assert stable_hash("persona-abc") != stable_hash("persona-abd")
    assert 0 <= stable_hash("x", 100_000) < 100_000


def test_no_builtin_hash_calls_in_source():
    """Seeds must use stable_hash; the salted builtin hash() is banned."""
    offenders = []
    for pkg in ("forum", "backtest", "personas"):
        for path in (REPO / pkg).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "hash"):
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, f"builtin hash() used (not process-stable): {offenders}"
