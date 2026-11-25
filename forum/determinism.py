"""Process-independent hashing for deriving seeds from strings.

Python's built-in ``hash()`` on ``str`` is salted per process (PYTHONHASHSEED),
so a seed derived from it changes every time the interpreter starts: the same
``--seed`` would give different personas' votes, speaker seeds, and stub output
on every run. Every seed FORUM derives from a string must go through
``stable_hash`` instead (tests/test_determinism.py enforces this).
"""

from __future__ import annotations

import hashlib


def stable_hash(text: str, modulo: int = 2**32) -> int:
    """Deterministic non-negative integer hash of ``text``, in ``[0, modulo)``.

    Identical across processes, machines, and Python versions.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo
