"""Suite-wide fixtures.

The environment fixture is load-bearing, not hygiene. `labkit.env.load_dotenv()` writes
straight into `os.environ`, and `monkeypatch.delenv(name, raising=False)` records nothing
when the name was already absent -- so there is nothing for pytest to restore and the
value leaks into every test that runs afterwards. With `-p randomly` the resulting
failure moves around, which is the worst kind: `test_contrasts_get_the_same_step_budget`
passed alone and failed in the suite.
"""
from __future__ import annotations

import os
import sys

import pytest

# Keep `import labkit` working from a bare checkout, with no install step.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True)
def _restore_environment():
    """Every test gets the process environment back exactly as it found it."""
    saved = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
