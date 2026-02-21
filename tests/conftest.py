import os
import sys

import pytest

from turboscan import HyperBoost

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)


@pytest.fixture(autouse=True)
def clean_hyperboost_state():
    """Reset HyperBoost state before each test."""

    HyperBoost.shutdown()

    yield
    HyperBoost.shutdown()


@pytest.fixture
def force_cpu_only(monkeypatch) -> None:
    """Simulate a machine with NO GPU and NO Ray."""
    monkeypatch.setattr("turboscan.execution.hyper_boost.GPU_AVAIL", False)
    monkeypatch.setattr("turboscan.execution.hyper_boost.RAY_AVAIL", False)
    monkeypatch.setattr("turboscan.execution.hyper_boost.GPU_COUNT", 0)


@pytest.fixture
def complex_data():
    """Generates data that usually breaks multiprocessing."""

    class DirtyClass:
        def __init__(self, val) -> None:
            self.val = val

        def __repr__(self) -> str:
            return f"Dirty({self.val})"

    return [DirtyClass(i) for i in range(100)]
