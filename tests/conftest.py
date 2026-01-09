import pytest
import sys
import os
from turboscan import HyperBoost, HARDWARE

# Add src to path so tests can run without installing package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture(autouse=True)
def clean_hyperboost_state():
    """Reset HyperBoost state before each test."""
    # Clear any active thread pools
    HyperBoost.shutdown()
    # Reset internal counters if possible
    yield
    HyperBoost.shutdown()

@pytest.fixture
def force_cpu_only(monkeypatch):
    """Simulate a machine with NO GPU and NO Ray."""
    monkeypatch.setattr('turboscan.execution.hyper_boost.GPU_AVAIL', False)
    monkeypatch.setattr('turboscan.execution.hyper_boost.RAY_AVAIL', False)
    monkeypatch.setattr('turboscan.execution.hyper_boost.GPU_COUNT', 0)

@pytest.fixture
def complex_data():
    """Generates data that usually breaks multiprocessing."""
    class DirtyClass:
        def __init__(self, val):
            self.val = val
        def __repr__(self):
            return f"Dirty({self.val})"
    
    return [DirtyClass(i) for i in range(100)]