import pytest
import threading
import numpy as np
import time
from turboscan.analysis.function_analyzer import FunctionAnalyzer

# --- Helper Functions (Must be at module level for inspect.getsource to work) ---

def simple_cpu_task(x):
    """Pure CPU task, no threads."""
    return x * x + 2

def heavy_numpy_task(arr):
    """Explicit CPU-bound patterns."""
    return np.mean(arr) + np.std(arr)

def thread_user_task():
    """Explicit Thread usage patterns."""
    with threading.Lock():
        time.sleep(0.1)
    return True

def io_bound_task(path):
    """Explicit IO patterns."""
    with open(path, 'r') as f:
        return f.read()

# Closure test helper
def make_closure_with_lock():
    lock = threading.Lock()
    def inner():
        # This function "closes over" a lock
        with lock:
            pass
    return inner

class TestFunctionAnalyzer:
    
    def setup_method(self):
        FunctionAnalyzer.clear_cache()

    def test_detects_cpu_bound(self):
        """Should detect numpy/math usage as CPU bound."""
        analysis = FunctionAnalyzer.analyze(heavy_numpy_task)
        assert analysis.estimated_weight == 'cpu_bound'
        assert analysis.prefers_processes is True

    def test_detects_io_bound(self):
        """Should detect file operations as IO bound."""
        analysis = FunctionAnalyzer.analyze(io_bound_task)
        assert analysis.estimated_weight == 'io_bound'
        # IO bound usually prefers threads, but your logic defaults to processes for safety
        # unless explicitly unpicklable.

    def test_detects_thread_usage_in_source(self):
        """Should see 'threading.Lock' in source code."""
        analysis = FunctionAnalyzer.analyze(thread_user_task)
        assert analysis.uses_threads is True
        # If it uses threads internally, it prefers processes (to get its own GIL)
        assert analysis.prefers_processes is True

    def test_detects_thread_usage_in_closure(self):
        """Should detect a Lock object inside the function's closure."""
        func = make_closure_with_lock()
        analysis = FunctionAnalyzer.analyze(func)
        assert analysis.has_closures is True
        assert analysis.uses_threads is True

    def test_analyzes_lambdas(self):
        """Should correctly handle lambdas."""
        func = lambda x: x + 1
        analysis = FunctionAnalyzer.analyze(func)
        assert analysis.is_lambda is True
        # Cloudpickle is available in dev deps, so this should be picklable
        assert analysis.is_picklable is True

    def test_caching_works(self):
        """Second call should return cached object."""
        a1 = FunctionAnalyzer.analyze(simple_cpu_task)
        a2 = FunctionAnalyzer.analyze(simple_cpu_task)
        assert a1 is a2

    def test_fails_gracefully_no_source(self):
        """Builtins have no source code, should not crash."""
        analysis = FunctionAnalyzer.analyze(sum)
        assert analysis.estimated_weight == 'unknown'
        assert analysis.is_picklable is True