import pytest
import threading
import pickle
from turboscan.execution.utils import _clean_for_pickle, patch_class_for_multiprocessing
import functools

try:
    import cloudpickle
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False


# ============================================================================
# TEST OBJECTS
# ============================================================================


class ComplexObject:
    def __init__(self, val):
        self.val = val
        # Use a Thread Lock, which cannot be pickled (even by cloudpickle usually)
        self.real_lock = threading.Lock()
        self.nice_lambda = lambda: "I survive!"

    @functools.lru_cache(maxsize=10)
    def cached_method(self, x):
        return self.val + x


class SimplePicklable:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value


class NestedComplexObject:
    def __init__(self, val):
        self.data = {'value': val, 'lock': threading.Lock()}
        self.nested_obj = ComplexObject(val)


# ============================================================================
# CLEANING TESTS
# ============================================================================


def test_clean_for_pickle_handles_attributes():
    """Verify true unpicklables are stripped, but cloudpickle-safe items stay."""
    original = ComplexObject(42)

    # Cleaning should return a NEW object because pickle.dumps(original) fails
    cleaned = _clean_for_pickle(original)

    assert cleaned is not original
    assert cleaned.val == 42

    # The Thread Lock must be gone (it's truly unpicklable)
    assert not hasattr(cleaned, 'real_lock') or cleaned.real_lock is None

    # The lambda SHOULD survive because TurboScan uses cloudpickle fallback!
    assert hasattr(cleaned, 'nice_lambda')
    assert cleaned.nice_lambda() == "I survive!"


def test_original_object_intact():
    """Verify the original object is NOT modified by cleaning."""
    original = ComplexObject(10)
    _clean_for_pickle(original)

    # Original should still have everything
    assert hasattr(original, 'real_lock')
    assert isinstance(original.real_lock, type(threading.Lock()))


def test_clean_simple_picklable():
    """Test that already picklable objects are returned as-is."""
    original = SimplePicklable(100)

    # Should be picklable already
    try:
        pickle.dumps(original)
        is_picklable = True
    except Exception:
        is_picklable = False

    if is_picklable:
        cleaned = _clean_for_pickle(original)
        # Might be same object or a copy, but should have same value
        assert cleaned.value == 100


def test_clean_nested_complex():
    """Test cleaning of nested complex objects."""
    original = NestedComplexObject(50)

    cleaned = _clean_for_pickle(original)

    # Should have copied the object
    assert cleaned is not original

    # Value should be preserved
    if hasattr(cleaned, 'nested_obj'):
        assert cleaned.nested_obj.val == 50


def test_clean_for_pickle_primitives():
    """Test that primitives don't need cleaning."""
    # These should all work without issues
    assert _clean_for_pickle(42) == 42
    assert _clean_for_pickle("hello") == "hello"
    assert _clean_for_pickle([1, 2, 3]) == [1, 2, 3]
    assert _clean_for_pickle({'key': 'value'}) == {'key': 'value'}


def test_clean_for_pickle_none():
    """Test handling of None."""
    assert _clean_for_pickle(None) is None


# ============================================================================
# PATCH CLASS TESTS
# ============================================================================


def test_patch_class_removes_lru_cache():
    """Verify @lru_cache is removed from class definition."""
    # Create a fresh class to avoid state pollution
    class TestClass:
        @functools.lru_cache(maxsize=10)
        def cached_method(self, x):
            return x * 2

    # Ensure it starts as an lru_cache
    assert hasattr(TestClass.cached_method, 'cache_info')

    # Patch it
    patch_class_for_multiprocessing(TestClass)

    # Should no longer have cache_info
    assert not hasattr(TestClass.cached_method, 'cache_info')


def test_patch_class_preserves_functionality():
    """Test that patched class still works correctly."""
    class TestClass:
        def __init__(self, base):
            self.base = base

        @functools.lru_cache(maxsize=10)
        def compute(self, x):
            return self.base + x

    # Patch the class
    patch_class_for_multiprocessing(TestClass)

    # Should still work
    obj = TestClass(10)
    assert obj.compute(5) == 15
    assert obj.compute(10) == 20


def test_patch_class_multiple_cached_methods():
    """Test patching a class with multiple cached methods."""
    class MultiCacheClass:
        @functools.lru_cache(maxsize=5)
        def method_a(self, x):
            return x * 2

        @functools.lru_cache(maxsize=10)
        def method_b(self, x):
            return x * 3

        def normal_method(self, x):
            return x * 4

    # Patch it
    patch_class_for_multiprocessing(MultiCacheClass)

    # Both cached methods should be patched
    assert not hasattr(MultiCacheClass.method_a, 'cache_info')
    assert not hasattr(MultiCacheClass.method_b, 'cache_info')

    # Normal method should still work
    obj = MultiCacheClass()
    assert obj.normal_method(5) == 20


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================


@pytest.mark.skipif(not CLOUDPICKLE_AVAILABLE, reason="cloudpickle not available")
def test_cloudpickle_lambda():
    """Test that cloudpickle can serialize lambdas."""
    f = lambda x: x * 2
    serialized = cloudpickle.dumps(f)
    deserialized = cloudpickle.loads(serialized)
    assert deserialized(5) == 10


@pytest.mark.skipif(not CLOUDPICKLE_AVAILABLE, reason="cloudpickle not available")
def test_cloudpickle_closure():
    """Test that cloudpickle can serialize closures."""
    multiplier = 3

    def closure_func(x):
        return x * multiplier

    serialized = cloudpickle.dumps(closure_func)
    deserialized = cloudpickle.loads(serialized)
    assert deserialized(4) == 12


def test_pickle_simple_function():
    """Test that simple functions can be pickled."""
    def simple_func(x):
        return x + 1

    import cloudpickle
    serialized = cloudpickle.dumps(simple_func)
    deserialized = cloudpickle.loads(serialized)
    assert deserialized(5) == 6


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_clean_circular_reference():
    """Test handling of circular references."""
    class CircularClass:
        def __init__(self):
            self.value = 42
            self.self_ref = None

    obj = CircularClass()
    obj.self_ref = obj

    # Should handle circular reference gracefully
    cleaned = _clean_for_pickle(obj)
    assert cleaned.value == 42


def test_clean_large_object():
    """Test cleaning of large objects."""
    class LargeObject:
        def __init__(self):
            self.data = list(range(10000))
            self.lock = threading.Lock()

    obj = LargeObject()
    cleaned = _clean_for_pickle(obj)

    # Should preserve data
    assert len(cleaned.data) == 10000
    # Lock should be removed
    assert not hasattr(cleaned, 'lock') or cleaned.lock is None


# ============================================================================
# ADDITIONAL SERIALIZATION TESTS
# ============================================================================


def test_clean_for_pickle_with_queue():
    """Test cleaning objects with queue.Queue."""
    import queue
    
    class ObjectWithQueue:
        def __init__(self):
            self.q = queue.Queue()
            self.value = 42
    
    obj = ObjectWithQueue()
    cleaned = _clean_for_pickle(obj)
    assert cleaned.value == 42


def test_clean_for_pickle_with_file_handle():
    """Test cleaning objects with file handles."""
    import tempfile
    
    class ObjectWithFile:
        def __init__(self, f):
            self.file = f
            self.name = "test"
    
    with tempfile.TemporaryFile() as f:
        obj = ObjectWithFile(f)
        cleaned = _clean_for_pickle(obj)
        assert cleaned.name == "test"


def test_prepare_for_serialization_basic():
    """Test prepare_for_serialization with basic types."""
    from turboscan.execution.utils import prepare_for_serialization
    
    # Basic types should pass through unchanged
    assert prepare_for_serialization(42) == 42
    assert prepare_for_serialization("hello") == "hello"
    assert prepare_for_serialization([1, 2, 3]) == [1, 2, 3]


def test_prepare_for_serialization_nested():
    """Test prepare_for_serialization with nested structures."""
    from turboscan.execution.utils import prepare_for_serialization
    
    nested = {
        'list': [1, 2, 3],
        'dict': {'a': 1, 'b': 2},
        'tuple': (4, 5, 6)
    }
    result = prepare_for_serialization(nested)
    assert result['list'] == [1, 2, 3]
    assert result['dict']['a'] == 1


def test_is_lru_cache_method_with_non_cache():
    """Test _is_lru_cache_method with non-cached functions."""
    from turboscan.execution.utils import _is_lru_cache_method
    
    def regular_func(x):
        return x * 2
    
    assert _is_lru_cache_method(regular_func) is False
    assert _is_lru_cache_method(None) is False
    assert _is_lru_cache_method(42) is False


def test_is_lru_cache_method_with_cache():
    """Test _is_lru_cache_method with cached functions."""
    from turboscan.execution.utils import _is_lru_cache_method
    
    @functools.lru_cache(maxsize=10)
    def cached_func(x):
        return x * 2
    
    assert _is_lru_cache_method(cached_func) is True


def test_find_lru_cache_methods():
    """Test _find_lru_cache_methods function."""
    from turboscan.execution.utils import _find_lru_cache_methods
    
    class TestClass:
        @functools.lru_cache(maxsize=5)
        def cached_method(self, x):
            return x
        
        def normal_method(self, x):
            return x * 2
    
    methods = _find_lru_cache_methods(TestClass)
    assert 'cached_method' in methods
    assert 'normal_method' not in methods


def test_patch_module_for_multiprocessing():
    """Test patching an entire module."""
    from turboscan.execution.utils import patch_module_for_multiprocessing
    import types
    
    # Create a mock module
    mock_module = types.ModuleType('test_module')
    mock_module.__name__ = 'test_module'
    
    # Add a class with lru_cache
    class TestClass:
        @functools.lru_cache(maxsize=5)
        def cached(self, x):
            return x
    
    TestClass.__module__ = 'test_module'
    mock_module.TestClass = TestClass
    
    # Patch it
    count = patch_module_for_multiprocessing(mock_module)
    # Should have patched the class


def test_worker_execute_success():
    """Test _hyperboost_worker_execute with successful execution."""
    from turboscan.execution.utils import _hyperboost_worker_execute
    
    def simple_func(x):
        return x * 2
    
    work_item = (simple_func, 5, None, {})
    result = _hyperboost_worker_execute(work_item)
    
    assert result[0] == 'success'
    assert result[1] == 10


def test_worker_execute_with_kwargs():
    """Test _hyperboost_worker_execute with keyword arguments."""
    from turboscan.execution.utils import _hyperboost_worker_execute
    
    def func_with_kwargs(x, multiplier=1):
        return x * multiplier
    
    work_item = (func_with_kwargs, 5, None, {'multiplier': 3})
    result = _hyperboost_worker_execute(work_item)
    
    assert result[0] == 'success'
    assert result[1] == 15


def test_worker_execute_error():
    """Test _hyperboost_worker_execute with error."""
    from turboscan.execution.utils import _hyperboost_worker_execute
    
    def failing_func(x):
        raise ValueError("Test error")
    
    work_item = (failing_func, 5, None, {})
    result = _hyperboost_worker_execute(work_item)
    
    assert result[0] == 'error'
    assert 'ValueError' in result[1]['type']


@pytest.mark.skipif(not CLOUDPICKLE_AVAILABLE, reason="cloudpickle not available")
def test_cloudpickle_worker():
    """Test _hyperboost_cloudpickle_worker."""
    from turboscan.execution.utils import _hyperboost_cloudpickle_worker
    
    def simple_func(x):
        return x * 2
    
    # Serialize the work item
    work_bytes = cloudpickle.dumps((simple_func, 5, None, {}))
    
    # Execute
    result_bytes = _hyperboost_cloudpickle_worker(work_bytes)
    result = cloudpickle.loads(result_bytes)
    
    assert result[0] == 'success'
    assert result[1] == 10


def test_clean_for_pickle_with_dataclass():
    """Test cleaning dataclass objects."""
    from dataclasses import dataclass
    
    @dataclass
    class DataItem:
        value: int
        name: str
    
    item = DataItem(value=42, name="test")
    cleaned = _clean_for_pickle(item)
    
    assert cleaned.value == 42
    assert cleaned.name == "test"


def test_clean_for_pickle_with_slots():
    """Test cleaning objects with __slots__."""
    class SlottedClass:
        __slots__ = ['x', 'y']
        
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    obj = SlottedClass(1, 2)
    # Should not crash
    cleaned = _clean_for_pickle(obj)