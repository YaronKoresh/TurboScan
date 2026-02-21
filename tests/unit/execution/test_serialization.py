import functools
import pickle
import threading
from typing import NoReturn

import pytest

from turboscan.execution.utils import (
    _clean_for_pickle,
    patch_class_for_multiprocessing,
)

try:
    import cloudpickle

    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False


class ComplexObject:
    def __init__(self, val) -> None:
        self.val = val

        self.real_lock = threading.Lock()
        self.nice_lambda = lambda: "I survive!"

    @functools.lru_cache(maxsize=10)
    def cached_method(self, x):
        return self.val + x


class SimplePicklable:
    def __init__(self, value) -> None:
        self.value = value

    def get_value(self):
        return self.value


class NestedComplexObject:
    def __init__(self, val) -> None:
        self.data = {"value": val, "lock": threading.Lock()}
        self.nested_obj = ComplexObject(val)


def test_clean_for_pickle_handles_attributes() -> None:
    """Verify true unpicklables are stripped, but cloudpickle-safe items stay."""
    original = ComplexObject(42)

    cleaned = _clean_for_pickle(original)

    assert cleaned is not original
    assert cleaned.val == 42

    assert not hasattr(cleaned, "real_lock") or cleaned.real_lock is None

    assert hasattr(cleaned, "nice_lambda")
    assert cleaned.nice_lambda() == "I survive!"


def test_original_object_intact() -> None:
    """Verify the original object is NOT modified by cleaning."""
    original = ComplexObject(10)
    _clean_for_pickle(original)

    assert hasattr(original, "real_lock")
    assert isinstance(original.real_lock, type(threading.Lock()))


def test_clean_simple_picklable() -> None:
    """Test that already picklable objects are returned as-is."""
    original = SimplePicklable(100)

    try:
        pickle.dumps(original)
        is_picklable = True
    except Exception:
        is_picklable = False

    if is_picklable:
        cleaned = _clean_for_pickle(original)

        assert cleaned.value == 100


def test_clean_nested_complex() -> None:
    """Test cleaning of nested complex objects."""
    original = NestedComplexObject(50)

    cleaned = _clean_for_pickle(original)

    assert cleaned is not original

    if hasattr(cleaned, "nested_obj"):
        assert cleaned.nested_obj.val == 50


def test_clean_for_pickle_primitives() -> None:
    """Test that primitives don't need cleaning."""

    assert _clean_for_pickle(42) == 42
    assert _clean_for_pickle("hello") == "hello"
    assert _clean_for_pickle([1, 2, 3]) == [1, 2, 3]
    assert _clean_for_pickle({"key": "value"}) == {"key": "value"}


def test_clean_for_pickle_none() -> None:
    """Test handling of None."""
    assert _clean_for_pickle(None) is None


def test_patch_class_removes_lru_cache() -> None:
    """Verify @lru_cache is removed from class definition."""

    class TestClass:
        @functools.lru_cache(maxsize=10)
        def cached_method(self, x):
            return x * 2

    assert hasattr(TestClass.cached_method, "cache_info")

    patch_class_for_multiprocessing(TestClass)

    assert not hasattr(TestClass.cached_method, "cache_info")


def test_patch_class_preserves_functionality() -> None:
    """Test that patched class still works correctly."""

    class TestClass:
        def __init__(self, base) -> None:
            self.base = base

        @functools.lru_cache(maxsize=10)
        def compute(self, x):
            return self.base + x

    patch_class_for_multiprocessing(TestClass)

    obj = TestClass(10)
    assert obj.compute(5) == 15
    assert obj.compute(10) == 20


def test_patch_class_multiple_cached_methods() -> None:
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

    patch_class_for_multiprocessing(MultiCacheClass)

    assert not hasattr(MultiCacheClass.method_a, "cache_info")
    assert not hasattr(MultiCacheClass.method_b, "cache_info")

    obj = MultiCacheClass()
    assert obj.normal_method(5) == 20


@pytest.mark.skipif(
    not CLOUDPICKLE_AVAILABLE, reason="cloudpickle not available"
)
def test_cloudpickle_lambda() -> None:
    """Test that cloudpickle can serialize lambdas."""

    def f(x):
        return x * 2

    serialized = cloudpickle.dumps(f)
    deserialized = cloudpickle.loads(serialized)
    assert deserialized(5) == 10


@pytest.mark.skipif(
    not CLOUDPICKLE_AVAILABLE, reason="cloudpickle not available"
)
def test_cloudpickle_closure() -> None:
    """Test that cloudpickle can serialize closures."""
    multiplier = 3

    def closure_func(x):
        return x * multiplier

    serialized = cloudpickle.dumps(closure_func)
    deserialized = cloudpickle.loads(serialized)
    assert deserialized(4) == 12


def test_pickle_simple_function() -> None:
    """Test that simple functions can be pickled."""

    def simple_func(x):
        return x + 1

    import cloudpickle

    serialized = cloudpickle.dumps(simple_func)
    deserialized = cloudpickle.loads(serialized)
    assert deserialized(5) == 6


def test_clean_circular_reference() -> None:
    """Test handling of circular references."""

    class CircularClass:
        def __init__(self) -> None:
            self.value = 42
            self.self_ref = None

    obj = CircularClass()
    obj.self_ref = obj

    cleaned = _clean_for_pickle(obj)
    assert cleaned.value == 42


def test_clean_large_object() -> None:
    """Test cleaning of large objects."""

    class LargeObject:
        def __init__(self) -> None:
            self.data = list(range(10000))
            self.lock = threading.Lock()

    obj = LargeObject()
    cleaned = _clean_for_pickle(obj)

    assert len(cleaned.data) == 10000

    assert not hasattr(cleaned, "lock") or cleaned.lock is None


def test_clean_for_pickle_with_queue() -> None:
    """Test cleaning objects with queue.Queue."""
    import queue

    class ObjectWithQueue:
        def __init__(self) -> None:
            self.q = queue.Queue()
            self.value = 42

    obj = ObjectWithQueue()
    cleaned = _clean_for_pickle(obj)
    assert cleaned.value == 42


def test_clean_for_pickle_with_file_handle() -> None:
    """Test cleaning objects with file handles."""
    import tempfile

    class ObjectWithFile:
        def __init__(self, f) -> None:
            self.file = f
            self.name = "test"

    with tempfile.TemporaryFile() as f:
        obj = ObjectWithFile(f)
        cleaned = _clean_for_pickle(obj)
        assert cleaned.name == "test"


def test_prepare_for_serialization_basic() -> None:
    """Test prepare_for_serialization with basic types."""
    from turboscan.execution.utils import prepare_for_serialization

    assert prepare_for_serialization(42) == 42
    assert prepare_for_serialization("hello") == "hello"
    assert prepare_for_serialization([1, 2, 3]) == [1, 2, 3]


def test_prepare_for_serialization_nested() -> None:
    """Test prepare_for_serialization with nested structures."""
    from turboscan.execution.utils import prepare_for_serialization

    nested = {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}, "tuple": (4, 5, 6)}
    result = prepare_for_serialization(nested)
    assert result["list"] == [1, 2, 3]
    assert result["dict"]["a"] == 1


def test_is_lru_cache_method_with_non_cache() -> None:
    """Test _is_lru_cache_method with non-cached functions."""
    from turboscan.execution.utils import _is_lru_cache_method

    def regular_func(x):
        return x * 2

    assert _is_lru_cache_method(regular_func) is False
    assert _is_lru_cache_method(None) is False
    assert _is_lru_cache_method(42) is False


def test_is_lru_cache_method_with_cache() -> None:
    """Test _is_lru_cache_method with cached functions."""
    from turboscan.execution.utils import _is_lru_cache_method

    @functools.lru_cache(maxsize=10)
    def cached_func(x):
        return x * 2

    assert _is_lru_cache_method(cached_func) is True


def test_find_lru_cache_methods() -> None:
    """Test _find_lru_cache_methods function."""
    from turboscan.execution.utils import _find_lru_cache_methods

    class TestClass:
        @functools.lru_cache(maxsize=5)
        def cached_method(self, x):
            return x

        def normal_method(self, x):
            return x * 2

    methods = _find_lru_cache_methods(TestClass)
    assert "cached_method" in methods
    assert "normal_method" not in methods


def test_patch_module_for_multiprocessing() -> None:
    """Test patching an entire module."""
    import types

    from turboscan.execution.utils import patch_module_for_multiprocessing

    mock_module = types.ModuleType("test_module")
    mock_module.__name__ = "test_module"

    class TestClass:
        @functools.lru_cache(maxsize=5)
        def cached(self, x):
            return x

    TestClass.__module__ = "test_module"
    mock_module.TestClass = TestClass

    patch_module_for_multiprocessing(mock_module)


def test_worker_execute_success() -> None:
    """Test _hyperboost_worker_execute with successful execution."""
    from turboscan.execution.utils import _hyperboost_worker_execute

    def simple_func(x):
        return x * 2

    work_item = (simple_func, 5, None, {})
    result = _hyperboost_worker_execute(work_item)

    assert result[0] == "success"
    assert result[1] == 10


def test_worker_execute_with_kwargs() -> None:
    """Test _hyperboost_worker_execute with keyword arguments."""
    from turboscan.execution.utils import _hyperboost_worker_execute

    def func_with_kwargs(x, multiplier=1):
        return x * multiplier

    work_item = (func_with_kwargs, 5, None, {"multiplier": 3})
    result = _hyperboost_worker_execute(work_item)

    assert result[0] == "success"
    assert result[1] == 15


def test_worker_execute_error() -> None:
    """Test _hyperboost_worker_execute with error."""
    from turboscan.execution.utils import _hyperboost_worker_execute

    def failing_func(x) -> NoReturn:
        raise ValueError("Test error")

    work_item = (failing_func, 5, None, {})
    result = _hyperboost_worker_execute(work_item)

    assert result[0] == "error"
    assert "ValueError" in result[1]["type"]


@pytest.mark.skipif(
    not CLOUDPICKLE_AVAILABLE, reason="cloudpickle not available"
)
def test_cloudpickle_worker() -> None:
    """Test _hyperboost_cloudpickle_worker."""
    from turboscan.execution.utils import _hyperboost_cloudpickle_worker

    def simple_func(x):
        return x * 2

    work_bytes = cloudpickle.dumps((simple_func, 5, None, {}))

    result_bytes = _hyperboost_cloudpickle_worker(work_bytes)
    result = cloudpickle.loads(result_bytes)

    assert result[0] == "success"
    assert result[1] == 10


def test_clean_for_pickle_with_dataclass() -> None:
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


def test_clean_for_pickle_with_slots() -> None:
    """Test cleaning objects with __slots__."""

    class SlottedClass:
        __slots__ = ["x", "y"]

        def __init__(self, x, y) -> None:
            self.x = x
            self.y = y

    obj = SlottedClass(1, 2)

    _clean_for_pickle(obj)
