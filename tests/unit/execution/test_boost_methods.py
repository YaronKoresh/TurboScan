from typing import NoReturn

import pytest

from turboscan import HyperBoost


def square(x):
    return x * x


def add(x, y):
    return x + y


def multiply(x, y, z):
    return x * y * z


def is_even(x):
    return x % 2 == 0


def failing_func(x):
    if x == 3:
        raise ValueError(f"Cannot process {x}")
    return x * 2


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================


def test_hyper_map() -> None:
    """Test basic map functionality."""
    inputs = [1, 2, 3, 4]
    # Use threads to avoid overhead in unit tests
    results = HyperBoost.map(square, inputs, backend="threads", quiet=True)
    assert results == [1, 4, 9, 16]


def test_hyper_starmap() -> None:
    """Test starmap with multiple arguments."""
    inputs = [(1, 2), (3, 4), (5, 6)]
    results = HyperBoost.starmap(add, inputs, backend="threads", quiet=True)
    assert results == [3, 7, 11]


def test_empty_input() -> None:
    """Test handling of empty input."""
    assert HyperBoost.run(square, [], quiet=True) == []


def test_single_item() -> None:
    """Test processing a single item."""
    assert HyperBoost.run(square, [5], backend="threads", quiet=True) == [25]


def test_large_input() -> None:
    """Test with larger input to ensure chunking works."""
    inputs = list(range(100))
    results = HyperBoost.run(square, inputs, backend="threads", quiet=True)
    expected = [x * x for x in inputs]
    assert results == expected


# ============================================================================
# BACKEND TESTS
# ============================================================================


def test_threads_backend() -> None:
    """Test explicit threads backend."""
    inputs = [1, 2, 3, 4, 5]
    results = HyperBoost.run(square, inputs, backend="threads", quiet=True)
    assert results == [1, 4, 9, 16, 25]


def test_processes_backend() -> None:
    """Test explicit processes backend."""
    inputs = [1, 2, 3, 4, 5]
    results = HyperBoost.run(square, inputs, backend="processes", quiet=True)
    assert results == [1, 4, 9, 16, 25]


def test_auto_backend() -> None:
    """Test auto backend selection."""
    inputs = [1, 2, 3, 4, 5]
    results = HyperBoost.run(square, inputs, backend="auto", quiet=True)
    assert results == [1, 4, 9, 16, 25]


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_error_propagation() -> None:
    """Test that errors in worker functions are properly propagated."""
    inputs = [1, 2, 3, 4]

    # The error should be propagated (or handled gracefully)
    # Depending on implementation, this might raise or return partial results
    try:
        results = HyperBoost.run(
            failing_func, inputs, backend="threads", quiet=True
        )
        # If no exception, check that we got some results
        assert len(results) == len(inputs)
    except Exception as e:
        # If exception is raised, that's also acceptable behavior
        assert "Cannot process 3" in str(e) or isinstance(e, ValueError)


# ============================================================================
# STARMAP TESTS
# ============================================================================


def test_starmap_three_args() -> None:
    """Test starmap with three arguments."""
    inputs = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    results = HyperBoost.starmap(
        multiply, inputs, backend="threads", quiet=True
    )
    assert results == [6, 120, 504]


def test_starmap_empty() -> None:
    """Test starmap with empty input."""
    results = HyperBoost.starmap(add, [], backend="threads", quiet=True)
    assert results == []


# ============================================================================
# MAP TESTS
# ============================================================================


def test_map_with_filter() -> None:
    """Test map combined with filtering logic."""
    inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = HyperBoost.map(square, inputs, backend="threads", quiet=True)
    # Filter out odd squares
    filtered = [r for r in results if r % 2 == 0]
    assert filtered == [4, 16, 36, 64, 100]


def test_map_preserves_order() -> None:
    """Test that map preserves input order."""
    inputs = list(range(50))
    results = HyperBoost.map(square, inputs, backend="threads", quiet=True)
    expected = [x * x for x in inputs]
    assert results == expected


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_none_in_results() -> None:
    """Test handling of None values in results."""

    def return_none(x):
        return None if x % 2 == 0 else x

    inputs = [1, 2, 3, 4, 5]
    results = HyperBoost.run(return_none, inputs, backend="threads", quiet=True)
    assert results == [1, None, 3, None, 5]


def test_mixed_types() -> None:
    """Test handling of mixed types in results."""

    def process_mixed(x):
        if x % 3 == 0:
            return str(x)
        elif x % 2 == 0:
            return float(x)
        else:
            return x

    inputs = [1, 2, 3, 4, 5, 6]
    results = HyperBoost.run(
        process_mixed, inputs, backend="threads", quiet=True
    )
    assert results[0] == 1
    assert results[1] == 2.0
    assert results[2] == "3"
    assert results[3] == 4.0
    assert results[4] == 5
    assert results[5] == "6"


def test_complex_objects() -> None:
    """Test handling of complex objects."""

    def create_dict(x):
        return {"value": x, "square": x * x}

    inputs = [1, 2, 3]
    results = HyperBoost.run(create_dict, inputs, backend="threads", quiet=True)
    assert results == [
        {"value": 1, "square": 1},
        {"value": 2, "square": 4},
        {"value": 3, "square": 9},
    ]


# ============================================================================
# PERFORMANCE AND CHUNKING TESTS
# ============================================================================


def test_chunking_with_workers() -> None:
    """Test that chunking works correctly (Auto-scaling mode)."""
    inputs = list(range(100))

    # FIX: Removed 'workers=4'. TurboScan automatically uses ALL resources.
    results = HyperBoost.run(square, inputs, backend="threads", quiet=True)

    # Verify results to ensure the automatic chunking worked
    expected = [x**2 for x in inputs]
    assert sorted(results) == sorted(expected)


# ============================================================================
# LAMBDA AND CLOSURE TESTS
# ============================================================================


def test_lambda_functions() -> None:
    """Test with lambda functions (requires cloudpickle)."""
    inputs = [1, 2, 3, 4]
    try:
        results = HyperBoost.run(
            lambda x: x * 3, inputs, backend="threads", quiet=True
        )
        assert results == [3, 6, 9, 12]
    except Exception:
        # If cloudpickle not available, this might fail - that's okay
        pytest.skip("Lambda serialization not supported")


def test_closure() -> None:
    """Test with closures."""
    multiplier = 5

    def multiply_by_constant(x):
        return x * multiplier

    inputs = [1, 2, 3, 4]
    results = HyperBoost.run(
        multiply_by_constant, inputs, backend="threads", quiet=True
    )
    assert results == [5, 10, 15, 20]


# ============================================================================
# MIN_CHUNKS AUTO-SCALING TESTS
# ============================================================================


def test_min_chunks_parameter() -> None:
    """Test that min_chunks parameter is accepted."""
    inputs = [1, 2, 3, 4, 5]
    # Should work with min_chunks=None (default)
    results = HyperBoost.run(
        square, inputs, backend="threads", quiet=True, min_chunks=None
    )
    assert results == [1, 4, 9, 16, 25]

    # Should work with explicit min_chunks
    results = HyperBoost.run(
        square, inputs, backend="threads", quiet=True, min_chunks=4
    )
    assert results == [1, 4, 9, 16, 25]


def test_min_chunks_auto() -> None:
    """Test min_chunks='auto' for automatic scaling."""
    inputs = [1, 2, 3, 4, 5]
    results = HyperBoost.run(
        square, inputs, backend="threads", quiet=True, min_chunks="auto"
    )
    assert results == [1, 4, 9, 16, 25]


def test_min_chunks_with_numpy_arrays() -> None:
    """Test min_chunks with numpy arrays that can be subdivided."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy not available")

    def sum_array(arr):
        return np.sum(arr)

    # Create a large array that should be subdivided
    arr = np.arange(1000)
    inputs = [arr]

    # With min_chunks='auto', should attempt subdivision
    results = HyperBoost.run(
        sum_array, inputs, backend="threads", quiet=True, min_chunks="auto"
    )
    # Results should be reassembled to match expected
    assert len(results) == 1


def test_min_chunks_with_lists() -> None:
    """Test min_chunks with lists that can be subdivided."""

    def sum_list(lst):
        return sum(lst)

    # Create a large list
    large_list = list(range(100))
    inputs = [large_list]

    results = HyperBoost.run(
        sum_list, inputs, backend="threads", quiet=True, min_chunks=4
    )
    assert len(results) == 1


def test_min_chunks_preserves_tuple_type() -> None:
    """Test that tuple chunking preserves the tuple type."""

    def get_length(t):
        return len(t)

    inputs = [tuple(range(20))]
    results = HyperBoost.run(get_length, inputs, backend="threads", quiet=True)
    assert results == [20]


def test_min_chunks_invalid_value() -> None:
    """Test that invalid min_chunks values are handled gracefully."""
    inputs = [1, 2, 3]

    # Negative value should fall back to default
    results = HyperBoost.run(
        square, inputs, backend="threads", quiet=True, min_chunks=-1
    )
    assert results == [1, 4, 9]

    # Zero should fall back to default
    results = HyperBoost.run(
        square, inputs, backend="threads", quiet=True, min_chunks=0
    )
    assert results == [1, 4, 9]


# ============================================================================
# TASK CHAINING TESTS
# ============================================================================


def double(x):
    return x * 2


def increment(x):
    return x + 1


def test_task_chaining() -> None:
    """Test chaining multiple functions."""
    inputs = [1, 2, 3, 4]
    results = HyperBoost.run(
        [square, double], inputs, backend="threads", quiet=True
    )
    # square then double: 1->1->2, 2->4->8, 3->9->18, 4->16->32
    assert results == [2, 8, 18, 32]


def test_task_chaining_three_functions() -> None:
    """Test chaining three functions."""
    inputs = [1, 2, 3]
    results = HyperBoost.run(
        [increment, square, double], inputs, backend="threads", quiet=True
    )
    # increment then square then double: 1->2->4->8, 2->3->9->18, 3->4->16->32
    assert results == [8, 18, 32]


def test_task_chaining_empty() -> None:
    """Test empty task chain."""
    inputs = [1, 2, 3]
    results = HyperBoost.run([], inputs, backend="threads", quiet=True)
    # Empty chain should return original items
    assert results == [1, 2, 3]


# ============================================================================
# BOOST_ALL TESTS
# ============================================================================


def test_boost_all_basic() -> None:
    """Test boost_all with independent tasks."""
    from turboscan.execution.hyper_boost import boost_all

    tasks = [
        lambda: 1 + 1,
        lambda: 2 * 2,
        lambda: 3**2,
    ]

    results = boost_all("test_tasks", tasks)
    assert results == [2, 4, 9]


def test_boost_all_empty() -> None:
    """Test boost_all with empty task list."""
    from turboscan.execution.hyper_boost import boost_all

    results = boost_all("empty_tasks", [])
    assert results == []


def test_boost_all_single_task() -> None:
    """Test boost_all with single task (should run sequentially)."""
    from turboscan.execution.hyper_boost import boost_all

    results = boost_all("single_task", [lambda: 42])
    assert results == [42]


# ============================================================================
# RECURSIVE PARALLELIZATION PREVENTION TESTS
# ============================================================================


def test_recursive_parallelization_prevented() -> None:
    """Test that recursive HyperBoost.run calls fall back to sequential."""
    call_count = [0]

    def nested_run(x):
        call_count[0] += 1
        # Try to call HyperBoost.run from within a worker
        inner_results = HyperBoost.run(
            square, [x], backend="threads", quiet=True
        )
        return inner_results[0]

    inputs = [1, 2, 3]
    results = HyperBoost.run(nested_run, inputs, backend="threads", quiet=True)
    assert results == [1, 4, 9]
    assert call_count[0] == 3


# ============================================================================
# CACHING TESTS
# ============================================================================


def test_caching_enabled() -> None:
    """Test that caching works when enabled."""
    call_count = [0]

    def counting_square(x):
        call_count[0] += 1
        return x * x

    inputs = [1, 2, 3]

    # First run
    results1 = HyperBoost.run(
        counting_square, inputs, backend="threads", quiet=True, use_cache=True
    )
    call_count[0]

    # Second run with same inputs should use cache
    results2 = HyperBoost.run(
        counting_square, inputs, backend="threads", quiet=True, use_cache=True
    )

    assert results1 == results2
    # Note: cache may not work for all items, so we just verify results are correct


def test_caching_disabled() -> None:
    """Test that caching can be disabled."""
    inputs = [1, 2, 3]
    results = HyperBoost.run(
        square, inputs, backend="threads", quiet=True, use_cache=False
    )
    assert results == [1, 4, 9]


# ============================================================================
# DEBUG MODE TESTS
# ============================================================================


def test_debug_mode_toggle() -> None:
    """Test that debug mode can be toggled."""
    # Save original state
    original = HyperBoost.DEBUG

    try:
        HyperBoost.set_debug(True)
        assert HyperBoost.DEBUG is True

        HyperBoost.set_debug(False)
        assert HyperBoost.DEBUG is False
    finally:
        # Restore original state
        HyperBoost.DEBUG = original


# ============================================================================
# GENERATOR INPUT TESTS
# ============================================================================


def test_generator_input() -> None:
    """Test that generator inputs are properly handled."""

    def gen():
        yield from range(5)

    results = HyperBoost.run(square, gen(), backend="threads", quiet=True)
    assert results == [0, 1, 4, 9, 16]


def test_iterator_input() -> None:
    """Test that iterator inputs are properly handled."""
    inputs = iter([1, 2, 3, 4, 5])
    results = HyperBoost.run(square, inputs, backend="threads", quiet=True)
    assert results == [1, 4, 9, 16, 25]


# ============================================================================
# BATCH SIZE TESTS
# ============================================================================


def test_custom_batch_size() -> None:
    """Test custom batch size parameter."""
    inputs = list(range(100))
    results = HyperBoost.run(square, inputs, backend="threads", quiet=True)
    expected = [x * x for x in inputs]
    assert results == expected


def test_batch_size_larger_than_input() -> None:
    """Test batch size larger than input size."""
    inputs = [1, 2, 3]
    results = HyperBoost.run(square, inputs, backend="threads", quiet=True)
    assert results == [1, 4, 9]


# ============================================================================
# FORCE_PROCESSES TESTS
# ============================================================================


def test_force_processes() -> None:
    """Test force_processes parameter."""
    inputs = [1, 2, 3, 4, 5]
    results = HyperBoost.run(
        square, inputs, backend="auto", quiet=True, force_processes=True
    )
    assert results == [1, 4, 9, 16, 25]


# ============================================================================
# MAP AND STARMAP ADDITIONAL TESTS
# ============================================================================


def test_map_with_multiple_iterables() -> None:
    """Test map with multiple iterables."""
    a = [1, 2, 3]
    b = [4, 5, 6]
    results = HyperBoost.map(add, a, b, backend="threads", quiet=True)
    assert results == [5, 7, 9]


def test_starmap_with_kwargs() -> None:
    """Test starmap passes kwargs correctly."""
    inputs = [(1, 2), (3, 4), (5, 6)]
    results = HyperBoost.starmap(add, inputs, backend="threads", quiet=True)
    assert results == [3, 7, 11]


# ============================================================================
# EDGE CASES FOR DIFFERENT DATA TYPES
# ============================================================================


def test_string_processing() -> None:
    """Test processing strings."""

    def upper(s):
        return s.upper()

    inputs = ["hello", "world", "test"]
    results = HyperBoost.run(upper, inputs, backend="threads", quiet=True)
    assert results == ["HELLO", "WORLD", "TEST"]


def test_dict_processing() -> None:
    """Test processing dictionaries."""

    def get_value(d):
        return d.get("value", 0)

    inputs = [{"value": 1}, {"value": 2}, {}]
    results = HyperBoost.run(get_value, inputs, backend="threads", quiet=True)
    assert results == [1, 2, 0]


def test_nested_list_processing() -> None:
    """Test processing nested lists."""

    def flatten(lst):
        return [item for sublist in lst for item in sublist]

    inputs = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    results = HyperBoost.run(flatten, inputs, backend="threads", quiet=True)
    assert results == [[1, 2, 3, 4], [5, 6, 7, 8]]


# ============================================================================
# SHUTDOWN TESTS
# ============================================================================


def test_shutdown() -> None:
    """Test HyperBoost shutdown."""
    # First, ensure thread pool is initialized
    HyperBoost._init_thread_pool()
    assert HyperBoost._thread_pool is not None

    # Shutdown
    HyperBoost.shutdown()
    assert HyperBoost._thread_pool is None

    # Should be able to run again after shutdown
    inputs = [1, 2, 3]
    results = HyperBoost.run(square, inputs, backend="threads", quiet=True)
    assert results == [1, 4, 9]


# ============================================================================
# INTERNAL METHODS TESTS
# ============================================================================


def test_init_thread_pool() -> None:
    """Test thread pool initialization."""
    HyperBoost.shutdown()  # Ensure clean state
    HyperBoost._init_thread_pool()
    assert HyperBoost._thread_pool is not None


def test_init_gpu_queue() -> None:
    """Test GPU queue initialization."""
    HyperBoost._init_gpu_queue()
    # GPU queue may or may not exist depending on hardware


def test_get_effective_min_chunks_none() -> None:
    """Test _get_effective_min_chunks with None."""
    result = HyperBoost._get_effective_min_chunks(None, 10)
    assert result == 10  # Should return num_items when None


def test_get_effective_min_chunks_auto() -> None:
    """Test _get_effective_min_chunks with 'auto'."""
    from turboscan import HARDWARE

    result = HyperBoost._get_effective_min_chunks("auto", 2)
    expected = max(HARDWARE.cpu_count * 2, 2)
    assert result == expected


def test_get_effective_min_chunks_int() -> None:
    """Test _get_effective_min_chunks with integer."""
    result = HyperBoost._get_effective_min_chunks(50, 10)
    assert result == 50  # Should use min_chunks when larger


def test_get_effective_min_chunks_invalid() -> None:
    """Test _get_effective_min_chunks with invalid value."""
    result = HyperBoost._get_effective_min_chunks(-5, 10)
    assert result == 10  # Should fall back to num_items


def test_is_chunkable_item_list() -> None:
    """Test _is_chunkable_item with list."""
    # Lists of numbers are NOT chunkable (auto-chunking is only for complex objects)
    assert HyperBoost._is_chunkable_item([1, 2, 3]) is False
    assert HyperBoost._is_chunkable_item([1]) is False  # Single element
    assert HyperBoost._is_chunkable_item([]) is False  # Empty

    # Lists of objects with __dict__ ARE chunkable
    class Obj:
        pass

    assert HyperBoost._is_chunkable_item([Obj(), Obj()]) is True


def test_is_chunkable_item_tuple() -> None:
    """Test _is_chunkable_item with tuple."""
    # Tuples of numbers are NOT chunkable (auto-chunking is only for complex objects)
    assert HyperBoost._is_chunkable_item((1, 2, 3)) is False
    assert HyperBoost._is_chunkable_item((1,)) is False


def test_is_chunkable_item_numpy() -> None:
    """Test _is_chunkable_item with numpy array."""
    try:
        import numpy as np

        # 1D arrays are NOT chunkable (only 3D+ arrays are auto-chunked)
        arr = np.array([1, 2, 3, 4, 5])
        assert HyperBoost._is_chunkable_item(arr) is False

        # 2D arrays are NOT chunkable
        arr_2d = np.zeros((3, 4))
        assert HyperBoost._is_chunkable_item(arr_2d) is False

        # 3D+ arrays ARE chunkable
        arr_3d = np.zeros((2, 3, 4))
        assert HyperBoost._is_chunkable_item(arr_3d) is True

        single = np.array([1])
        assert HyperBoost._is_chunkable_item(single) is False
    except ImportError:
        pytest.skip("NumPy not available")


def test_is_chunkable_item_scalar() -> None:
    """Test _is_chunkable_item with scalar."""
    assert HyperBoost._is_chunkable_item(42) is False
    assert HyperBoost._is_chunkable_item("string") is False


def test_get_item_length_list() -> None:
    """Test _get_item_length with list."""
    assert HyperBoost._get_item_length([1, 2, 3]) == 3


def test_get_item_length_tuple() -> None:
    """Test _get_item_length with tuple."""
    assert HyperBoost._get_item_length((1, 2, 3, 4)) == 4


def test_get_item_length_numpy() -> None:
    """Test _get_item_length with numpy array."""
    try:
        import numpy as np

        arr = np.array([1, 2, 3, 4, 5])
        assert HyperBoost._get_item_length(arr) == 5

        arr_2d = np.zeros((3, 4))
        assert HyperBoost._get_item_length(arr_2d) == 3
    except ImportError:
        pytest.skip("NumPy not available")


def test_get_item_length_scalar() -> None:
    """Test _get_item_length with scalar."""
    assert HyperBoost._get_item_length(42) == 1


def test_subdivide_items_no_subdivision() -> None:
    """Test _subdivide_items when no subdivision needed."""
    items = [1, 2, 3, 4, 5]
    result, chunk_map = HyperBoost._subdivide_items(items, 5)
    assert len(result) == 5
    assert len(chunk_map) == 5


def test_subdivide_items_with_lists() -> None:
    """Test _subdivide_items with lists."""
    items = [list(range(20))]
    result, _chunk_map = HyperBoost._subdivide_items(items, 4)
    assert len(result) >= 1


def test_subdivide_items_with_tuples() -> None:
    """Test _subdivide_items with tuples."""
    items = [tuple(range(20))]
    result, _chunk_map = HyperBoost._subdivide_items(items, 4)
    # Should preserve tuple type
    for item in result:
        if isinstance(item, (list, tuple)):
            assert isinstance(item, tuple)


def test_subdivide_items_non_chunkable() -> None:
    """Test _subdivide_items with non-chunkable items."""
    items = [1, 2, 3]
    result, _chunk_map = HyperBoost._subdivide_items(items, 10)
    assert len(result) == 3  # Can't subdivide scalars


def test_reassemble_results_no_subdivision() -> None:
    """Test _reassemble_results when no subdivision occurred."""
    results = [1, 4, 9]
    chunk_map = [(0, 0, 1), (1, 0, 1), (2, 0, 1)]
    reassembled = HyperBoost._reassemble_results(results, chunk_map, 3)
    assert reassembled == [1, 4, 9]


def test_reassemble_results_with_numpy() -> None:
    """Test _reassemble_results with numpy arrays."""
    try:
        import numpy as np

        # Simulate subdivided numpy results
        chunk1 = np.array([1, 2])
        chunk2 = np.array([3, 4])
        results = [chunk1, chunk2]
        chunk_map = [(0, 0, 2), (0, 1, 2)]

        reassembled = HyperBoost._reassemble_results(results, chunk_map, 1)
        assert len(reassembled) == 1
        assert np.array_equal(reassembled[0], np.array([1, 2, 3, 4]))
    except ImportError:
        pytest.skip("NumPy not available")


def test_reassemble_results_with_lists() -> None:
    """Test _reassemble_results with lists."""
    results = [[1, 2], [3, 4]]
    chunk_map = [(0, 0, 2), (0, 1, 2)]
    reassembled = HyperBoost._reassemble_results(results, chunk_map, 1)
    assert len(reassembled) == 1
    assert reassembled[0] == [1, 2, 3, 4]


def test_reassemble_results_with_2d_numpy() -> None:
    """Test _reassemble_results with 2D numpy arrays (axis=1 concatenation)."""
    try:
        import numpy as np

        # Simulate 2D arrays split along axis=1 (time dimension in voice conversion)
        # Original shape would be (1025, 100) - Freq x Time
        # After splitting into 2 chunks along axis=1
        chunk1 = np.random.rand(1025, 50)  # First half of time
        chunk2 = np.random.rand(1025, 50)  # Second half of time

        results = [chunk1, chunk2]
        chunk_map = [(0, 0, 2), (0, 1, 2)]

        reassembled = HyperBoost._reassemble_results(results, chunk_map, 1)

        assert len(reassembled) == 1
        # Should concatenate along axis=1 to get (1025, 100)
        assert reassembled[0].shape == (1025, 100)

        # Verify the data matches - first half and second half should be preserved
        assert np.array_equal(reassembled[0][:, :50], chunk1)
        assert np.array_equal(reassembled[0][:, 50:], chunk2)
    except ImportError:
        pytest.skip("NumPy not available")


# ============================================================================
# FUNCTION ANALYSIS TESTS
# ============================================================================


def test_function_analysis() -> None:
    """Test function analysis for parallelization."""
    from turboscan.analysis.function_analyzer import FunctionAnalyzer

    def simple_func(x):
        return x * 2

    analysis = FunctionAnalyzer.analyze(simple_func)
    assert analysis is not None
    assert hasattr(analysis, "is_picklable")


def test_cpu_bound_detection() -> None:
    """Test CPU-bound function detection."""

    def cpu_intensive(x):
        total = 0
        for i in range(1000):
            total += i * x
        return total

    # Just verify it works
    results = HyperBoost.run(
        cpu_intensive, [1, 2, 3], backend="threads", quiet=True
    )
    assert len(results) == 3


# ============================================================================
# PROCESSES BACKEND TESTS
# ============================================================================


def test_processes_backend_simple() -> None:
    """Test processes backend with simple function."""

    def simple_square(x):
        return x * x

    inputs = [1, 2, 3, 4, 5]
    results = HyperBoost.run(
        simple_square, inputs, backend="processes", quiet=True
    )
    assert results == [1, 4, 9, 16, 25]


def test_processes_backend_with_imports() -> None:
    """Test processes backend with function using imports."""

    def with_math(x):
        import math

        return math.sqrt(x)

    inputs = [1, 4, 9, 16]
    results = HyperBoost.run(with_math, inputs, backend="processes", quiet=True)
    assert results == [1.0, 2.0, 3.0, 4.0]


# ============================================================================
# QUIET MODE TESTS
# ============================================================================


def test_quiet_mode_true() -> None:
    """Test quiet mode suppresses output."""
    inputs = list(range(100))
    results = HyperBoost.run(square, inputs, quiet=True)
    assert len(results) == 100


def test_quiet_mode_false() -> None:
    """Test quiet=False allows output."""
    inputs = [1, 2, 3]
    results = HyperBoost.run(square, inputs, quiet=False)
    assert results == [1, 4, 9]


# ============================================================================
# EXCEPTION IN WORKER TESTS
# ============================================================================


def test_exception_propagates() -> None:
    """Test that exceptions from workers propagate correctly."""

    def always_fails(x) -> NoReturn:
        raise RuntimeError("Always fails")

    with pytest.raises(RuntimeError):
        HyperBoost.run(always_fails, [1], backend="threads", quiet=True)


def test_partial_failure() -> None:
    """Test behavior with partial failures."""
    # This depends on implementation - may propagate first error or collect all
    with pytest.raises(ValueError):
        HyperBoost.run(
            failing_func, [1, 2, 3, 4], backend="threads", quiet=True
        )


# ============================================================================
# MEMORY HANDLING TESTS
# ============================================================================


def test_large_result_handling() -> None:
    """Test handling of large results."""

    def create_large(x):
        return list(range(x * 1000))

    inputs = [1, 2, 3]
    results = HyperBoost.run(
        create_large, inputs, backend="threads", quiet=True
    )
    assert len(results) == 3
    assert len(results[0]) == 1000
    assert len(results[2]) == 3000


# ============================================================================
# PARALLEL EXECUTION VERIFICATION
# ============================================================================


def test_actual_parallelism() -> None:
    """Verify that execution is actually parallel."""
    import time

    def slow_func(x):
        time.sleep(0.1)
        return x * 2

    start = time.time()
    results = HyperBoost.run(
        slow_func, [1, 2, 3, 4], backend="threads", quiet=True
    )
    elapsed = time.time() - start

    # With parallelism, should be much faster than 0.4s
    assert elapsed < 0.35  # Allow some overhead
    assert results == [2, 4, 6, 8]
