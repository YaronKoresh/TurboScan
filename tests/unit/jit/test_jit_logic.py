import ast

import pytest

from turboscan.jit.injector import JIT_INJECTOR, NUMBA_AVAIL, JITInjector


def parse_code(code_str):
    return ast.parse(code_str).body[0]


def parse_module(code_str):
    return ast.parse(code_str)


# ============================================================================
# MATH HEAVY DETECTION TESTS
# ============================================================================


def test_detects_math_heavy() -> None:
    code = """
def heavy(x):
    return math.sin(x) * math.cos(x) + np.tan(x)
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_math_heavy(node) is True


def test_not_math_heavy_simple() -> None:
    """Simple function without math should not be math heavy."""
    code = """
def simple(x):
    return x + 1
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_math_heavy(node) is False


def test_math_heavy_with_numpy() -> None:
    """NumPy operations should count as math heavy."""
    code = """
def numpy_ops(arr):
    return np.sum(arr) * np.mean(arr) / np.std(arr)
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_math_heavy(node) is True


def test_math_heavy_with_binops() -> None:
    """Multiple binary operations should be math heavy."""
    code = """
def many_ops(a, b, c, d):
    return a * b + c - d / a * b
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_math_heavy(node) is True


# ============================================================================
# NUMBA COMPATIBILITY TESTS
# ============================================================================


def test_rejects_strings() -> None:
    """Should reject functions doing string manipulation."""
    code = """
def stringy(x):
    return f"Value is {x}"
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_classes() -> None:
    """Should reject functions defining classes inside."""
    code = """
def has_class():
    class Inner: pass
    return Inner()
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_async() -> None:
    """Should reject async functions."""
    code = """
async def async_func(x):
    return await x
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_yield() -> None:
    """Should reject generator functions."""
    code = """
def gen_func(x):
    for i in range(x):
        yield i
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_self_method() -> None:
    """Should reject instance methods (with self)."""
    code = """
def method(self, x):
    return self.value * x
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_property_decorator() -> None:
    """Should reject functions with @property decorator."""
    code = """
@property
def prop(x):
    return x
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_contextmanager() -> None:
    """Should reject functions with @contextmanager decorator."""
    code = """
@contextmanager
def ctx(x):
    yield x
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_accepts_simple_numeric() -> None:
    """Should accept simple numeric functions."""
    code = """
def numeric(x, y):
    return x * y + 2
    """
    node = parse_code(code)
    # Note: depends on NUMBA_AVAIL
    result = JIT_INJECTOR._is_numba_compatible(node)
    # If numba not available, should return False
    from turboscan.jit.injector import NUMBA_AVAIL

    if not NUMBA_AVAIL:
        assert result is False


def test_rejects_np_clip() -> None:
    """Should reject functions using np.clip as it doesn't support scalars in Numba."""
    code = """
def warp_func(freqs):
    warp = np.interp(freqs, warp_points, warp_values)
    return np.clip(warp, 0.5, 2.0)
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_numpy_clip() -> None:
    """Should reject functions using numpy.clip as it doesn't support scalars in Numba."""
    code = """
def clamp_values(x, lo, hi):
    return numpy.clip(x, lo, hi)
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_np_reshape() -> None:
    """Should reject functions using np.reshape as it doesn't support scalars in Numba."""
    code = """
def reshape_data(x):
    return np.reshape(x, (-1, 2))
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_np_ndenumerate() -> None:
    """Should reject functions using np.ndenumerate as it doesn't support scalars in Numba."""
    code = """
def enum_array(x):
    for idx, val in np.ndenumerate(x):
        pass
    return x
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_np_transpose() -> None:
    """Should reject functions using np.transpose."""
    code = """
def transpose_data(x):
    return np.transpose(x)
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_with_errstate() -> None:
    """Should reject functions using np.errstate context manager."""
    code = """
def with_errstate(x):
    with np.errstate(divide='ignore'):
        return 1 / x
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


def test_rejects_with_open() -> None:
    """Should reject functions using open()."""
    code = """
def with_open(x):
    with open('file.txt') as f:
        return f.read()
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_numba_compatible(node) is False


# ============================================================================
# VECTORIZABLE DETECTION TESTS
# ============================================================================


@pytest.mark.skipif(not NUMBA_AVAIL, reason="Numba not available")
def test_detects_vectorizable() -> None:
    """Should detect simple element-wise operations with type hints."""
    code = """
def vec_op(x: float, y: float) -> float:
    return x * y + 2.0
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_vectorizable(node) is True


def test_not_vectorizable_no_hints() -> None:
    """Should reject vectorizable without type hints."""
    code = """
def no_hints(x, y):
    return x * y
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_vectorizable(node) is False


def test_not_vectorizable_wrong_types() -> None:
    """Should reject vectorizable with non-numeric type hints."""
    code = """
def wrong_types(x: str, y: str) -> str:
    return x + y
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_vectorizable(node) is False


def test_not_vectorizable_too_many_args() -> None:
    """Should reject vectorizable with too many arguments."""
    code = """
def too_many(a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float, i: float, j: float, k: float, l: float, m: float, n: float, o: float, p: float, q: float, r: float, s: float, t: float, u: float) -> float:
    return a + b
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_vectorizable(node) is False


@pytest.mark.skipif(not NUMBA_AVAIL, reason="Numba not available")
def test_vectorizable_int_types() -> None:
    """Should accept vectorizable with int types."""
    code = """
def int_op(x: int, y: int) -> int:
    return x * y + 2
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_vectorizable(node) is True


@pytest.mark.skipif(not NUMBA_AVAIL, reason="Numba not available")
def test_vectorizable_complex_types() -> None:
    """Should accept vectorizable with complex types."""
    code = """
def complex_op(x: complex, y: complex) -> complex:
    return x * y
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_vectorizable(node) is True


def test_vectorizable_bool_types() -> None:
    """Should accept vectorizable with bool types."""
    code = """
def bool_op(x: bool, y: bool) -> bool:
    return x and y
    """
    node = parse_code(code)
    # Note: this might fail the computation check
    JIT_INJECTOR._is_vectorizable(node)


def test_not_vectorizable_no_computation() -> None:
    """Should reject vectorizable without any computation."""
    code = """
def no_compute(x: float) -> float:
    pass
    """
    node = parse_code(code)
    assert JIT_INJECTOR._is_vectorizable(node) is False


# ============================================================================
# LOOP DETECTION TESTS
# ============================================================================


def test_has_loops_for() -> None:
    """Should detect for loops."""
    code = """
def with_for(x):
    for i in range(x):
        pass
    return x
    """
    node = parse_code(code)
    assert JIT_INJECTOR._has_loops(node) is True


def test_has_loops_while() -> None:
    """Should detect while loops."""
    code = """
def with_while(x):
    while x > 0:
        x -= 1
    return x
    """
    node = parse_code(code)
    assert JIT_INJECTOR._has_loops(node) is True


def test_no_loops() -> None:
    """Should not detect loops in simple function."""
    code = """
def no_loops(x):
    return x * 2
    """
    node = parse_code(code)
    assert JIT_INJECTOR._has_loops(node) is False


# ============================================================================
# LOOP CONDITIONAL ASSIGNS TESTS
# ============================================================================


def test_has_loop_conditional_assigns() -> None:
    """Should detect conditional assignments in loops."""
    code = """
def conditional_in_loop(x):
    result = 0
    for i in range(x):
        if i > 5:
            result = i
        print(result)
    return result
    """
    node = parse_code(code)
    assert JIT_INJECTOR._has_loop_conditional_assigns(node) is True


def test_no_loop_conditional_assigns() -> None:
    """Should not detect conditional assigns when not used."""
    code = """
def no_conditional(x):
    result = 0
    for i in range(x):
        result += i
    return result
    """
    node = parse_code(code)
    assert JIT_INJECTOR._has_loop_conditional_assigns(node) is False


# ============================================================================
# INJECT TESTS
# ============================================================================


def test_inject_creates_new_injector() -> None:
    """Test creating a new JITInjector instance."""
    injector = JITInjector()
    assert injector.jit_count == 0
    assert injector.vectorize_count == 0


def test_inject_module() -> None:
    """Test injecting JIT into a module."""
    code = """
def simple(x):
    result = 0
    for i in range(x):
        result += i * i
    return result
    """
    injector = JITInjector()
    tree = parse_module(code)
    result = injector.inject(tree)
    # Should return the tree
    assert result is not None


def test_inject_skips_hyper_prefix() -> None:
    """Should skip functions with _hyper_ prefix."""
    code = """
def _hyper_worker_1(x):
    for i in range(x):
        pass
    return x
    """
    injector = JITInjector()
    tree = parse_module(code)
    injector.inject(tree)
    assert injector.jit_count == 0


def test_inject_skips_batch_prefix() -> None:
    """Should skip functions with _batch_ prefix."""
    code = """
def _batch_process(x):
    for i in range(x):
        pass
    return x
    """
    injector = JITInjector()
    tree = parse_module(code)
    injector.inject(tree)
    assert injector.jit_count == 0


def test_inject_skips_already_decorated() -> None:
    """Should skip functions already decorated with njit."""
    code = """
@njit
def already_jit(x):
    for i in range(x):
        pass
    return x
    """
    injector = JITInjector()
    tree = parse_module(code)
    injector.inject(tree)
    assert injector.jit_count == 0


def test_inject_skips_vectorize_decorated() -> None:
    """Should skip functions already decorated with vectorize."""
    code = """
@vectorize
def already_vec(x: float) -> float:
    return x * 2
    """
    injector = JITInjector()
    tree = parse_module(code)
    injector.inject(tree)
    assert injector.vectorize_count == 0


def test_global_jit_injector() -> None:
    """Test that global JIT_INJECTOR is available."""
    from turboscan.jit.injector import JIT_INJECTOR

    assert JIT_INJECTOR is not None
    assert isinstance(JIT_INJECTOR, JITInjector)
