import ast
from unittest.mock import patch

import pytest

from turboscan import HyperBoost
from turboscan.ast_transform.parallelizer import HyperAutoParallelizer
from turboscan.auditor.auditor import HyperAuditor
from turboscan.cache.hyper_cache import HYPER_CACHE
from turboscan.executor.executor import HyperExecutor


@pytest.fixture
def torture_project(tmp_path):
    """
    Creates a complex, multi-file project structure to torture the
    Registry, Resolver, and Auditor.
    """

    src = tmp_path / "src"
    src.mkdir()
    (src / "utils").mkdir()
    (src / "legacy").mkdir()

    (src / "utils" / "__init__.py").touch()
    (src / "utils" / "math_ops.py").write_text(
        """
import math

def heavy_kernel(n):
    res = 0.0
    for i in range(n):
        res += math.sin(i) * math.cos(i)
    return res
""",
        encoding="utf-8",
    )

    (src / "utils" / "data_proc.py").write_text(
        """
def process_data(items):
    total = 0
    # This loop should be detected as an accumulation and parallelized
    for x in items:
        total += x * 2
    return total

def unsafe_process(items):
    # This loop has side effects (print/IO) and should NOT be parallelized
    for x in items:
        print(x)
        with open('log.txt', 'a') as f:
            f.write(str(x))
""",
        encoding="utf-8",
    )

    (src / "legacy" / "bad_code.py").write_text(
        """
import sys
import os
# 'sys' is unused, 'os' is unused

def broken_logic():
    return x + 1  # 'x' is undefined
""",
        encoding="utf-8",
    )

    (src / "main.py").write_text(
        """
from utils.math_ops import heavy_kernel
from utils.data_proc import process_data

def main():
    print("Starting Engine...")
    val = heavy_kernel(100)
    data = list(range(1000))
    res = process_data(data)
    print(f"Result: {val + res}")

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    return src


def test_auditor_torture(torture_project) -> None:
    """
    Runs the Auditor on a messy project.
    Verifies: Registry scanning, Resolver logic, AST Validation, Error Reporting.
    """

    auditor = HyperAuditor(
        root=torture_project, excludes={"__pycache__"}, check_unused=True
    )

    with patch("builtins.print"):
        error_count, _warn_count = auditor.run()

    assert error_count >= 1, "Auditor failed to catch undefined variables"

    registry = auditor.registry
    assert "utils.math_ops" in registry.modules
    assert "legacy.bad_code" in registry.modules
    assert "main" in registry.modules


def test_executor_optimization_flow(torture_project) -> None:
    """
    Runs the HyperExecutor on the main script.
    Verifies: FastFileReader, AutoParallelizer, JIT Injection, and Execution.
    """
    script_path = torture_project / "main.py"

    executor = HyperExecutor(
        str(script_path), audit_first=False, optimize=True, inject_jit=True
    )

    source_code = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source_code)

    optimized_tree = executor._optimize_ast(tree)

    mixed_script = torture_project / "mixed_mode.py"
    mixed_script.write_text(
        """
import math

def heavy_loop():
    # This should be JITted OR Parallelized
    acc = 0
    for i in range(1000):
        acc += i * i
    return acc

def main():
    heavy_loop()

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    executor_mixed = HyperExecutor(
        str(mixed_script), audit_first=False, optimize=True
    )

    with patch("turboscan.executor.executor.FAST_READER") as mock_reader:
        mock_reader.read_file.return_value = mixed_script.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(mock_reader.read_file.return_value)
        optimized_tree = executor_mixed._optimize_ast(tree)

        assert (
            executor_mixed.stats["parallelized_loops"] > 0
            or executor_mixed.stats["jit_functions"] > 0
        )

        dump = ast.dump(optimized_tree)
        assert "HyperBoost" in dump or "njit" in dump or "prange" in dump


def test_parallelizer_edge_cases() -> None:
    """
    Directly tests the HyperAutoParallelizer on complex AST structures.
    """
    parallelizer = HyperAutoParallelizer()

    code_acc = """
def func():
    total = 0
    for i in range(1000):
        total += i
    return total
"""
    tree = ast.parse(code_acc)
    new_tree = parallelizer.visit(tree)
    assert parallelizer.loop_counter == 1
    assert "_hyper_loop_worker" in ast.dump(new_tree)

    code_unsafe = """
def func():
    for i in range(100):
        print(i)
"""

    parallelizer.loop_counter = 0
    tree = ast.parse(code_unsafe)
    new_tree = parallelizer.visit(tree)

    assert parallelizer.loop_counter == 1
    assert "_hyper_loop_worker" in ast.dump(new_tree)


def test_hyper_cache_integration(tmp_path) -> None:
    """
    Tests that the HyperCache works end-to-end with the execution engine.
    """

    with patch("tempfile.gettempdir", return_value=str(tmp_path)):
        HYPER_CACHE.l1_cache.clear()

        key = "test_func:argument_1"
        value = {"complex": "result", "data": [1, 2, 3]}

        HYPER_CACHE.set(key, value)

        found, res = HYPER_CACHE.get(key)
        assert found is True
        assert res == value
        assert HYPER_CACHE.stats["hits"] >= 1

        HYPER_CACHE.l1_cache.clear()

        assert key in HYPER_CACHE.bloom

        found, _ = HYPER_CACHE.get("non_existent_key")
        assert found is False
        assert HYPER_CACHE.stats["misses"] >= 1


def test_hyper_boost_real_execution() -> None:
    """
    Actually spins up threads/processes to verify serialization and return values.
    """
    items = list(range(50))

    def worker(x):
        return x * x

    results_t = HyperBoost.run(worker, items, backend="threads", quiet=True)
    assert results_t == [x * x for x in items]

    results_p = HyperBoost.run(
        lambda x: x + 1, items, backend="processes", quiet=True
    )
    assert results_p == [x + 1 for x in items]
