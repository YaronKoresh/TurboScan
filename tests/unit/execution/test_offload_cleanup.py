import glob
import os
import pickle
import tempfile

import pytest

from turboscan import HyperBoost
from turboscan.execution.hyper_boost import _discard_offloaded_files
from turboscan.execution.utils import _offload_large_result


def _ipc_files():
    return set(
        glob.glob(os.path.join(tempfile.gettempdir(), "turbo_ipc_*.bin"))
    )


class TestDiscardOffloadedFiles:
    def test_removes_files_of_offloaded_results_only(self) -> None:
        path = _offload_large_result([1, 2, 3], pickle)
        raw = [None, ("success", 5), ("offloaded", path), "not a tuple"]

        _discard_offloaded_files(raw, lambda r: r)

        assert not os.path.exists(path)

    def test_ignores_already_removed_files(self) -> None:
        path = _offload_large_result([1], pickle)
        os.remove(path)

        _discard_offloaded_files([("offloaded", path)], lambda r: r)


class TestFailedRunLeavesNoTempFiles:
    def test_error_in_one_item_removes_offloaded_files_of_others(self) -> None:
        before = _ipc_files()

        def work(i):
            if i == 0:
                raise ValueError("boom")
            return b"x" * (11 * 1024 * 1024)

        with pytest.raises(ValueError, match="boom"):
            HyperBoost.run(work, [0, 1, 2, 3], quiet=True, use_cache=False)

        assert _ipc_files() - before == set()


class _Unpicklable:
    def __reduce__(self):
        raise RuntimeError("cannot pickle")


class TestFailedOffloadRemovesItsFile:
    def test_serializer_error_removes_the_temp_file(self) -> None:
        before = _ipc_files()

        with pytest.raises(RuntimeError, match="cannot pickle"):
            _offload_large_result(_Unpicklable(), pickle)

        assert _ipc_files() - before == set()

    def test_successful_offload_keeps_the_file(self) -> None:
        path = _offload_large_result([1, 2, 3], pickle)
        try:
            with open(path, "rb") as f:
                assert pickle.load(f) == [1, 2, 3]
        finally:
            os.remove(path)
