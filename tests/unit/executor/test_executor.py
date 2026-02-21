from unittest.mock import patch

import pytest

from turboscan.executor.executor import HyperExecutor


class TestHyperExecutor:
    def test_execute_flow(self, tmp_path) -> None:
        """Test the full execution pipeline (Audit -> Optimize -> JIT -> Run)."""

        script_file = tmp_path / "script.py"
        script_file.write_text("print('hello')", encoding="utf-8")

        with patch(
            "turboscan.executor.executor.HyperAuditor"
        ) as MockAuditor, patch(
            "turboscan.executor.executor.HyperBoost"
        ), patch("turboscan.executor.executor.JITInjector") as MockJIT, patch(
            "turboscan.executor.executor.HyperAutoParallelizer"
        ) as MockParallelizer, patch(
            "turboscan.executor.executor.FAST_READER"
        ) as MockReader, patch("turboscan.executor.executor.NUMBA_AVAIL", True):
            MockAuditor.return_value.run.return_value = (0, 0)

            MockReader.read_file.return_value = "print('hello')"

            executor = HyperExecutor(
                str(script_file), audit_first=True, optimize=True
            )
            executor.execute()

            MockAuditor.assert_called()
            MockParallelizer.return_value.visit.assert_called()
            MockJIT.return_value.inject.assert_called()

    def test_execute_stops_on_audit_failure(self, tmp_path) -> None:
        """Should exit if audit finds errors and user says 'n'."""

        script_file = tmp_path / "bad_script.py"
        script_file.touch()

        with patch(
            "turboscan.executor.executor.HyperAuditor"
        ) as MockAuditor, patch("builtins.input", return_value="n"), patch(
            "turboscan.executor.executor.RICH_AVAIL", False
        ):
            MockAuditor.return_value.run.return_value = (5, 0)

            executor = HyperExecutor(str(script_file), audit_first=True)

            with pytest.raises(SystemExit) as excinfo:
                executor.execute()

            assert excinfo.value.code == 1
