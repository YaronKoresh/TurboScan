from unittest.mock import patch

import pytest

from turboscan.executor.executor import HyperExecutor


class TestHyperExecutor:
    def test_execute_flow(self, tmp_path) -> None:
        """Test the full execution pipeline (Audit -> Optimize -> JIT -> Run)."""
        # 1. Create a real temporary script file
        script_file = tmp_path / "script.py"
        script_file.write_text("print('hello')", encoding="utf-8")

        # 2. Setup Mocks using context managers to avoid decorator conflicts
        with patch(
            "turboscan.executor.executor.HyperAuditor"
        ) as MockAuditor, patch(
            "turboscan.executor.executor.HyperBoost"
        ), patch("turboscan.executor.executor.JITInjector") as MockJIT, patch(
            "turboscan.executor.executor.HyperAutoParallelizer"
        ) as MockParallelizer, patch(
            "turboscan.executor.executor.FAST_READER"
        ) as MockReader, patch("turboscan.executor.executor.NUMBA_AVAIL", True):
            # Configure Audit to PASS (0 errors)
            MockAuditor.return_value.run.return_value = (0, 0)

            # Mock reading the file content
            MockReader.read_file.return_value = "print('hello')"

            # Initialize & Run
            executor = HyperExecutor(
                str(script_file), audit_first=True, optimize=True
            )
            executor.execute()

            # Verify the pipeline steps were called
            MockAuditor.assert_called()  # Audit ran
            MockParallelizer.return_value.visit.assert_called()  # Optimization ran
            MockJIT.return_value.inject.assert_called()  # JIT ran

    def test_execute_stops_on_audit_failure(self, tmp_path) -> None:
        """Should exit if audit finds errors and user says 'n'."""
        # 1. Create a dummy script file
        script_file = tmp_path / "bad_script.py"
        script_file.touch()

        # 2. Mock Auditor to find errors, Input to say 'n', and RICH_AVAIL to False
        with patch(
            "turboscan.executor.executor.HyperAuditor"
        ) as MockAuditor, patch("builtins.input", return_value="n"), patch(
            "turboscan.executor.executor.RICH_AVAIL", False
        ):
            # Simulate audit finding 5 errors
            MockAuditor.return_value.run.return_value = (5, 0)

            executor = HyperExecutor(str(script_file), audit_first=True)

            # Expect exit code 1 due to user aborting
            with pytest.raises(SystemExit) as excinfo:
                executor.execute()

            assert excinfo.value.code == 1
