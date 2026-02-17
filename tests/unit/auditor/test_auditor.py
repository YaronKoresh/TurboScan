from pathlib import Path
from unittest.mock import patch

import pytest

from turboscan.auditor.auditor import HyperAuditor


class TestHyperAuditor:
    @pytest.fixture
    def mock_deps(self):
        """Mock out the heavy dependencies (Registry, Resolver, Boost)."""
        with patch("turboscan.auditor.auditor.HyperRegistry") as reg, patch(
            "turboscan.auditor.auditor.HyperResolver"
        ) as res, patch(
            "turboscan.auditor.auditor.HyperValidator"
        ) as val, patch("turboscan.auditor.auditor.HyperBoost") as boost, patch(
            "turboscan.auditor.auditor.FAST_READER"
        ) as reader:
            # Setup Registry Mock
            reg_instance = reg.return_value
            reg_instance.modules = {}
            reg_instance.files_map = {Path("test.py"): "test"}
            reg_instance._file_contents = {}

            # Setup Reader Mock to return valid code string
            # This fixes the "compile() arg 1 must be a string" error
            reader.read_file.return_value = "x = 1"

            # Setup Boost to run synchronously for tests
            def fake_run(func, items, **kwargs):
                return [func(i) for i in items]

            boost.run.side_effect = fake_run

            yield reg, res, val, boost, reader

    def test_init_scans_registry(self, mock_deps) -> None:
        """Auditor should initialize and scan the registry."""
        reg_cls, _res_cls, _, _, _ = mock_deps
        root = Path("/project")

        HyperAuditor(root, excludes={"test_exclude"})

        reg_cls.assert_called_with(root, {"test_exclude"})
        reg_cls.return_value.scan.assert_called()

    def test_validate_file_valid(self, mock_deps) -> None:
        """Should return empty list for a clean file."""
        _, _, val_cls, _, _ = mock_deps

        val_instance = val_cls.return_value
        val_instance.errors = []
        val_instance.warnings = []
        val_instance.get_unused_imports.return_value = []

        auditor = HyperAuditor(Path("."), set())
        issues = auditor._validate_file(Path("clean.py"))

        assert issues == []

    def test_validate_file_detects_issues(self, mock_deps) -> None:
        """Should format errors, warnings, and unused imports correctly."""
        _, _, val_cls, _, _ = mock_deps

        val_instance = val_cls.return_value
        val_instance.errors = [(10, "Syntax error")]
        val_instance.warnings = [(5, "Deprecation warning")]
        val_instance.get_unused_imports.return_value = ["unused_sys"]

        auditor = HyperAuditor(Path("."), set(), check_unused=True)
        issues = auditor._validate_file(Path("dirty.py"))

        # Check formatting
        assert any("[ERROR]" in i and "Syntax error" in i for i in issues)
        assert any("[WARN]" in i and "Deprecation warning" in i for i in issues)
        assert any("[UNUSED]" in i and "unused_sys" in i for i in issues)

    def test_run_counts_errors(self, mock_deps) -> None:
        """Run method should aggregate counts from all files."""
        _, _, val_cls, _, _ = mock_deps

        val_instance = val_cls.return_value
        val_instance.errors = [(1, "Error")]
        val_instance.warnings = []
        val_instance.get_unused_imports.return_value = []

        auditor = HyperAuditor(Path("."), set())
        error_count, _warn_count = auditor.run()

        assert error_count == 1
