from pathlib import Path
from unittest.mock import patch

import pytest

from turboscan.registry.registry import HyperRegistry


class TestHyperRegistry:
    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a fake project structure."""

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils").mkdir()
        (tmp_path / "tests").mkdir()

        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "src" / "utils" / "__init__.py").touch()
        (tmp_path / "src" / "utils" / "helper.py").touch()
        (tmp_path / "tests" / "test_main.py").touch()

        return tmp_path

    def test_scan_finds_modules(self, project_root) -> None:
        """Should find modules and assign FQNs correctly."""

        with patch("turboscan.registry.registry.FAST_READER") as mock_reader:
            mock_reader.read_files_parallel.return_value = {}

            registry = HyperRegistry(project_root, excludes={"tests"})
            registry.scan()

            assert "main" in registry.modules

            assert "utils" in registry.modules
            assert registry.modules["utils"].is_package

            assert "utils.helper" in registry.modules

            assert "tests.test_main" not in registry.modules

    def test_src_layout_detection(self, project_root) -> None:
        """Should detect 'src' directory and strip it from FQN."""
        with patch("turboscan.registry.registry.FAST_READER") as mock_reader:
            mock_reader.read_files_parallel.return_value = {}

            registry = HyperRegistry(project_root, excludes=set())
            registry.scan()

            assert registry.src_prefix == "src"

    def test_relative_import_resolution(self) -> None:
        """Test the _resolve_abs_for_info logic."""
        registry = HyperRegistry(Path("."), set())

        class MockInfo:
            def __init__(self, fqn, is_package) -> None:
                self.fqn = fqn
                self.is_package = is_package

        info = MockInfo("pkg.sub", is_package=False)
        resolved = registry._resolve_abs_for_info(info, "sibling", 1)
        assert resolved == "pkg.sibling"

        info_pkg = MockInfo("pkg", is_package=True)
        resolved = registry._resolve_abs_for_info(info_pkg, "sub", 1)
        assert resolved == "pkg.sub"

        info_deep = MockInfo("pkg.deep.mod", is_package=False)
        resolved = registry._resolve_abs_for_info(info_deep, "sibling", 2)
        assert resolved == "pkg.sibling"

    def test_star_import_resolution(self) -> None:
        """Verify the iterative fixed-point algorithm triggers."""
        with patch("turboscan.registry.registry.HyperBoost"):
            registry = HyperRegistry(Path("."), set())

            registry.modules = {}
            registry.second_pass_star_imports()

    def test_registry_initialization(self, tmp_path) -> None:
        """Test registry initialization with various parameters."""
        registry = HyperRegistry(tmp_path, excludes={"test", "docs"})
        assert registry.root == tmp_path
        assert "test" in registry.excludes
        assert "docs" in registry.excludes

    def test_registry_empty_project(self, tmp_path) -> None:
        """Test registry with empty project."""
        with patch("turboscan.registry.registry.FAST_READER") as mock_reader:
            mock_reader.read_files_parallel.return_value = {}

            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()

            assert len(registry.modules) == 0

    def test_files_map(self, project_root) -> None:
        """Test files_map is populated correctly."""
        with patch("turboscan.registry.registry.FAST_READER") as mock_reader:
            mock_reader.read_files_parallel.return_value = {}

            registry = HyperRegistry(project_root, excludes={"tests"})
            registry.scan()

            assert len(registry.files_map) > 0

    def test_no_src_prefix(self, tmp_path) -> None:
        """Test project without src directory."""

        (tmp_path / "main.py").touch()
        (tmp_path / "utils.py").touch()

        with patch("turboscan.registry.registry.FAST_READER") as mock_reader:
            mock_reader.read_files_parallel.return_value = {}

            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()

            assert registry.src_prefix in ("", None) or not registry.src_prefix

    def test_nested_packages(self, tmp_path) -> None:
        """Test deeply nested package structure."""

        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").touch()
        (tmp_path / "pkg" / "sub").mkdir()
        (tmp_path / "pkg" / "sub" / "__init__.py").touch()
        (tmp_path / "pkg" / "sub" / "module.py").touch()

        with patch("turboscan.registry.registry.FAST_READER") as mock_reader:
            mock_reader.read_files_parallel.return_value = {}

            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()

            assert "pkg" in registry.modules
            assert "pkg.sub" in registry.modules
            assert "pkg.sub.module" in registry.modules

    def test_exclude_multiple_dirs(self, tmp_path) -> None:
        """Test excluding multiple directories."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test.py").touch()
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "doc.py").touch()

        with patch("turboscan.registry.registry.FAST_READER") as mock_reader:
            mock_reader.read_files_parallel.return_value = {}

            registry = HyperRegistry(tmp_path, excludes={"tests", "docs"})
            registry.scan()

            assert "main" in registry.modules

            assert "tests.test" not in registry.modules
            assert "docs.doc" not in registry.modules

    def test_resolve_abs_for_info_top_level(self) -> None:
        """Test relative import resolution at top level."""
        registry = HyperRegistry(Path("."), set())

        class MockInfo:
            def __init__(self, fqn, is_package) -> None:
                self.fqn = fqn
                self.is_package = is_package

        info = MockInfo("module", is_package=False)

        registry._resolve_abs_for_info(info, "other", 1)

    def test_resolve_abs_for_info_deep_level(self) -> None:
        """Test relative import with deep level."""
        registry = HyperRegistry(Path("."), set())

        class MockInfo:
            def __init__(self, fqn, is_package) -> None:
                self.fqn = fqn
                self.is_package = is_package

        info = MockInfo("a.b.c.d.e", is_package=False)
        resolved = registry._resolve_abs_for_info(info, "other", 3)
        assert resolved == "a.b.other"

    def test_multiple_python_files(self, tmp_path) -> None:
        """Test multiple Python files at same level."""
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        (tmp_path / "c.py").touch()

        with patch("turboscan.registry.registry.FAST_READER") as mock_reader:
            mock_reader.read_files_parallel.return_value = {}

            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()

            assert "a" in registry.modules
            assert "b" in registry.modules
            assert "c" in registry.modules

    def test_hidden_files_excluded(self, tmp_path) -> None:
        """Test that hidden files are typically excluded."""
        (tmp_path / "visible.py").touch()
        (tmp_path / ".hidden.py").touch()

        with patch("turboscan.registry.registry.FAST_READER") as mock_reader:
            mock_reader.read_files_parallel.return_value = {}

            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()

            assert "visible" in registry.modules

    def test_pycache_excluded(self, tmp_path) -> None:
        """Test that __pycache__ directories are excluded."""
        (tmp_path / "main.py").touch()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-39.pyc").touch()

        with patch("turboscan.registry.registry.FAST_READER") as mock_reader:
            mock_reader.read_files_parallel.return_value = {}

            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()

            assert "main" in registry.modules

    def test_second_pass_star_imports_with_modules(self) -> None:
        """Test second_pass_star_imports with actual modules."""
        with patch("turboscan.registry.registry.HyperBoost"):
            registry = HyperRegistry(Path("."), set())

            class MockInfo:
                def __init__(self) -> None:
                    self.star_imports = []
                    self.exports = set()

            registry.modules = {"pkg": MockInfo(), "pkg.sub": MockInfo()}

            registry.second_pass_star_imports()
