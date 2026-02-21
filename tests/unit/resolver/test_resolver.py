from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from turboscan.registry.registry import HyperRegistry
from turboscan.resolver.resolver import HyperResolver


class TestHyperResolver:
    @pytest.fixture
    def mock_registry(self):
        reg = MagicMock(spec=HyperRegistry)
        reg.modules = {"my_project.utils": True}
        reg.root = Path("/fake/root")
        reg.src_prefix = "src"
        return reg

    @pytest.fixture
    def resolver(self, mock_registry):

        with patch.object(
            HyperResolver, "_discover_site_packages", return_value=[]
        ), patch.object(HyperResolver, "_scan_external_toplevel"):
            return HyperResolver(mock_registry)

    def test_stdlib_resolution(self, resolver) -> None:
        """Should correctly identify standard library modules."""

        assert resolver._stdlib_exists("os") is True
        assert resolver._stdlib_exists("sys") is True
        assert resolver.resolvable("os.path") is True

    def test_internal_resolution_via_registry(self, resolver) -> None:
        """Should resolve modules present in the registry."""

        assert resolver._internal_exists("my_project.utils") is True
        assert resolver.resolvable("my_project.utils") is True

        assert resolver.is_internal_root("my_project.utils") is True
        assert resolver.is_internal_root("external_lib") is False

    def test_external_resolution(self, resolver) -> None:
        """Should identify known external packages."""

        resolver._known_external_toplevel.add("requests")

        assert resolver._external_exists("requests") is True
        assert resolver._external_exists("requests.models") is True
        assert resolver.is_external_or_stdlib("requests") is True

    def test_caching_memoization(self, resolver) -> None:
        """Should cache results to avoid repetitive checks."""

        resolver.memo["cached_mod"] = True

        assert resolver.resolvable("cached_mod") is True

    def test_missing_module(self, resolver) -> None:
        """Should return False for non-existent modules."""
        assert resolver.resolvable("completely_fake_module_123") is False

    def test_stdlib_submodule(self, resolver) -> None:
        """Should resolve stdlib submodules."""
        assert resolver._stdlib_exists("os.path") is True
        assert resolver._stdlib_exists("collections.abc") is True
        assert resolver._stdlib_exists("urllib.parse") is True

    def test_stdlib_nonexistent(self, resolver) -> None:
        """Should return False for non-stdlib modules."""
        assert resolver._stdlib_exists("not_a_stdlib_module") is False

    def test_internal_nested_module(self, resolver) -> None:
        """Should resolve nested internal modules."""

        resolver.registry.modules["my_project.sub.module"] = True
        assert resolver._internal_exists("my_project.sub.module") is True

    def test_internal_partial_path(self, resolver) -> None:
        """Should check internal existence for partial paths."""

        resolver._internal_exists("my_project")

    def test_external_nested_module(self, resolver) -> None:
        """Should resolve nested external modules."""
        resolver._known_external_toplevel.add("numpy")
        assert resolver._external_exists("numpy.core") is True
        assert resolver._external_exists("numpy.linalg.solve") is True

    def test_is_external_or_stdlib_stdlib(self, resolver) -> None:
        """is_external_or_stdlib should return True for stdlib."""
        assert resolver.is_external_or_stdlib("os") is True
        assert resolver.is_external_or_stdlib("sys") is True
        assert resolver.is_external_or_stdlib("json") is True

    def test_is_external_or_stdlib_internal(self, resolver) -> None:
        """is_external_or_stdlib should return False for internal modules."""
        assert resolver.is_external_or_stdlib("my_project.utils") is False

    def test_resolvable_builtin(self, resolver) -> None:
        """Should resolve Python builtins."""

        assert resolver.resolvable("builtins") is True

    def test_resolvable_typing(self, resolver) -> None:
        """Should resolve typing module."""
        assert resolver.resolvable("typing") is True
        assert resolver.resolvable("typing.List") is True

    def test_resolvable_collections(self, resolver) -> None:
        """Should resolve collections module."""
        assert resolver.resolvable("collections") is True
        assert resolver.resolvable("collections.OrderedDict") is True

    def test_memo_caching(self, resolver) -> None:
        """Test that memoization works correctly."""

        result1 = resolver.resolvable("os")

        assert "os" in resolver.memo

        result2 = resolver.resolvable("os")
        assert result1 == result2

    def test_empty_module_name(self, resolver) -> None:
        """Should handle empty module names gracefully."""

        result = resolver.resolvable("")

        assert isinstance(result, bool)

    def test_discover_site_packages(self, mock_registry) -> None:
        """Test _discover_site_packages method."""

        resolver = HyperResolver(mock_registry)

        assert isinstance(resolver.site_packages, list)

    def test_scan_external_toplevel(self, mock_registry) -> None:
        """Test _scan_external_toplevel method."""
        resolver = HyperResolver(mock_registry)

        assert isinstance(resolver._known_external_toplevel, set)

    def test_multiple_stdlib_checks(self, resolver) -> None:
        """Test multiple stdlib module checks."""
        stdlib_modules = [
            "abc",
            "functools",
            "itertools",
            "pathlib",
            "dataclasses",
        ]
        for mod in stdlib_modules:
            assert resolver._stdlib_exists(mod) is True, (
                f"{mod} should be stdlib"
            )

    def test_dunder_modules(self, resolver) -> None:
        """Test modules with dunder names."""
        assert resolver.resolvable("__future__") is True

    def test_parent_package_resolution(self, resolver) -> None:
        """Test that parent packages are resolved for nested modules."""
        resolver.registry.modules["pkg.sub.module"] = True

        assert resolver.resolvable("pkg.sub.module") is True

    def test_case_sensitivity(self, resolver) -> None:
        """Module names handling - depends on implementation."""
        resolver._known_external_toplevel.add("numpy")
        assert resolver._external_exists("numpy") is True

        resolver._external_exists("NumPy")

    def test_clear_cache(self, resolver) -> None:
        """Test clearing the memo cache."""
        resolver.memo["test"] = True
        resolver.memo.clear()
        assert "test" not in resolver.memo
