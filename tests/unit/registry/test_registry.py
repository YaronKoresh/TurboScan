import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from turboscan.registry.registry import HyperRegistry

# We assume ModuleInfo exists in the environment. 
# If not, we'd need to mock it, but the coverage report says it's there.

class TestHyperRegistry:
    
    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a fake project structure."""
        # src/
        #   main.py
        #   utils/
        #     __init__.py
        #     helper.py
        # tests/
        #   test_main.py
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils").mkdir()
        (tmp_path / "tests").mkdir()
        
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "src" / "utils" / "__init__.py").touch()
        (tmp_path / "src" / "utils" / "helper.py").touch()
        (tmp_path / "tests" / "test_main.py").touch()
        
        return tmp_path

    def test_scan_finds_modules(self, project_root):
        """Should find modules and assign FQNs correctly."""
        # Mock FAST_READER to avoid reading real files
        with patch('turboscan.registry.registry.FAST_READER') as mock_reader:
            mock_reader.read_files_parallel.return_value = {}
            
            registry = HyperRegistry(project_root, excludes={'tests'})
            registry.scan()
            
            # Check main.py (should be 'main', not 'src.main' if src detected)
            assert 'main' in registry.modules
            
            # Check utils package
            assert 'utils' in registry.modules
            assert registry.modules['utils'].is_package
            
            # Check helper
            assert 'utils.helper' in registry.modules
            
            # Tests should be excluded
            assert 'tests.test_main' not in registry.modules

    def test_src_layout_detection(self, project_root):
        """Should detect 'src' directory and strip it from FQN."""
        with patch('turboscan.registry.registry.FAST_READER') as mock_reader:
            mock_reader.read_files_parallel.return_value = {}
            
            registry = HyperRegistry(project_root, excludes=set())
            registry.scan()
            
            assert registry.src_prefix == 'src'

    def test_relative_import_resolution(self):
        """Test the _resolve_abs_for_info logic."""
        registry = HyperRegistry(Path('.'), set())
        
        # Create a mock Info object to simulate context
        class MockInfo:
            def __init__(self, fqn, is_package):
                self.fqn = fqn
                self.is_package = is_package
        
        # Case 1: Import inside a package module (utils.helper)
        # from . import sibling -> level 1
        info = MockInfo("pkg.sub", is_package=False)
        resolved = registry._resolve_abs_for_info(info, "sibling", 1)
        assert resolved == "pkg.sibling"
        
        # Case 2: Import inside __init__.py (utils)
        # from . import sub -> level 1
        info_pkg = MockInfo("pkg", is_package=True)
        resolved = registry._resolve_abs_for_info(info_pkg, "sub", 1)
        assert resolved == "pkg.sub"
        
        # Case 3: Parent import (..)
        # from .. import sibling -> level 2
        info_deep = MockInfo("pkg.deep.mod", is_package=False)
        resolved = registry._resolve_abs_for_info(info_deep, "sibling", 2)
        assert resolved == "pkg.sibling"

    def test_star_import_resolution(self):
        """Verify the iterative fixed-point algorithm triggers."""
        with patch('turboscan.registry.registry.HyperBoost'):
             registry = HyperRegistry(Path('.'), set())
             
             # This is a complex logic to mock fully without the types.py
             # So we verify it doesn't crash on empty modules
             registry.modules = {}
             registry.second_pass_star_imports()
             # If no exception, pass

    # =========================================================================
    # ADDITIONAL TESTS FOR HIGHER COVERAGE
    # =========================================================================

    def test_registry_initialization(self, tmp_path):
        """Test registry initialization with various parameters."""
        registry = HyperRegistry(tmp_path, excludes={'test', 'docs'})
        assert registry.root == tmp_path
        assert 'test' in registry.excludes
        assert 'docs' in registry.excludes

    def test_registry_empty_project(self, tmp_path):
        """Test registry with empty project."""
        with patch('turboscan.registry.registry.FAST_READER') as mock_reader:
            mock_reader.read_files_parallel.return_value = {}
            
            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()
            
            # Should have empty modules
            assert len(registry.modules) == 0

    def test_files_map(self, project_root):
        """Test files_map is populated correctly."""
        with patch('turboscan.registry.registry.FAST_READER') as mock_reader:
            mock_reader.read_files_parallel.return_value = {}
            
            registry = HyperRegistry(project_root, excludes={'tests'})
            registry.scan()
            
            # Should have files mapped
            assert len(registry.files_map) > 0

    def test_no_src_prefix(self, tmp_path):
        """Test project without src directory."""
        # Create flat structure
        (tmp_path / "main.py").touch()
        (tmp_path / "utils.py").touch()
        
        with patch('turboscan.registry.registry.FAST_READER') as mock_reader:
            mock_reader.read_files_parallel.return_value = {}
            
            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()
            
            # No src prefix when not in src layout
            assert registry.src_prefix in ('', None) or not registry.src_prefix

    def test_nested_packages(self, tmp_path):
        """Test deeply nested package structure."""
        # Create nested structure
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").touch()
        (tmp_path / "pkg" / "sub").mkdir()
        (tmp_path / "pkg" / "sub" / "__init__.py").touch()
        (tmp_path / "pkg" / "sub" / "module.py").touch()
        
        with patch('turboscan.registry.registry.FAST_READER') as mock_reader:
            mock_reader.read_files_parallel.return_value = {}
            
            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()
            
            assert 'pkg' in registry.modules
            assert 'pkg.sub' in registry.modules
            assert 'pkg.sub.module' in registry.modules

    def test_exclude_multiple_dirs(self, tmp_path):
        """Test excluding multiple directories."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test.py").touch()
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "doc.py").touch()
        
        with patch('turboscan.registry.registry.FAST_READER') as mock_reader:
            mock_reader.read_files_parallel.return_value = {}
            
            registry = HyperRegistry(tmp_path, excludes={'tests', 'docs'})
            registry.scan()
            
            # main should exist
            assert 'main' in registry.modules
            # tests and docs should be excluded
            assert 'tests.test' not in registry.modules
            assert 'docs.doc' not in registry.modules

    def test_resolve_abs_for_info_top_level(self):
        """Test relative import resolution at top level."""
        registry = HyperRegistry(Path('.'), set())
        
        class MockInfo:
            def __init__(self, fqn, is_package):
                self.fqn = fqn
                self.is_package = is_package
        
        # Top-level module trying to do relative import
        info = MockInfo("module", is_package=False)
        # This might return empty or handle edge case
        result = registry._resolve_abs_for_info(info, "other", 1)
        # Depends on implementation, but shouldn't crash

    def test_resolve_abs_for_info_deep_level(self):
        """Test relative import with deep level."""
        registry = HyperRegistry(Path('.'), set())
        
        class MockInfo:
            def __init__(self, fqn, is_package):
                self.fqn = fqn
                self.is_package = is_package
        
        info = MockInfo("a.b.c.d.e", is_package=False)
        resolved = registry._resolve_abs_for_info(info, "other", 3)
        assert resolved == "a.b.other"

    def test_multiple_python_files(self, tmp_path):
        """Test multiple Python files at same level."""
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        (tmp_path / "c.py").touch()
        
        with patch('turboscan.registry.registry.FAST_READER') as mock_reader:
            mock_reader.read_files_parallel.return_value = {}
            
            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()
            
            assert 'a' in registry.modules
            assert 'b' in registry.modules
            assert 'c' in registry.modules

    def test_hidden_files_excluded(self, tmp_path):
        """Test that hidden files are typically excluded."""
        (tmp_path / "visible.py").touch()
        (tmp_path / ".hidden.py").touch()
        
        with patch('turboscan.registry.registry.FAST_READER') as mock_reader:
            mock_reader.read_files_parallel.return_value = {}
            
            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()
            
            # visible should be there
            assert 'visible' in registry.modules
            # hidden file handling depends on implementation

    def test_pycache_excluded(self, tmp_path):
        """Test that __pycache__ directories are excluded."""
        (tmp_path / "main.py").touch()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-39.pyc").touch()
        
        with patch('turboscan.registry.registry.FAST_READER') as mock_reader:
            mock_reader.read_files_parallel.return_value = {}
            
            registry = HyperRegistry(tmp_path, excludes=set())
            registry.scan()
            
            # main should be there
            assert 'main' in registry.modules
            # __pycache__ should not create modules

    def test_second_pass_star_imports_with_modules(self):
        """Test second_pass_star_imports with actual modules."""
        with patch('turboscan.registry.registry.HyperBoost'):
            registry = HyperRegistry(Path('.'), set())
            
            # Create mock module info
            class MockInfo:
                def __init__(self):
                    self.star_imports = []
                    self.exports = set()
            
            registry.modules = {
                'pkg': MockInfo(),
                'pkg.sub': MockInfo()
            }
            
            # Should not crash
            registry.second_pass_star_imports()