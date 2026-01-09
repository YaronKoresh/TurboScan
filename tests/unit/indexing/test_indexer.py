import pytest
import ast
from turboscan.indexing.types import ModuleInfo
from turboscan.indexing.indexer import HyperIndexer

class TestHyperIndexer:
    @pytest.fixture
    def mod_info(self):
        return ModuleInfo("test_mod", "test_mod.py", False)

    def index_code(self, code, info):
        tree = ast.parse(code)
        indexer = HyperIndexer(info)
        indexer.visit(tree)
        return info

    def test_function_signature(self, mod_info):
        """Should correctly parse positional, keyword, and default args."""
        code = "def my_func(a, b=1, *args, c, d=2, **kwargs): pass"
        self.index_code(code, mod_info)
        
        sym = mod_info.symbols['my_func']
        assert sym.kind == 'func'
        sig = sym.signature
        
        assert sig.pos_args == ['a', 'b']
        assert sig.kwonly_args == {'c', 'd'}
        assert sig.vararg == 'args'
        assert sig.kwarg == 'kwargs'
        assert sig.min_pos == 1  # 'a' is required, 'b' has default
        assert sig.required_kwonly == {'c'} # 'd' has default

    def test_getattr_detection(self, mod_info):
        """Should detect dynamic module attributes."""
        code = "def __getattr__(name): pass"
        self.index_code(code, mod_info)
        assert mod_info.has_getattr is True

    def test_class_def(self, mod_info):
        code = "class MyClass: pass"
        self.index_code(code, mod_info)
        assert 'MyClass' in mod_info.symbols
        assert mod_info.symbols['MyClass'].kind == 'class'

    def test_variable_assignment(self, mod_info):
        code = """
x = 1
y, z = 2, 3
        """
        self.index_code(code, mod_info)
        assert 'x' in mod_info.symbols
        assert 'y' in mod_info.symbols
        assert 'z' in mod_info.symbols

    def test_exports_parsing(self, mod_info):
        """Should parse __all__ list to determine public API."""
        code = "__all__ = ['func', 'var']"
        self.index_code(code, mod_info)
        assert mod_info.has_all is True
        assert mod_info.exports == {'func', 'var'}

    def test_imports(self, mod_info):
        """Should map imports to their sources and levels."""
        code = """
import os
import sys as system
from math import sin
from . import sibling
from ..parent import *
        """
        self.index_code(code, mod_info)
        
        # os -> mapped to os
        assert 'os' in mod_info.imports
        # sys -> mapped to system
        assert 'system' in mod_info.imports
        assert mod_info.imports['system'] == ('sys', 0, None)
        
        # math.sin
        assert 'sin' in mod_info.imports
        assert mod_info.imports['sin'] == ('math', 0, 'sin')
        
        # Relative import
        assert 'sibling' in mod_info.imports
        assert mod_info.imports['sibling'][1] == 1 # level 1
        
        # Star import
        assert len(mod_info.star_imports) == 1
        assert mod_info.star_imports[0] == ('parent', 2) # level 2