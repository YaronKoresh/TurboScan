import ast

import pytest

from turboscan.indexing.indexer import HyperIndexer
from turboscan.indexing.types import ModuleInfo


class TestHyperIndexer:
    @pytest.fixture
    def mod_info(self):
        return ModuleInfo("test_mod", "test_mod.py", False)

    def index_code(self, code, info):
        tree = ast.parse(code)
        indexer = HyperIndexer(info)
        indexer.visit(tree)
        return info

    def test_function_signature(self, mod_info) -> None:
        """Should correctly parse positional, keyword, and default args."""
        code = "def my_func(a, b=1, *args, c, d=2, **kwargs): pass"
        self.index_code(code, mod_info)

        sym = mod_info.symbols["my_func"]
        assert sym.kind == "func"
        sig = sym.signature

        assert sig.pos_args == ["a", "b"]
        assert sig.kwonly_args == {"c", "d"}
        assert sig.vararg == "args"
        assert sig.kwarg == "kwargs"
        assert sig.min_pos == 1
        assert sig.required_kwonly == {"c"}

    def test_getattr_detection(self, mod_info) -> None:
        """Should detect dynamic module attributes."""
        code = "def __getattr__(name): pass"
        self.index_code(code, mod_info)
        assert mod_info.has_getattr is True

    def test_class_def(self, mod_info) -> None:
        code = "class MyClass: pass"
        self.index_code(code, mod_info)
        assert "MyClass" in mod_info.symbols
        assert mod_info.symbols["MyClass"].kind == "class"

    def test_variable_assignment(self, mod_info) -> None:
        code = """
x = 1
y, z = 2, 3
        """
        self.index_code(code, mod_info)
        assert "x" in mod_info.symbols
        assert "y" in mod_info.symbols
        assert "z" in mod_info.symbols

    def test_exports_parsing(self, mod_info) -> None:
        """Should parse __all__ list to determine public API."""
        code = "__all__ = ['func', 'var']"
        self.index_code(code, mod_info)
        assert mod_info.has_all is True
        assert mod_info.exports == {"func", "var"}

    def test_imports(self, mod_info) -> None:
        """Should map imports to their sources and levels."""
        code = """
import os
import sys as system
from math import sin
from . import sibling
from ..parent import *
        """
        self.index_code(code, mod_info)

        assert "os" in mod_info.imports

        assert "system" in mod_info.imports
        assert mod_info.imports["system"] == ("sys", 0, None)

        assert "sin" in mod_info.imports
        assert mod_info.imports["sin"] == ("math", 0, "sin")

        assert "sibling" in mod_info.imports
        assert mod_info.imports["sibling"][1] == 1

        assert len(mod_info.star_imports) == 1
        assert mod_info.star_imports[0] == ("parent", 2)


class TestHyperIndexerScopes:
    @pytest.fixture
    def mod_info(self):
        return ModuleInfo("test_mod", "test_mod.py", False)

    def index_code(self, code, info):
        HyperIndexer(info).visit(ast.parse(code))
        return info

    def test_methods_are_not_module_symbols(self, mod_info) -> None:
        code = """
class Box:
    size = 3

    def method(self):
        return 1
"""
        self.index_code(code, mod_info)
        assert "Box" in mod_info.symbols
        assert "method" not in mod_info.symbols
        assert "size" not in mod_info.symbols

    def test_function_locals_are_not_module_symbols(self, mod_info) -> None:
        code = """
import os


def work():
    import json
    from re import match
    local = 1
    typed: int = 2

    def inner():
        pass

    class Local:
        pass

    return local
"""
        self.index_code(code, mod_info)
        assert set(mod_info.symbols) == {"os", "work"}
        assert set(mod_info.imports) == {"os"}

    def test_nested_getattr_is_not_a_module_hook(self, mod_info) -> None:
        code = """
class Proxy:
    def __getattr__(self, name):
        return name
"""
        self.index_code(code, mod_info)
        assert mod_info.has_getattr is False

    def test_global_declaration_defines_a_module_symbol(self, mod_info) -> None:
        code = """
def setup():
    global CONFIG
    CONFIG = {}
"""
        self.index_code(code, mod_info)
        assert mod_info.symbols["CONFIG"].kind == "var"

    def test_definitions_inside_module_level_blocks_are_kept(
        self, mod_info
    ) -> None:
        code = """
try:
    import fast as impl
except ImportError:
    impl = None

if True:
    def helper():
        pass
"""
        self.index_code(code, mod_info)
        assert {"impl", "helper"} <= set(mod_info.symbols)
