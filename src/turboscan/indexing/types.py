from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

class Signature:
    def __init__(self, pos_args, kwonly_args, vararg, kwarg, defaults_count, kw_defaults_count):
        self.pos_args = pos_args
        self.kwonly_args = set(kwonly_args)
        self.vararg = vararg
        self.kwarg = kwarg
        self.min_pos = len(pos_args) - defaults_count
        self.max_pos = len(pos_args)
        self.required_kwonly = set(kwonly_args[:len(kwonly_args) - kw_defaults_count])

class SymbolDef:
    def __init__(self, name, kind, source_fqn, signature=None):
        self.name = name
        self.kind = kind
        self.source_fqn = source_fqn
        self.signature = signature

class ModuleInfo:
    def __init__(self, fqn, file_path, is_package):
        self.fqn = fqn
        self.file_path = file_path
        self.is_package = is_package
        self.symbols: Dict[str, SymbolDef] = {}
        self.submodules: Set[str] = set()
        self.imports: Dict[str, Tuple[str, int, Optional[str]]] = {}
        self.star_imports: List[Tuple[str, int]] = []
        self.star_imported_symbols: Set[str] = set()
        self.exports: Set[str] = set()
        self.has_all = False
        self.has_getattr = False

class Scope:
    def __init__(self, is_class=False):
        self.used_names: Set[str] = set()
        self.defined_names: Set[str] = set()
        self.is_class = is_class
