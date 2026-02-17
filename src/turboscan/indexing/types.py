from typing import Dict, List, Optional, Set, Tuple


class Signature:
    def __init__(
        self,
        pos_args,
        kwonly_args,
        vararg,
        kwarg,
        defaults_count,
        kw_defaults_count,
    ) -> None:
        self.pos_args = pos_args
        self.kwonly_args = set(kwonly_args)
        self.vararg = vararg
        self.kwarg = kwarg
        self.min_pos = len(pos_args) - defaults_count
        self.max_pos = len(pos_args)
        self.required_kwonly = set(
            kwonly_args[: len(kwonly_args) - kw_defaults_count]
        )
        self.__turbo_id__ = (
            f"sig:{len(pos_args)}:{len(self.kwonly_args)}:{vararg}:{kwarg}"
        )


class SymbolDef:
    def __init__(self, name, kind, source_fqn, signature=None) -> None:
        self.name = name
        self.kind = kind
        self.source_fqn = source_fqn
        self.signature = signature
        self.__turbo_id__ = f"sym:{source_fqn}.{name}"


class ModuleInfo:
    def __init__(self, fqn, file_path, is_package) -> None:
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
        self.__turbo_id__ = f"mod:{fqn}"


class Scope:
    def __init__(self, is_class=False) -> None:
        self.used_names: Set[str] = set()
        self.defined_names: Set[str] = set()
        self.is_class = is_class
