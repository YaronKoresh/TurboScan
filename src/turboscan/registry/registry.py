import ast
from pathlib import Path
from typing import Dict, Optional, Set

from turboscan.execution.hyper_boost import HyperBoost
from turboscan.indexing.indexer import HyperIndexer
from turboscan.indexing.types import ModuleInfo
from turboscan.io.file_reader import FAST_READER


class HyperRegistry:
    def __init__(self, root: Path, excludes: Set[str]) -> None:
        self.root = root
        self.excludes = excludes
        self.modules: Dict[str, ModuleInfo] = {}
        self.files_map: Dict[Path, str] = {}
        self.package_dirs: Set[Path] = set()
        self.src_prefix: Optional[str] = None
        self._file_contents: Dict[Path, str] = {}

    def scan(self) -> None:
        for candidate in ["src", "lib", "app"]:
            if (self.root / candidate).is_dir():
                self.src_prefix = candidate
                break
        all_files = []
        for path in self.root.rglob("*.py"):
            if any(excl in path.parts for excl in self.excludes):
                continue
            try:
                path.relative_to(self.root)
                all_files.append(path)
            except ValueError:
                continue
        self._file_contents = FAST_READER.read_files_parallel(all_files)
        for path in all_files:
            try:
                rel_path = path.relative_to(self.root)
            except ValueError:
                continue
            parts = list(rel_path.with_suffix("").parts)
            if parts and self.src_prefix and (parts[0] == self.src_prefix):
                parts = parts[1:]
            if not parts:
                continue
            is_init = parts[-1] == "__init__"
            if is_init:
                parts = parts[:-1]
                is_package = True
            else:
                is_package = False
            fqn = ".".join(parts) if parts else ""
            if fqn in self.modules:
                existing = self.modules[fqn]
                if is_package and (not existing.is_package):
                    existing.file_path = path
                    existing.is_package = True
                    self.files_map[path] = fqn
                continue
            info = ModuleInfo(fqn, path, is_package)
            self.modules[fqn] = info
            self.files_map[path] = fqn
        for fqn, info in self.modules.items():
            if info.is_package:
                self.package_dirs.add(info.file_path.parent)
            if "." in fqn:
                parent = fqn.rsplit(".", 1)[0]
                child_name = fqn.rsplit(".", 1)[1]
                if parent in self.modules:
                    self.modules[parent].submodules.add(child_name)
        for path in self.root.rglob("*"):
            if not path.is_dir():
                continue
            if any(excl in path.parts for excl in self.excludes):
                continue
            has_py = any(
                f.suffix == ".py" for f in path.iterdir() if f.is_file()
            )
            if has_py:
                self.package_dirs.add(path)

    def _index_one(self, info: ModuleInfo) -> bool:
        try:
            content = self._file_contents.get(info.file_path, "")
            if not content:
                content = FAST_READER.read_file(info.file_path)
            tree = ast.parse(content)
            indexer = HyperIndexer(info)
            indexer.visit(tree)
            return True
        except Exception:
            return False

    def first_pass_indexing(self) -> None:
        HyperBoost.run(
            self._index_one,
            list(self.modules.values()),
            quiet=True,
            backend="threads",
        )

    def second_pass_star_imports(self) -> None:
        changed = True
        max_iterations = 10
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for info in self.modules.values():
                for mod_name, level in info.star_imports:
                    abs_mod = self._resolve_abs_for_info(info, mod_name, level)
                    if abs_mod and abs_mod in self.modules:
                        source_mod = self.modules[abs_mod]
                        if source_mod.has_all:
                            symbols_to_add = source_mod.exports
                        else:
                            symbols_to_add = {
                                name
                                for name, sym in source_mod.symbols.items()
                                if not name.startswith("_")
                            }
                        symbols_to_add = (
                            symbols_to_add | source_mod.star_imported_symbols
                        )
                        before = len(info.star_imported_symbols)
                        info.star_imported_symbols.update(symbols_to_add)
                        if len(info.star_imported_symbols) > before:
                            changed = True

    def _resolve_abs_for_info(
        self, info: ModuleInfo, name: Optional[str], level: int
    ) -> Optional[str]:
        if level == 0:
            return name
        parts = info.fqn.split(".") if info.fqn else []
        base_parts = parts if info.is_package else parts[:-1] if parts else []
        up = level - 1
        if up > len(base_parts):
            return None
        base_parts = base_parts[: len(base_parts) - up]
        base = ".".join(base_parts)
        if name:
            return f"{base}.{name}" if base else name
        return base
