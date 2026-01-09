import sys
import sysconfig
from pathlib import Path
from typing import Dict, List, Optional, Set
from turboscan.registry.registry import HyperRegistry

class HyperResolver:
    def __init__(self, registry: HyperRegistry):
        self.registry = registry
        self.memo: Dict[str, bool] = {}
        self.builtins = set(sys.builtin_module_names) | {'builtins', '_thread', '_frozen_importlib'}
        self.stdlib = set(getattr(sys, 'stdlib_module_names', set()))
        stdlib_path = sysconfig.get_paths().get('stdlib', '')
        self.stdlib_path = Path(stdlib_path) if stdlib_path else None
        self.site_packages = self._discover_site_packages()
        self._known_external_toplevel: Set[str] = set()
        self._scan_external_toplevel()
    def _discover_site_packages(self) -> List[Path]:
        paths = []
        seen = set()
        for p in sys.path:
            if not p:
                continue
            pp = Path(p)
            if pp in seen:
                continue
            seen.add(pp)
            if 'site-packages' in pp.parts or 'dist-packages' in pp.parts:
                if pp.exists():
                    paths.append(pp)
        return paths
    def _scan_external_toplevel(self):
        for sp in self.site_packages:
            if not sp.exists():
                continue
            try:
                for item in sp.iterdir():
                    name = item.name
                    if item.is_dir():
                        if name.endswith('.dist-info') or name.endswith('.egg-info'):
                            pkg_name = name.rsplit('-', 1)[0].replace('-', '_')
                            self._known_external_toplevel.add(pkg_name.lower())
                        elif not name.startswith('_'):
                            self._known_external_toplevel.add(name.lower())
                    elif item.is_file():
                        if name.endswith('.py'):
                            self._known_external_toplevel.add(name[:-3].lower())
                        elif '.cpython' in name or name.endswith('.pyd') or name.endswith('.so'):
                            base = name.split('.')[0]
                            self._known_external_toplevel.add(base.lower())
            except PermissionError:
                continue
    def _internal_exists(self, fqn: str) -> bool:
        if fqn in self.registry.modules:
            return True
        parts = fqn.split('.')
        if self.registry.src_prefix:
            mod_path = self.registry.root / self.registry.src_prefix / Path(*parts)
        else:
            mod_path = self.registry.root / Path(*parts)
        if mod_path.with_suffix('.py').exists():
            return True
        if mod_path.exists() and mod_path.is_dir():
            return True
        return False
    def _stdlib_exists(self, fqn: str) -> bool:
        top = fqn.split('.')[0]
        if top in self.stdlib or top in self.builtins:
            return True
        if self.stdlib_path and self.stdlib_path.exists():
            parts = fqn.split('.')
            mod_file = self.stdlib_path / Path(*parts)
            if mod_file.with_suffix('.py').exists():
                return True
            if mod_file.exists() and mod_file.is_dir():
                return True
        return False
    def _external_exists(self, fqn: str) -> bool:
        top = fqn.split('.')[0].lower()
        if top in self._known_external_toplevel:
            return True
        for base in self.site_packages:
            parts = fqn.split('.')
            mod_path = base / Path(*parts)
            if mod_path.with_suffix('.py').exists():
                return True
            if mod_path.exists() and mod_path.is_dir():
                return True
        return False
    def resolvable(self, fqn: str) -> bool:
        if not fqn:
            return True
        if fqn in self.memo:
            return self.memo[fqn]
        ok = self._internal_exists(fqn) or self._stdlib_exists(fqn) or self._external_exists(fqn)
        self.memo[fqn] = ok
        return ok
    def is_internal_root(self, fqn: str) -> bool:
        root = fqn.split('.')[0]
        for m in self.registry.modules:
            if m == root or m.startswith(root + '.'):
                return True
        return False
    def is_external_or_stdlib(self, fqn: str) -> bool:
        top = fqn.split('.')[0]
        return self._stdlib_exists(top) or self._external_exists(top)
