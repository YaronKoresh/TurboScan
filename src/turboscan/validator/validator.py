import ast
import builtins
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from turboscan.indexing.types import Scope, SymbolDef
from turboscan.registry.registry import HyperRegistry
from turboscan.resolver.resolver import HyperResolver

PYTHON_BUILTINS = frozenset(dir(builtins))


class HyperValidator(ast.NodeVisitor):
    def __init__(
        self,
        registry: HyperRegistry,
        current_file: Path,
        resolver: HyperResolver,
        check_unused: bool = True,
    ) -> None:
        self.registry = registry
        self.resolver = resolver
        self.current_file = current_file
        self.current_fqn = registry.files_map.get(current_file, "")
        self.current_mod_info = registry.modules.get(self.current_fqn)
        self.errors: List[Tuple[int, str]] = []
        self.warnings: List[Tuple[int, str]] = []
        self.scope_stack: List[Scope] = [Scope()]
        self.imports_map: Dict[
            str, Tuple[Optional[str], int, Optional[str]]
        ] = {}
        self.is_init_file = current_file.name == "__init__.py"
        self.exported_names: Set[str] = set()
        self.check_unused = check_unused
        self.all_used_identifiers: Set[str] = set()

    def report_error(self, lineno: int, msg: str) -> None:
        self.errors.append((lineno, msg))

    def report_warning(self, lineno: int, msg: str) -> None:
        self.warnings.append((lineno, msg))

    def _resolve_abs(self, name: Optional[str], level: int) -> Optional[str]:
        if level == 0:
            return name
        parts = self.current_fqn.split(".") if self.current_fqn else []
        if self.current_mod_info and self.current_mod_info.is_package:
            base_parts = parts
        else:
            base_parts = parts[:-1] if parts else []
        up = level - 1
        if up > len(base_parts):
            return None
        base_parts = base_parts[: len(base_parts) - up]
        base = ".".join(base_parts)
        if name:
            return f"{base}.{name}" if base else name
        return base

    def _lookup_symbol(self, name: str) -> Optional[SymbolDef]:
        if self.current_mod_info and name in self.current_mod_info.symbols:
            return self.current_mod_info.symbols[name]
        if (
            self.current_mod_info
            and name in self.current_mod_info.star_imported_symbols
        ):
            return SymbolDef(name, "star_import", self.current_fqn)
        return None

    def _is_name_defined(self, name: str) -> bool:
        if name in PYTHON_BUILTINS:
            return True
        if name in self.imports_map:
            return True
        for scope in reversed(self.scope_stack):
            if name in scope.defined_names:
                return True
        if self.current_mod_info:
            if name in self.current_mod_info.symbols:
                return True
            if name in self.current_mod_info.star_imported_symbols:
                return True
        return False

    def visit_Name(self, node) -> None:
        if isinstance(node.ctx, ast.Load):
            self.scope_stack[-1].used_names.add(node.id)
            self.all_used_identifiers.add(node.id)
            if not self._is_name_defined(node.id):
                self.report_error(
                    node.lineno, f"Name '{node.id}' is not defined"
                )
        elif isinstance(node.ctx, ast.Store):
            self.scope_stack[-1].defined_names.add(node.id)

    def visit_FunctionDef(self, node) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns:
            self.visit(node.returns)
        for default in node.args.defaults:
            self.visit(default)
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.report_error(
                    node.lineno,
                    f"Mutable default argument in function '{node.name}'",
                )
        for default in node.args.kw_defaults:
            if default:
                self.visit(default)
        all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
        if node.args.vararg:
            all_args.append(node.args.vararg)
        if node.args.kwarg:
            all_args.append(node.args.kwarg)
        for arg in all_args:
            if arg.annotation:
                self.visit(arg.annotation)
        self.scope_stack[-1].defined_names.add(node.name)
        self.scope_stack.append(Scope())
        for arg in all_args:
            self.scope_stack[-1].defined_names.add(arg.arg)
        for item in node.body:
            self.visit(item)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self.scope_stack[-1].defined_names.add(node.name)
        self.scope_stack.append(Scope(is_class=True))
        for item in node.body:
            self.visit(item)
        self.scope_stack.pop()

    def visit_Import(self, node) -> None:
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name.split(".")[0]
            self.imports_map[asname] = (name, 0, None)
            self.scope_stack[-1].defined_names.add(asname)
            if not self.resolver.resolvable(name):
                if self.resolver.is_internal_root(name):
                    self.report_error(
                        node.lineno, f"Unresolvable import '{name}'"
                    )
                else:
                    self.report_warning(
                        node.lineno, f"Module '{name}' not found"
                    )

    def visit_ImportFrom(self, node) -> None:
        mod_name = node.module
        level = node.level
        abs_mod = self._resolve_abs(mod_name, level)
        if abs_mod is None and level > 0:
            self.report_error(
                node.lineno, "Relative import beyond top-level package"
            )
            return
        target_mod = abs_mod or ""
        if target_mod and (not self.resolver.resolvable(target_mod)):
            if self.resolver.is_internal_root(target_mod):
                self.report_error(
                    node.lineno, f"Module '{target_mod}' not found"
                )
            else:
                self.report_warning(
                    node.lineno, f"Module '{target_mod}' not found"
                )
            return
        mod_info = self.registry.modules.get(target_mod)
        for alias in node.names:
            if alias.name == "*":
                continue
            asname = alias.asname or alias.name
            self.imports_map[asname] = (mod_name, level, alias.name)
            self.scope_stack[-1].defined_names.add(asname)
            potential_submod = (
                f"{target_mod}.{alias.name}" if target_mod else alias.name
            )
            if self.resolver.resolvable(potential_submod):
                continue
            if self.resolver.is_external_or_stdlib(target_mod):
                continue
            if mod_info:
                is_defined = alias.name in mod_info.symbols
                is_submodule = alias.name in mod_info.submodules
                is_exported = alias.name in mod_info.exports
                is_star_imported = alias.name in mod_info.star_imported_symbols
                is_dynamic = mod_info.has_getattr
                if not (
                    is_defined
                    or is_submodule
                    or is_exported
                    or is_star_imported
                    or is_dynamic
                ):
                    self.report_error(
                        node.lineno,
                        f"Cannot import '{alias.name}' from '{target_mod}'",
                    )

    def visit_Assign(self, node) -> None:
        for t in node.targets:
            # Use helper to handle all unpacking patterns
            self._extract_names_from_target(t)
            # Special handling for __all__ exports
            if (
                isinstance(t, ast.Name)
                and t.id == "__all__"
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(
                        elt.value, str
                    ):
                        self.exported_names.add(elt.value)
        self.generic_visit(node)

    def visit_For(self, node) -> None:
        # Use helper to handle all unpacking patterns
        self._extract_names_from_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node) -> None:
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                self.scope_stack[-1].defined_names.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node) -> None:
        if node.name:
            self.scope_stack[-1].defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node) -> None:
        for name in node.names:
            self.scope_stack[-1].defined_names.add(name)

    def visit_Nonlocal(self, node) -> None:
        for name in node.names:
            self.scope_stack[-1].defined_names.add(name)

    def _extract_names_from_target(self, target) -> None:
        """Recursively extract all variable names from an assignment/unpacking target.

        Handles:
        - Simple names: x
        - Tuples/Lists: (a, b) or [a, b]
        - Nested unpacking: (a, (b, c))
        - Starred expressions: *rest
        """
        if isinstance(target, ast.Name):
            self.scope_stack[-1].defined_names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._extract_names_from_target(elt)
        elif isinstance(target, ast.Starred):
            self._extract_names_from_target(target.value)
        # Note: We ignore ast.Subscript and ast.Attribute as they don't define new names
        # (e.g., obj.attr = value or list[0] = value)

    def _visit_comprehension(self, node) -> None:
        """Helper to handle all comprehension types (ListComp, SetComp, DictComp, GeneratorExp).

        Comprehensions create their own scope, so iteration variables should not leak out
        but should be available within the comprehension expression.
        """
        # Create a new scope for the comprehension
        self.scope_stack.append(Scope())

        try:
            # Process generators (for clauses)
            # Note: The first generator's iterator uses the outer scope,
            # but subsequent generators can use variables from previous generators
            for comp in node.generators:
                # Visit the iterator expression (uses current scope state)
                self.visit(comp.iter)

                # Then define the target variable(s) in the comprehension scope
                # Use helper to handle all unpacking patterns
                self._extract_names_from_target(comp.target)

                # Visit any filter conditions
                for if_clause in comp.ifs:
                    self.visit(if_clause)

            # Now visit the element expression(s) with target variables in scope
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                # ListComp, SetComp, GeneratorExp all have 'elt'
                self.visit(node.elt)
        finally:
            # Always pop the comprehension scope, even if an exception occurred
            self.scope_stack.pop()
        # Note: We don't call generic_visit because we've explicitly visited all child nodes

    def visit_ListComp(self, node) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node) -> None:
        self._visit_comprehension(node)

    def visit_Lambda(self, node) -> None:
        """Handle lambda expressions which create their own scope for parameters."""
        # Create a new scope for the lambda
        self.scope_stack.append(Scope())

        try:
            # Add all lambda parameters to the scope
            # Handle args, posonlyargs (Python 3.8+), kwonlyargs
            for arg in node.args.args:
                if hasattr(arg, "arg"):
                    self.scope_stack[-1].defined_names.add(arg.arg)
            # posonlyargs may not exist in older Python versions
            if hasattr(node.args, "posonlyargs"):
                for arg in node.args.posonlyargs:
                    if hasattr(arg, "arg"):
                        self.scope_stack[-1].defined_names.add(arg.arg)
            for arg in node.args.kwonlyargs:
                if hasattr(arg, "arg"):
                    self.scope_stack[-1].defined_names.add(arg.arg)
            if node.args.vararg and hasattr(node.args.vararg, "arg"):
                self.scope_stack[-1].defined_names.add(node.args.vararg.arg)
            if node.args.kwarg and hasattr(node.args.kwarg, "arg"):
                self.scope_stack[-1].defined_names.add(node.args.kwarg.arg)

            # Visit the lambda body with parameters in scope
            self.visit(node.body)
        finally:
            # Always pop the lambda scope, even if an exception occurred
            self.scope_stack.pop()

    def get_unused_imports(self) -> List[str]:
        if self.is_init_file:
            return []
        all_used = self.all_used_identifiers
        unused = []
        for name in self.imports_map:
            if name.startswith("_"):
                continue
            if name in all_used:
                continue
            if name in self.exported_names:
                continue
            if self.current_mod_info and self.current_mod_info.has_all:
                if name in self.current_mod_info.exports:
                    continue
            unused.append(name)
        return unused
