import ast

from turboscan.indexing.types import ModuleInfo, Signature, SymbolDef


class HyperIndexer(ast.NodeVisitor):
    def __init__(self, mod_info: ModuleInfo) -> None:
        self.mod_info = mod_info
        self._scope_depth = 0

    def _visit_scope(self, node) -> None:
        self._scope_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._scope_depth -= 1

    def visit_Global(self, node) -> None:
        for name in node.names:
            self.mod_info.symbols.setdefault(
                name, SymbolDef(name, "var", self.mod_info.fqn)
            )

    def _get_sig(self, node) -> Signature:
        pos_names = [a.arg for a in node.args.args]
        kwonly_names = [a.arg for a in node.args.kwonlyargs]
        vararg = node.args.vararg.arg if node.args.vararg else None
        kwarg = node.args.kwarg.arg if node.args.kwarg else None
        defaults_len = len(node.args.defaults)
        kw_defaults_len = sum(1 for d in node.args.kw_defaults if d is not None)
        return Signature(
            pos_names,
            kwonly_names,
            vararg,
            kwarg,
            defaults_len,
            kw_defaults_len,
        )

    def visit_FunctionDef(self, node) -> None:
        if self._scope_depth == 0:
            if node.name == "__getattr__":
                self.mod_info.has_getattr = True
            sig = self._get_sig(node)
            self.mod_info.symbols[node.name] = SymbolDef(
                node.name, "func", self.mod_info.fqn, sig
            )
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node) -> None:
        if self._scope_depth == 0:
            self.mod_info.symbols[node.name] = SymbolDef(
                node.name, "class", self.mod_info.fqn
            )
        self._visit_scope(node)

    def visit_Lambda(self, node) -> None:
        self._visit_scope(node)

    def visit_Assign(self, node) -> None:
        if self._scope_depth > 0:
            self.generic_visit(node)
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.mod_info.symbols[target.id] = SymbolDef(
                    target.id, "var", self.mod_info.fqn
                )
                if target.id == "__all__":
                    self.mod_info.has_all = True
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                self.mod_info.exports.add(elt.value)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.mod_info.symbols[elt.id] = SymbolDef(
                            elt.id, "var", self.mod_info.fqn
                        )
        self.generic_visit(node)

    def visit_AnnAssign(self, node) -> None:
        if self._scope_depth == 0 and isinstance(node.target, ast.Name):
            self.mod_info.symbols[node.target.id] = SymbolDef(
                node.target.id, "var", self.mod_info.fqn
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node) -> None:
        if self._scope_depth > 0:
            return
        mod_name = node.module
        level = node.level
        for alias in node.names:
            if alias.name == "*":
                self.mod_info.star_imports.append((mod_name, level))
            else:
                asname = alias.asname or alias.name
                self.mod_info.imports[asname] = (mod_name, level, alias.name)
                self.mod_info.symbols[asname] = SymbolDef(
                    asname, "import", self.mod_info.fqn
                )

    def visit_Import(self, node) -> None:
        if self._scope_depth > 0:
            return
        for alias in node.names:
            asname = alias.asname or alias.name.split(".")[0]
            self.mod_info.imports[asname] = (alias.name, 0, None)
            self.mod_info.symbols[asname] = SymbolDef(
                asname, "import", self.mod_info.fqn
            )
