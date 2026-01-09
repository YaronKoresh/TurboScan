import ast
from turboscan.indexing.types import ModuleInfo, Signature, SymbolDef

class HyperIndexer(ast.NodeVisitor):
    def __init__(self, mod_info: ModuleInfo):
        self.mod_info = mod_info
    def _get_sig(self, node) -> Signature:
        pos_names = [a.arg for a in node.args.args]
        kwonly_names = [a.arg for a in node.args.kwonlyargs]
        vararg = node.args.vararg.arg if node.args.vararg else None
        kwarg = node.args.kwarg.arg if node.args.kwarg else None
        defaults_len = len(node.args.defaults)
        kw_defaults_len = sum((1 for d in node.args.kw_defaults if d is not None))
        return Signature(pos_names, kwonly_names, vararg, kwarg, defaults_len, kw_defaults_len)
    def visit_FunctionDef(self, node):
        if node.name == '__getattr__':
            self.mod_info.has_getattr = True
        sig = self._get_sig(node)
        self.mod_info.symbols[node.name] = SymbolDef(node.name, 'func', self.mod_info.fqn, sig)
        self.generic_visit(node)
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)
    def visit_ClassDef(self, node):
        self.mod_info.symbols[node.name] = SymbolDef(node.name, 'class', self.mod_info.fqn)
        self.generic_visit(node)
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.mod_info.symbols[target.id] = SymbolDef(target.id, 'var', self.mod_info.fqn)
                if target.id == '__all__':
                    self.mod_info.has_all = True
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                self.mod_info.exports.add(elt.value)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.mod_info.symbols[elt.id] = SymbolDef(elt.id, 'var', self.mod_info.fqn)
        self.generic_visit(node)
    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.mod_info.symbols[node.target.id] = SymbolDef(node.target.id, 'var', self.mod_info.fqn)
        self.generic_visit(node)
    def visit_ImportFrom(self, node):
        mod_name = node.module
        level = node.level
        for alias in node.names:
            if alias.name == '*':
                self.mod_info.star_imports.append((mod_name, level))
            else:
                asname = alias.asname or alias.name
                self.mod_info.imports[asname] = (mod_name, level, alias.name)
                self.mod_info.symbols[asname] = SymbolDef(asname, 'import', self.mod_info.fqn)
    def visit_Import(self, node):
        for alias in node.names:
            asname = alias.asname or alias.name.split('.')[0]
            self.mod_info.imports[asname] = (alias.name, 0, None)
            self.mod_info.symbols[asname] = SymbolDef(asname, 'import', self.mod_info.fqn)
