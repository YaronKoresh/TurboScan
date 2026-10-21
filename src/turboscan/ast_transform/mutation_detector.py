import ast


class MutationDetector(ast.NodeVisitor):
    def __init__(self, method_node: ast.FunctionDef) -> None:
        self.is_pure = True
        self.current_method = method_node
        self.self_name = "self"
        if method_node.args.args:
            self.self_name = method_node.args.args[0].arg

    def _roots_at_self(self, node) -> bool:
        """True for self.x, self.x.y, self.x[i] and similar chains."""
        if not isinstance(node, (ast.Attribute, ast.Subscript)):
            return False
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return isinstance(node, ast.Name) and node.id == self.self_name

    def _check_target(self, target) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._check_target(elt)
        elif isinstance(target, ast.Starred):
            self._check_target(target.value)
        elif self._roots_at_self(target):
            self.is_pure = False

    def visit_Assign(self, node) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node) -> None:
        if node.value is not None:
            self._check_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_For(self, node) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node) -> None:
        self.visit_For(node)

    def visit_withitem(self, node) -> None:
        if node.optional_vars is not None:
            self._check_target(node.optional_vars)
        self.generic_visit(node)

    def visit_Call(self, node) -> None:
        if isinstance(node.func, ast.Attribute) and (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.self_name
        ):
            self.is_pure = False
        elif isinstance(node.func, ast.Attribute) and self._roots_at_self(
            node.func.value
        ):
            self.is_pure = False
        self.generic_visit(node)
