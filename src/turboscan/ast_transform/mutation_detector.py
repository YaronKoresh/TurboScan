import ast


class MutationDetector(ast.NodeVisitor):
    def __init__(self, method_node: ast.FunctionDef) -> None:
        self.is_pure = True
        self.current_method = method_node
        self.self_name = "self"
        if method_node.args.args:
            self.self_name = method_node.args.args[0].arg

    def visit_Assign(self, node) -> None:
        for target in node.targets:
            if isinstance(target, ast.Attribute) and (
                isinstance(target.value, ast.Name)
                and target.value.id == self.self_name
            ):
                self.is_pure = False
        self.generic_visit(node)

    def visit_AugAssign(self, node) -> None:
        if isinstance(node.target, ast.Attribute) and (
            isinstance(node.target.value, ast.Name)
            and node.target.value.id == self.self_name
        ):
            self.is_pure = False
        self.generic_visit(node)

    def visit_Call(self, node) -> None:
        if isinstance(node.func, ast.Attribute) and (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.self_name
        ):
            self.is_pure = False
        self.generic_visit(node)
