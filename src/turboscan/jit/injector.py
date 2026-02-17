"""Automatic Numba JIT decorator injection for math-heavy functions."""

import ast
import threading

try:
    from numba import jit, njit, prange, vectorize

    NUMBA_AVAIL = True
except ImportError:
    NUMBA_AVAIL = False


class JITInjector:
    SKIP_JIT_PREFIXES = ("_hyper_", "_batch_")
    JIT_CANDIDATES = {
        "numpy",
        "np",
        "math",
        "cmath",
        "statistics",
        "itertools",
        "functools",
        "operator",
    }
    NUMBA_SAFE_OPS = {
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.LShift,
        ast.RShift,
        ast.BitOr,
        ast.BitXor,
        ast.BitAnd,
        ast.FloorDiv,
        ast.MatMult,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Invert,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    }
    # NumPy functions that don't support scalar inputs in Numba and may cause TypingError
    NUMBA_SCALAR_INCOMPATIBLE_FUNCS = {
        "clip",
        "reshape",
        "transpose",
        "ndenumerate",
        "ndindex",
    }
    _jit_cache = {}
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.jit_count = 0
        self.vectorize_count = 0

    def _is_math_heavy(self, node: ast.FunctionDef) -> bool:
        math_ops = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.BinOp, ast.UnaryOp, ast.Compare)) or (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id in ("np", "numpy", "math")
            ):
                math_ops += 1
        return math_ops > 2

    def _is_numba_compatible(self, node: ast.FunctionDef) -> bool:
        if not NUMBA_AVAIL:
            return False
        INCOMPATIBLE_DECORATORS = {
            "property",
            "contextmanager",
            "asynccontextmanager",
            "wraps",
            "singledispatch",
            "singledispatchmethod",
            "abstractmethod",
            "dataclass",
            "pytest",
            "fixture",
            "mark",
            "parametrize",
        }
        for dec in node.decorator_list:
            dec_name = None
            try:
                if isinstance(dec, ast.Name):
                    dec_name = dec.id
                elif isinstance(dec, ast.Attribute):
                    dec_name = dec.attr
                elif isinstance(dec, ast.Call):
                    if isinstance(dec.func, ast.Name):
                        dec_name = dec.func.id
                    elif isinstance(dec.func, ast.Attribute):
                        dec_name = dec.func.attr
            except Exception:
                pass
            if dec_name and dec_name in INCOMPATIBLE_DECORATORS:
                return False
        if node.args.args and node.args.args[0].arg == "self":
            return False
        arg_names = {a.arg for a in node.args.args}
        for child in ast.walk(node):
            if isinstance(
                child,
                (ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom),
            ):
                return False
            if isinstance(child, ast.ClassDef) and child != node:
                return False
            if isinstance(child, (ast.JoinedStr, ast.FormattedValue)):
                return False
            # Check for NumPy functions that don't support scalar inputs in Numba
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and (
                    isinstance(child.func.value, ast.Name)
                    and child.func.value.id in ("np", "numpy")
                    and child.func.attr in self.NUMBA_SCALAR_INCOMPATIBLE_FUNCS
                )
            ):
                return False
            if isinstance(child, ast.With):
                for item in child.items:
                    ctx = item.context_expr
                    if isinstance(ctx, ast.Call):
                        if isinstance(ctx.func, ast.Attribute):
                            if ctx.func.attr in (
                                "errstate",
                                "catch_warnings",
                                "printoptions",
                            ):
                                return False
                            if isinstance(ctx.func.value, ast.Name):
                                if ctx.func.value.id in (
                                    "np",
                                    "numpy",
                                    "warnings",
                                ):
                                    return False
                        elif isinstance(ctx.func, ast.Name):
                            if ctx.func.id in ("open", "errstate"):
                                return False
            if isinstance(child, ast.Attribute) and isinstance(
                child.value, ast.Name
            ):
                obj_name = child.value.id
                attr_name = child.attr
                if obj_name in (
                    "np",
                    "numpy",
                    "math",
                    "cmath",
                    "random",
                    "torch",
                ):
                    continue
                if obj_name in arg_names:
                    continue
                SAFE_ATTRS = {
                    "shape",
                    "dtype",
                    "ndim",
                    "size",
                    "T",
                    "real",
                    "imag",
                    "astype",
                    "view",
                    "copy",
                    "ravel",
                    "flatten",
                    "item",
                    "min",
                    "max",
                    "sum",
                    "mean",
                    "std",
                    "var",
                    "arg",
                }
                if attr_name in SAFE_ATTRS:
                    continue
                return False
        return True

    def _has_loops(self, node: ast.FunctionDef) -> bool:
        return any(
            isinstance(child, (ast.For, ast.While)) for child in ast.walk(node)
        )

    def _has_loop_conditional_assigns(self, node: ast.FunctionDef) -> bool:

        def scan_loop_body(loop_node):
            cond_vars = set()
            used_vars = set()

            def collect_in_conditional(n, in_cond=False) -> None:
                if isinstance(n, (ast.If, ast.IfExp)):
                    in_cond = True
                if in_cond:
                    if isinstance(n, ast.Assign):
                        for target in n.targets:
                            if isinstance(target, ast.Name):
                                cond_vars.add(target.id)
                            elif isinstance(target, ast.Tuple):
                                for elt in target.elts:
                                    if isinstance(elt, ast.Name):
                                        cond_vars.add(elt.id)
                    elif isinstance(n, ast.AugAssign):
                        if isinstance(n.target, ast.Name):
                            cond_vars.add(n.target.id)
                    elif isinstance(n, ast.AnnAssign):
                        if (
                            isinstance(n.target, ast.Name)
                            and n.value is not None
                        ):
                            cond_vars.add(n.target.id)
                    elif isinstance(n, ast.NamedExpr) and isinstance(
                        n.target, ast.Name
                    ):
                        cond_vars.add(n.target.id)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    used_vars.add(n.id)
                for child in ast.iter_child_nodes(n):
                    collect_in_conditional(child, in_cond)

            for stmt in loop_node.body:
                collect_in_conditional(stmt)
            if hasattr(loop_node, "orelse") and loop_node.orelse:
                for stmt in loop_node.orelse:
                    collect_in_conditional(stmt)
            return cond_vars, used_vars

        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                cond_vars, used_vars = scan_loop_body(child)
                if cond_vars & used_vars:
                    return True
        return False

    def _is_vectorizable(self, node: ast.FunctionDef) -> bool:
        if not NUMBA_AVAIL:
            return False
        for arg in node.args.args:
            if arg.annotation is None:
                return False
            if isinstance(arg.annotation, ast.Name):
                if arg.annotation.id not in ("int", "float", "complex", "bool"):
                    return False
        args = node.args.args
        if len(args) > 20:
            return False
        body_size = sum(1 for _ in ast.walk(node))
        if body_size > 500:
            return False
        has_computation = False
        for child in ast.walk(node):
            if isinstance(child, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                has_computation = True
                break
            if isinstance(child, ast.Call):
                has_computation = True
                break
        if not has_computation:
            return False
        return self._is_numba_compatible(node)

    def inject(self, tree: ast.Module) -> ast.Module:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._process_function(node)
        return tree

    def _process_function(self, node: ast.FunctionDef) -> None:
        if node.name.startswith(self.SKIP_JIT_PREFIXES):
            return
        for dec in node.decorator_list:
            dec_name = None
            if isinstance(dec, ast.Name):
                dec_name = dec.id
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                dec_name = dec.func.id
            if dec_name in (
                "njit",
                "jit",
                "vectorize",
                "guvectorize",
                "cuda_jit",
            ):
                return
        if self._is_numba_compatible(node) and (
            self._has_loops(node) or self._is_math_heavy(node)
        ):
            njit_decorator = ast.Name(id="njit", ctx=ast.Load())
            use_parallel = not self._has_loop_conditional_assigns(node)
            njit_call = ast.Call(
                func=njit_decorator,
                args=[],
                keywords=[
                    ast.keyword(arg="cache", value=ast.Constant(value=True)),
                    ast.keyword(arg="fastmath", value=ast.Constant(value=True)),
                    ast.keyword(
                        arg="parallel", value=ast.Constant(value=use_parallel)
                    ),
                    ast.keyword(
                        arg="error_model", value=ast.Constant(value="python")
                    ),
                ],
            )
            node.decorator_list.append(njit_call)
            self.jit_count += 1
        elif self._is_vectorizable(node):
            vec_decorator = ast.Name(id="vectorize", ctx=ast.Load())
            vec_call = ast.Call(
                func=vec_decorator,
                args=[],
                keywords=[
                    ast.keyword(arg="cache", value=ast.Constant(value=True))
                ],
            )
            node.decorator_list.append(vec_call)
            self.vectorize_count += 1


JIT_INJECTOR = JITInjector()
