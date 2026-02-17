"""
HyperAutoParallelizer

Key fixes:
1. _get_loop_local_vars now recursively finds ALL assignments (nested in if/try/with/etc)
2. Added _get_all_assigned_names helper for comprehensive assignment detection
3. Added _has_problematic_loop_patterns to detect unsafe parallelization patterns
4. Fixed handling of augmented assignments and named expressions (walrus operator)
5. Better detection of variables that "escape" the loop
6. Removes @lru_cache decorators to enable multiprocessing serialization
7. CRITICAL FIX: Properly detects multi-dimensional array subscripts (array[:, i]) to avoid shape errors
"""

import ast
import builtins
from typing import Dict, List, Optional, Set, Tuple

try:
    import cloudpickle

    CLOUDPICKLE_AVAIL = True
except ImportError:
    CLOUDPICKLE_AVAIL = False
try:
    import numpy as np

    NUMPY_AVAIL = True
except ImportError:
    NUMPY_AVAIL = False
    np = None


# ============================================================================
# LRU CACHE REMOVAL - Critical fix for multiprocessing serialization
# ============================================================================


class LRUCacheRemover(ast.NodeTransformer):
    """
    AST transformer that removes @lru_cache and @functools.lru_cache decorators.

    This is CRITICAL for multiprocessing because:
    - @lru_cache creates wrapper objects containing threading.RLock
    - RLock objects cannot be pickled
    - cloudpickle serializes class definitions, not just instances
    - Classes with @lru_cache methods fail to serialize

    By removing the decorators at AST level (before code execution),
    the resulting classes are fully picklable.
    """

    LRU_CACHE_NAMES = {"lru_cache", "cache"}
    LRU_CACHE_ATTRS = {
        ("functools", "lru_cache"),
        ("functools", "cache"),
    }

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.removed_count = 0
        self.removed_locations: List[
            Tuple[str, int, str]
        ] = []  # (class, line, method)
        self._current_class: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Track current class name for reporting."""
        old_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old_class
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Remove @lru_cache from regular functions."""
        return self._process_function(node)

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        """Remove @lru_cache from async functions."""
        return self._process_function(node)

    def _process_function(self, node):
        """Process a function node and remove lru_cache decorators."""
        if not node.decorator_list:
            return node

        new_decorators = []
        for decorator in node.decorator_list:
            if self._is_lru_cache_decorator(decorator):
                self.removed_count += 1
                class_name = self._current_class or "<module>"
                self.removed_locations.append(
                    (class_name, node.lineno, node.name)
                )

                if self.verbose:
                    print(
                        f"  [LRU Fix] Removed @lru_cache from {class_name}.{node.name} (line {node.lineno})"
                    )
            else:
                new_decorators.append(decorator)

        node.decorator_list = new_decorators
        return node

    def _is_lru_cache_decorator(self, decorator: ast.expr) -> bool:
        """Check if a decorator is @lru_cache or @functools.lru_cache."""
        # Case 1: @lru_cache or @cache (bare name)
        if isinstance(decorator, ast.Name):
            return decorator.id in self.LRU_CACHE_NAMES

        # Case 2: @lru_cache(maxsize=N) or @cache() (call)
        if isinstance(decorator, ast.Call):
            return self._is_lru_cache_decorator(decorator.func)

        # Case 3: @functools.lru_cache or @functools.cache (attribute)
        if isinstance(decorator, ast.Attribute) and isinstance(
            decorator.value, ast.Name
        ):
            return (decorator.value.id, decorator.attr) in self.LRU_CACHE_ATTRS

        return False

    def get_report(self) -> str:
        """Get a human-readable report of removed decorators."""
        if not self.removed_locations:
            return ""

        lines = [f"Removed {self.removed_count} @lru_cache decorators:"]
        for class_name, line, method_name in self.removed_locations:
            lines.append(f"  - {class_name}.{method_name} (line {line})")
        return "\n".join(lines)


def remove_lru_cache_from_ast(
    tree: ast.Module, verbose: bool = False
) -> Tuple[ast.Module, int]:
    """
    Remove @lru_cache decorators from an AST.

    Args:
        tree: The AST to transform
        verbose: Print info about removed decorators

    Returns:
        Tuple of (transformed_tree, count_removed)
    """
    transformer = LRUCacheRemover(verbose=verbose)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    return tree, transformer.removed_count


# ============================================================================
# MAIN PARALLELIZER CLASS
# ============================================================================


class HyperAutoParallelizer(ast.NodeTransformer):
    def __init__(
        self, remove_lru_cache: bool = True, verbose: bool = False
    ) -> None:
        """
        Initialize the parallelizer.

        Args:
            remove_lru_cache: If True, remove @lru_cache decorators for multiprocessing
            verbose: Print debug information
        """
        self.loop_counter = 0
        self.task_counter = 0
        self.process_safe_count = 0
        self.vectorized_count = 0
        self.scope_stack = []
        self.STRICT_UNSAFE_MODULES = {
            "socket",
            "subprocess",
            "threading",
            "multiprocessing",
            "requests",
            "urllib",
            "http",
            "smtplib",
            "ftplib",
        }
        self.UNSAFE_CALLS = {
            "os.remove",
            "os.unlink",
            "os.rmdir",
            "os.mkdir",
            "os.makedirs",
            "os.rename",
            "os.replace",
            "os.symlink",
            "os.chmod",
            "os.chown",
            "shutil.rmtree",
            "shutil.move",
            "shutil.copy",
            "shutil.copy2",
            "os.chdir",
            "os.chroot",
            "os.putenv",
            "sys.exit",
            "input",
            "open",
            "random.seed",
            "numpy.random.seed",
        }
        self.UNSAFE_METHOD_VERBS = {
            "write",
            "writelines",
            "dump",
            "save",
            "send",
            "sendall",
            "recv",
            "bind",
            "connect",
            "listen",
            "accept",
            "close",
        }

        # LRU cache removal settings
        self.remove_lru_cache = remove_lru_cache
        self.verbose = verbose
        self.lru_removed_count = 0

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """
        Visit the module - this is where we remove @lru_cache FIRST,
        before any other transformations.
        """
        # CRITICAL: Remove @lru_cache decorators FIRST
        if self.remove_lru_cache:
            lru_remover = LRUCacheRemover(verbose=self.verbose)
            node = lru_remover.visit(node)
            self.lru_removed_count = lru_remover.removed_count

            if self.lru_removed_count > 0 and self.verbose:
                print(
                    f"  [LRU Fix] Removed {self.lru_removed_count} @lru_cache decorators for multiprocessing"
                )

        # Then continue with normal parallelization
        self.generic_visit(node)
        return node

    def _are_nodes_equal(self, node1: ast.AST, node2: ast.AST) -> bool:
        if node1 is node2:
            return True
        return ast.dump(node1) == ast.dump(node2)

    def _get_rw_sets(self, node: ast.AST) -> Tuple[Set[str], Set[str]]:
        reads = set()
        writes = set()
        is_transparent_call = False
        if isinstance(node, (ast.Expr, ast.Call)):
            call_node = node.value if isinstance(node, ast.Expr) else node
            if isinstance(call_node, ast.Call):
                func_name = ""
                if isinstance(call_node.func, ast.Name):
                    func_name = call_node.func.id
                elif isinstance(call_node.func, ast.Attribute):
                    func_name = call_node.func.attr
                if func_name in (
                    "print",
                    "info",
                    "debug",
                    "warning",
                    "error",
                    "log",
                    "write",
                    "tqdm",
                ):
                    is_transparent_call = True
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Store):
                    writes.add(child.id)
                elif isinstance(child.ctx, ast.Load):
                    reads.add(child.id)
            elif isinstance(child, ast.arg):
                writes.add(child.arg)
        if is_transparent_call:
            writes.clear()
        return (reads, writes)

    def _resolve_name(self, node) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._resolve_name(node.value)
            if value:
                return f"{value}.{node.attr}"
        return None

    def _analyze_local_scope(
        self, node: ast.FunctionDef
    ) -> Tuple[Set[str], Set[str]]:
        locals_created = set()
        arguments = set()
        args = node.args
        all_args = args.args + args.posonlyargs + args.kwonlyargs
        if args.vararg:
            all_args.append(args.vararg)
        if args.kwarg:
            all_args.append(args.kwarg)
        for arg in all_args:
            arguments.add(arg.arg)
        for child in ast.walk(node):
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = (
                    child.targets
                    if isinstance(child, ast.Assign)
                    else [child.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        locals_created.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                locals_created.add(elt.id)
        return (locals_created, arguments)

    def _is_safe_block(self, nodes: List[ast.AST]) -> bool:
        if not self.scope_stack:
            return False
        locals_created, _arguments = self.scope_stack[-1]
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, (ast.Global, ast.Nonlocal)):
                    return False
                if isinstance(child, ast.Call):
                    full_name = self._resolve_name(child.func)
                    if full_name:
                        root_module = full_name.split(".")[0]
                        if root_module in self.STRICT_UNSAFE_MODULES:
                            return False
                    if full_name in self.UNSAFE_CALLS:
                        return False
                    if isinstance(child.func, ast.Attribute):
                        method_name = child.func.attr
                        if method_name in self.UNSAFE_METHOD_VERBS:
                            return False
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name):
                                obj_name = target.value.id
                                if obj_name not in locals_created:
                                    return False
        return True

    def _is_vectorizable_loop(self, node: ast.For) -> bool:
        if not NUMPY_AVAIL:
            return False
        if not isinstance(node.iter, ast.Call):
            return False
        if not isinstance(node.iter.func, ast.Name):
            return False
        if node.iter.func.id != "range":
            return False
        return all(
            not isinstance(child, (ast.Call, ast.If, ast.For, ast.While))
            for child in ast.walk(node)
        )

    def _create_vectorized_loop(self, node: ast.For) -> List[ast.AST]:
        self.vectorized_count += 1
        return [node]

    # ========== CRITICAL FIX: New helper methods for proper variable detection ==========

    def _extract_names_from_target(self, target: ast.AST) -> Set[str]:
        """Extract all variable names from an assignment target (handles unpacking)."""
        names = set()
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                names.update(self._extract_names_from_target(elt))
        elif isinstance(target, ast.Starred):
            names.update(self._extract_names_from_target(target.value))
        # Subscript and Attribute targets don't define new names
        return names

    def _get_all_assigned_names(
        self, nodes: List[ast.AST], include_aug: bool = True
    ) -> Set[str]:
        """
        Recursively find ALL variable names assigned anywhere in the given AST nodes.
        Args:
            nodes: List of AST nodes to scan
            include_aug: If True, include targets of AugAssign (+=).
                         If False, ignore them (useful for detecting purely local vars).
        """
        assigned = set()

        for node in nodes:
            for child in ast.walk(node):
                # Regular assignment: x = value
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        assigned.update(self._extract_names_from_target(target))

                # Annotated assignment: x: int = value
                elif isinstance(child, ast.AnnAssign):
                    if child.target is not None:
                        assigned.update(
                            self._extract_names_from_target(child.target)
                        )

                # Augmented assignment: x += value
                elif isinstance(child, ast.AugAssign):
                    if include_aug and isinstance(child.target, ast.Name):
                        assigned.add(child.target.id)

                # Named expression: (x := value)
                elif isinstance(child, ast.NamedExpr):
                    if isinstance(child.target, ast.Name):
                        assigned.add(child.target.id)

                # For loop targets (nested loops)
                elif isinstance(child, ast.For):
                    assigned.update(
                        self._extract_names_from_target(child.target)
                    )

                # With statement targets
                elif isinstance(child, ast.With):
                    for item in child.items:
                        if item.optional_vars:
                            assigned.update(
                                self._extract_names_from_target(
                                    item.optional_vars
                                )
                            )

                # Exception handlers
                elif isinstance(child, ast.ExceptHandler):
                    if child.name:
                        assigned.add(child.name)

                # Comprehensions
                elif isinstance(child, ast.comprehension):
                    assigned.update(
                        self._extract_names_from_target(child.target)
                    )

        return assigned

    def _get_loop_local_vars(self, node: ast.For) -> Set[str]:
        """
        Get variables that are initialized/defined INSIDE this loop.
        We explicitly exclude AugAssign (+=) because those imply the variable
        already existed (likely in outer scope), making them candidates for accumulation.
        """
        local_vars = set()

        # Add loop iteration variable(s)
        local_vars.update(self._extract_names_from_target(node.target))

        # Add variables explicitly assigned (=), but NOT just augmented (+=)
        # passing include_aug=False is the key fix here
        local_vars.update(
            self._get_all_assigned_names(node.body, include_aug=False)
        )

        return local_vars

    def _get_all_read_names(self, nodes: List[ast.AST]) -> Set[str]:
        """
        Recursively find ALL variable names read (loaded) anywhere in the given AST nodes.
        """
        reads = set()

        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(
                    child.ctx, ast.Load
                ):
                    reads.add(child.id)

        return reads

    def _has_problematic_loop_patterns(
        self,
        node: ast.For,
        outer_scope_vars: Set[str],
        allowed_accumulators: Optional[Set[str]] = None,
    ) -> bool:
        """
        Detect loop patterns that would break when parallelized:
        1. Variables assigned in loop that are used AFTER the loop in outer scope
        2. Variables assigned conditionally that are expected to persist
        3. Loop-carried dependencies (iteration N depends on iteration N-1)

        Returns True if the loop should NOT be parallelized.
        """
        allowed_accumulators = allowed_accumulators or set()

        # Get all variables assigned anywhere in the loop body
        loop_assigns = self._get_all_assigned_names(node.body)

        # Get the loop iteration variable(s)
        loop_targets = self._extract_names_from_target(node.target)

        # Variables that are assigned in the loop but are NOT the loop variable
        # These could potentially be "result" variables
        potential_results = loop_assigns - loop_targets

        # Check for assignments to variables that might exist in outer scope
        # and are being modified by the loop (not just used locally)
        for var in potential_results:
            # If this variable is also READ before being assigned in the same iteration,
            # it might be a loop-carried dependency.
            if var in outer_scope_vars:
                # FIX: If it's a known accumulation variable, it's allowed!
                if var in allowed_accumulators:
                    continue

                # Variable from outer scope is being assigned in loop
                # This is problematic because parallel workers won't share state
                return True

        # Check for break/early-return patterns that depend on finding something
        # These often indicate loops that shouldn't be parallelized
        return any(isinstance(child, ast.Break) for child in ast.walk(node))

    # ========== END CRITICAL FIX ==========

    def _create_loop_worker(
        self,
        node: ast.For,
        backend_choice: str,
        outer_reads: Optional[Set[str]] = None,
        accumulations: Optional[Dict[str, ast.AST]] = None,
    ) -> List[ast.AST]:
        self.loop_counter += 1
        worker_name = f"_hyper_loop_worker_{self.loop_counter}"
        outer_reads = outer_reads or set()
        accumulations = accumulations or {}
        uses_complex_objects = self._contains_unpicklable(node.body)

        if isinstance(node.target, ast.Name):
            loop_var_names = {node.target.id}
            target_id = node.target.id
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            loop_var_names = {
                elt.id for elt in node.target.elts if isinstance(elt, ast.Name)
            }
            target_id = "item"
        else:
            target_id = "item"
            loop_var_names = set()

        # CRITICAL FIX: Filter out variables that are assigned inside the loop
        # They should NOT be passed from outer scope
        loop_body_assigns = self._get_all_assigned_names(node.body)

        # Remove loop-assigned variables from outer_reads
        outer_reads = outer_reads - loop_body_assigns

        filtered_outer_reads = (
            outer_reads
            - loop_var_names
            - {
                "np",
                "numpy",
                "torch",
                "self",
                "cls",
                "HyperBoost",
                "Boost",
                "True",
                "False",
                "None",
                "print",
                "len",
                "range",
                "int",
                "float",
                "str",
                "list",
                "dict",
                "set",
                "tuple",
                "bool",
                "type",
                "sum",
                "min",
                "max",
                "abs",
                "enumerate",
                "zip",
                "map",
                "filter",
                "any",
                "all",
                "sorted",
                "reversed",
                "isinstance",
                "hasattr",
                "getattr",
                "setattr",
                "open",
                "super",
                "object",
                "Exception",
                "ValueError",
                "TypeError",
                "KeyError",
                "IndexError",
                "RuntimeError",
                "StopIteration",
            }
        )

        can_use_processes = False
        target_list_name = None
        assignment_node = None
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Assign)
            and (not accumulations)
        ):
            assign = node.body[0]
            if len(assign.targets) == 1 and isinstance(
                assign.targets[0], ast.Subscript
            ):
                sub = assign.targets[0]
                if isinstance(sub.value, ast.Name):
                    # CRITICAL FIX: Check if this is a simple 1D subscript
                    # Multi-dimensional subscripts (like array[:, i]) are NOT safe
                    slice_node = sub.slice
                    is_simple_subscript = (
                        isinstance(
                            slice_node, (ast.Name, ast.Constant)
                        )  # array[0] (unlikely but safe)
                    )
                    if is_simple_subscript:
                        target_list_name = sub.value.id
                        can_use_processes = not uses_complex_objects
                        assignment_node = assign

        worker_body = []
        return_values = []
        if accumulations:
            for var_name, aug_node in accumulations.items():
                return_values.append(aug_node.value)
            for stmt in node.body:
                if not isinstance(stmt, ast.AugAssign) or (
                    isinstance(stmt, ast.AugAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id not in accumulations
                ):
                    worker_body.append(stmt)
            if len(return_values) == 1:
                worker_body.append(ast.Return(value=return_values[0]))
            else:
                worker_body.append(
                    ast.Return(
                        value=ast.Tuple(elts=return_values, ctx=ast.Load())
                    )
                )
            effective_backend = "threads" if uses_complex_objects else "auto"
        elif can_use_processes and assignment_node:
            value_expression = assignment_node.value
            worker_body = [ast.Return(value=value_expression)]
            effective_backend = "auto"
            self.process_safe_count += 1
        else:
            worker_body = list(node.body)
            effective_backend = (
                "threads" if uses_complex_objects else backend_choice
            )

        if not worker_body:
            worker_body = [ast.Pass()]

        worker_args = [ast.arg(arg=target_id)]
        for outer_var in sorted(filtered_outer_reads):
            worker_args.append(ast.arg(arg=outer_var))

        worker_func = ast.FunctionDef(
            name=worker_name,
            args=ast.arguments(
                posonlyargs=[],
                args=worker_args,
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=worker_body,
            decorator_list=[],
        )

        if isinstance(node.target, (ast.Tuple, ast.List)):
            unpack = ast.Assign(
                targets=[node.target], value=ast.Name(id="item", ctx=ast.Load())
            )
            worker_func.body.insert(0, unpack)

        if filtered_outer_reads:
            wrapper_name = f"_hyper_wrapper_{self.loop_counter}"
            wrapper_call = ast.Call(
                func=ast.Name(id=worker_name, ctx=ast.Load()),
                args=[ast.Name(id=target_id, ctx=ast.Load())]
                + [
                    ast.Name(id=v, ctx=ast.Load())
                    for v in sorted(filtered_outer_reads)
                ],
                keywords=[],
            )
            wrapper_func = ast.FunctionDef(
                name=wrapper_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=target_id)],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[ast.Return(value=wrapper_call)],
                decorator_list=[],
            )
            if isinstance(node.target, (ast.Tuple, ast.List)):
                unpack = ast.Assign(
                    targets=[node.target],
                    value=ast.Name(id="item", ctx=ast.Load()),
                )
                wrapper_func.body.insert(0, unpack)
            run_func_name = wrapper_name
            extra_funcs = [worker_func, wrapper_func]
        else:
            run_func_name = worker_name
            extra_funcs = [worker_func]

        run_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="HyperBoost", ctx=ast.Load()),
                attr="run",
                ctx=ast.Load(),
            ),
            args=[ast.Name(id=run_func_name, ctx=ast.Load()), node.iter],
            keywords=[
                ast.keyword(arg="quiet", value=ast.Constant(value=True)),
                ast.keyword(
                    arg="backend", value=ast.Constant(value=effective_backend)
                ),
            ],
        )

        result_nodes = list(extra_funcs)
        if accumulations:
            results_var = f"_hyper_deltas_{self.loop_counter}"
            results_assign = ast.Assign(
                targets=[ast.Name(id=results_var, ctx=ast.Store())],
                value=run_call,
            )
            result_nodes.append(results_assign)
            for var_name, aug_node in accumulations.items():
                delta_var = f"_d_{self.loop_counter}"
                agg_loop = ast.For(
                    target=ast.Name(id=delta_var, ctx=ast.Store()),
                    iter=ast.Name(id=results_var, ctx=ast.Load()),
                    body=[
                        ast.AugAssign(
                            target=ast.Name(id=var_name, ctx=ast.Store()),
                            op=aug_node.op,
                            value=ast.Name(id=delta_var, ctx=ast.Load()),
                        )
                    ],
                    orelse=[],
                )
                result_nodes.append(agg_loop)
        elif can_use_processes and target_list_name:
            boost_action = ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id=target_list_name, ctx=ast.Load()),
                        slice=ast.Slice(lower=None, upper=None, step=None),
                        ctx=ast.Store(),
                    )
                ],
                value=run_call,
            )
            result_nodes.append(boost_action)
        else:
            boost_action = ast.Expr(value=run_call)
            result_nodes.append(boost_action)

        for n in result_nodes:
            ast.fix_missing_locations(n)
        return result_nodes

    def _get_free_variables(self, statements: List[ast.AST]) -> Set[str]:
        all_reads = set()
        all_writes = set()
        for stmt in statements:
            reads, writes = self._get_rw_sets(stmt)
            all_reads.update(reads - all_writes)
            all_writes.update(writes)
        builtins_names = set(dir(builtins))
        common_globals = {
            "True",
            "False",
            "None",
            "print",
            "len",
            "range",
            "int",
            "float",
            "str",
            "list",
            "dict",
            "set",
            "tuple",
            "bool",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "min",
            "max",
            "sum",
            "abs",
            "round",
            "sorted",
            "enumerate",
            "zip",
            "map",
            "filter",
            "any",
            "all",
            "open",
            "super",
            "HyperBoost",
            "Boost",
            "np",
            "numpy",
            "torch",
            "librosa",
        }
        return all_reads - builtins_names - common_globals

    def _get_conditionally_defined_vars(
        self, body_list: List[ast.AST], current_idx: int
    ) -> Set[str]:
        conditional_vars = set()
        for i in range(current_idx):
            node = body_list[i]
            if isinstance(node, (ast.If, ast.Try, ast.With)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                conditional_vars.add(target.id)
                            elif isinstance(target, ast.Tuple):
                                for elt in target.elts:
                                    if isinstance(elt, ast.Name):
                                        conditional_vars.add(elt.id)
                    elif isinstance(child, ast.AnnAssign) and isinstance(
                        child.target, ast.Name
                    ):
                        conditional_vars.add(child.target.id)
        return conditional_vars

    def _has_conditional_dependency(
        self,
        statements: List[ast.AST],
        body_list: List[ast.AST],
        start_idx: int,
    ) -> bool:
        conditional_vars = self._get_conditionally_defined_vars(
            body_list, start_idx
        )
        if not conditional_vars:
            return False
        for stmt in statements:
            reads, _ = self._get_rw_sets(stmt)
            if reads & conditional_vars:
                return True
        return False

    def _contains_unpicklable(self, statements: List[ast.AST]) -> bool:
        if CLOUDPICKLE_AVAIL:
            return False
        for stmt in statements:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call) and isinstance(
                    child.func, ast.Attribute
                ):
                    if isinstance(child.func.value, ast.Name):
                        obj_name = child.func.value.id
                        if obj_name == "self":
                            return True
                    elif isinstance(child.func.value, ast.Attribute):
                        return True
                if isinstance(child, ast.Lambda):
                    return True
        return False

    def _create_task_dispatcher(
        self, statements: List[ast.AST]
    ) -> List[ast.AST]:
        self.task_counter += 1
        worker_name = f"_hyper_task_worker_{self.task_counter}"
        result_name = f"_res_{self.task_counter}"
        use_threads = self._contains_unpicklable(statements)
        backend = "threads" if use_threads else "auto"
        task_funcs = []
        targets = []
        for idx, stmt in enumerate(statements):
            if isinstance(stmt, ast.Assign):
                action = stmt.value
                targets.append((idx, stmt.targets))
            elif isinstance(stmt, ast.Expr):
                action = stmt.value
                targets.append((idx, None))
            else:
                continue
            task_lambda = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=action,
            )
            task_funcs.append(task_lambda)
        task_list_name = f"_tasks_{self.task_counter}"
        task_list_assign = ast.Assign(
            targets=[ast.Name(id=task_list_name, ctx=ast.Store())],
            value=ast.List(elts=task_funcs, ctx=ast.Load()),
        )
        worker_func = ast.FunctionDef(
            name=worker_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="fn")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[
                ast.Return(
                    value=ast.Call(
                        func=ast.Name(id="fn", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    )
                )
            ],
            decorator_list=[],
        )
        run_call = ast.Assign(
            targets=[ast.Name(id=result_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="HyperBoost", ctx=ast.Load()),
                    attr="run",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Name(id=worker_name, ctx=ast.Load()),
                    ast.Name(id=task_list_name, ctx=ast.Load()),
                ],
                keywords=[
                    ast.keyword(arg="quiet", value=ast.Constant(value=True)),
                    ast.keyword(
                        arg="backend", value=ast.Constant(value=backend)
                    ),
                ],
            ),
        )
        unpack_stmts = []
        for idx, target_vars in targets:
            if target_vars:
                assign = ast.Assign(
                    targets=target_vars,
                    value=ast.Subscript(
                        value=ast.Name(id=result_name, ctx=ast.Load()),
                        slice=ast.Constant(value=idx),
                        ctx=ast.Load(),
                    ),
                )
                unpack_stmts.append(assign)
        result_nodes = [task_list_assign, worker_func, run_call, *unpack_stmts]
        for n in result_nodes:
            ast.fix_missing_locations(n)
        return result_nodes

    def _is_candidate_for_task(self, node: ast.AST) -> bool:
        # Side-effect detection for standalone expressions
        if isinstance(node, ast.Expr):
            val = node.value
            if isinstance(val, ast.Call):
                # If calling a method on an object (e.g. self.init(), obj.update()),
                # assume it mutates the object.
                # Running this in a process would lose the mutation in the main process.
                if isinstance(val.func, ast.Attribute):
                    # Exception: Allow known transparent/io calls like print, logger.info, etc.
                    name = val.func.attr
                    if name not in (
                        "print",
                        "info",
                        "debug",
                        "warning",
                        "error",
                        "log",
                        "write",
                        "tqdm",
                    ):
                        return False

        # Standard candidate check
        val = node.value if isinstance(node, (ast.Expr, ast.Assign)) else node
        if not isinstance(val, ast.Call):
            return False

        func_name = ""
        if isinstance(val.func, ast.Name):
            func_name = val.func.id
        elif isinstance(val.func, ast.Attribute):
            func_name = val.func.attr

        TRIVIAL_OPS = {
            "print",
            "len",
            "str",
            "int",
            "float",
            "bool",
            "type",
            "isinstance",
            "getattr",
            "setattr",
            "hasattr",
            "append",
            "extend",
            "debug",
            "info",
            "warning",
            "parse_args",
            "add_argument",
            "add_subparsers",
            "add_parser",
            "ArgumentParser",
            "parse_known_args",
        }
        return func_name not in TRIVIAL_OPS

    def _optimize_sequence(self, body_list: List[ast.AST]) -> List[ast.AST]:
        if not body_list or len(body_list) < 2:
            return body_list
        optimized_body = []
        current_group = []
        group_reads = set()
        group_writes = set()
        for stmt in body_list:
            stmt_reads, stmt_writes = self._get_rw_sets(stmt)
            is_candidate = self._is_candidate_for_task(stmt)
            has_conflict = (
                not stmt_reads.isdisjoint(group_writes)
                or not stmt_writes.isdisjoint(group_reads)
                or not stmt_writes.isdisjoint(group_writes)
            )
            if is_candidate and not has_conflict:
                current_group.append(stmt)
                group_reads.update(stmt_reads)
                group_writes.update(stmt_writes)
            else:
                if current_group:
                    if len(current_group) > 1:
                        optimized_body.extend(
                            self._create_task_dispatcher(current_group)
                        )
                    else:
                        optimized_body.extend(current_group)
                    current_group = []
                    group_reads = set()
                    group_writes = set()
                if is_candidate:
                    current_group.append(stmt)
                    group_reads.update(stmt_reads)
                    group_writes.update(stmt_writes)
                else:
                    optimized_body.append(stmt)
        if current_group:
            if len(current_group) > 1:
                optimized_body.extend(
                    self._create_task_dispatcher(current_group)
                )
            else:
                optimized_body.extend(current_group)
        return optimized_body

    def _is_transparent(self, node: ast.AST) -> bool:
        if isinstance(node, (ast.Expr, ast.Call)):
            call = node.value if isinstance(node, ast.Expr) else node
            if isinstance(call, ast.Call):
                func_name = ""
                if isinstance(call.func, ast.Name):
                    func_name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    func_name = call.func.attr
                return func_name in (
                    "print",
                    "info",
                    "debug",
                    "warning",
                    "error",
                    "log",
                    "write",
                    "tqdm",
                )
        return False

    def _are_tasks_compatible(self, stmt1: ast.AST, stmt2: ast.AST) -> bool:
        if not self._is_method_call(stmt1) or not self._is_method_call(stmt2):
            return False
        obj1, method1, _ = self._extract_call_info(stmt1)
        obj2, method2, _ = self._extract_call_info(stmt2)
        if method1 != method2 or not self._are_nodes_equal(obj1, obj2):
            return False
        _, writes1 = self._get_rw_sets(stmt1)
        reads2, _ = self._get_rw_sets(stmt2)
        return writes1.isdisjoint(reads2)

    def _is_safe_to_move_up(
        self, candidate_stmt: ast.AST, intervening_stmts: List[ast.AST]
    ) -> bool:
        cand_reads, cand_writes = self._get_rw_sets(candidate_stmt)
        obj_node, _, _ = self._extract_call_info(candidate_stmt)
        if isinstance(obj_node, ast.Name):
            cand_reads.add(obj_node.id)
        for stmt in intervening_stmts:
            stmt_reads, stmt_writes = self._get_rw_sets(stmt)
            if not cand_reads.isdisjoint(stmt_writes):
                return False
            if not cand_writes.isdisjoint(stmt_reads):
                return False
        return True

    def _is_method_call(self, node: ast.AST) -> bool:
        call = None
        if (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ) or (
            isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        ):
            call = node.value
        return bool(call and isinstance(call.func, ast.Attribute))

    def _extract_call_info(self, node: ast.AST):
        call_node = node.value if isinstance(node, ast.Expr) else node.value
        return (call_node.func.value, call_node.func.attr, call_node.args)

    def _create_parallel_batch_call(
        self, stmts: List[ast.AST], obj_node: ast.AST, method_name: str
    ) -> List[ast.AST]:
        first_call = (
            stmts[0].value
            if isinstance(stmts[0], (ast.Assign, ast.Expr))
            else stmts[0].value
        )
        ref_args_len = len(first_call.args)
        ref_keywords = {k.arg for k in first_call.keywords}
        args_list = []
        targets_list = []
        for stmt in stmts:
            call = (
                stmt.value
                if isinstance(stmt, (ast.Assign, ast.Expr))
                else stmt.value
            )
            if len(call.args) != ref_args_len:
                return stmts
            curr_kw = {k.arg for k in call.keywords}
            if curr_kw != ref_keywords:
                return stmts
            args_list.append(
                call.args[0] if call.args else ast.Constant(value=None)
            )
            if isinstance(stmt, ast.Assign):
                targets_list.append(stmt.targets[0])
            else:
                targets_list.append(None)
        args_name = f"_batch_args_{self.task_counter}"
        self.task_counter += 1
        assign_args = ast.Assign(
            targets=[ast.Name(id=args_name, ctx=ast.Store())],
            value=ast.List(elts=args_list, ctx=ast.Load()),
        )
        method_access = ast.Attribute(
            value=obj_node, attr=method_name, ctx=ast.Load()
        )
        if ref_args_len > 1 or ref_keywords:
            return stmts
        results_name = f"_batch_res_{self.task_counter}"
        run_call = ast.Assign(
            targets=[ast.Name(id=results_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="HyperBoost", ctx=ast.Load()),
                    attr="run",
                    ctx=ast.Load(),
                ),
                args=[method_access, ast.Name(id=args_name, ctx=ast.Load())],
                keywords=[
                    ast.keyword(
                        arg="backend", value=ast.Constant(value="processes")
                    )
                ],
            ),
        )
        unpack_stmts = []
        for idx, target in enumerate(targets_list):
            if target:
                unpack_stmts.append(
                    ast.Assign(
                        targets=[target],
                        value=ast.Subscript(
                            value=ast.Name(id=results_name, ctx=ast.Load()),
                            slice=ast.Constant(value=idx),
                            ctx=ast.Load(),
                        ),
                    )
                )
        return [assign_args, run_call, *unpack_stmts]

    def visit_FunctionDef(self, node):
        locals_map, args_map = self._analyze_local_scope(node)
        self.scope_stack.append((locals_map, args_map))
        self.generic_visit(node)
        node.body = self._optimize_sequence(node.body)
        self.scope_stack.pop()
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def _visit_container(self, node):
        self.generic_visit(node)
        node.body = self._optimize_sequence(node.body)
        if hasattr(node, "orelse") and node.orelse:
            node.orelse = self._optimize_sequence(node.orelse)
        return node

    def visit_With(self, node):
        node = self._visit_container(node)
        assigned_vars = set()
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                assigned_vars.add(elt.id)
        context_vars = set()
        for item in node.items:
            if item.optional_vars:
                if isinstance(item.optional_vars, ast.Name):
                    context_vars.add(item.optional_vars.id)
                elif isinstance(item.optional_vars, (ast.Tuple, ast.List)):
                    for elt in item.optional_vars.elts:
                        if isinstance(elt, ast.Name):
                            context_vars.add(elt.id)
        vars_to_preserve = assigned_vars - context_vars
        if not vars_to_preserve:
            return node
        node._vars_to_preserve = vars_to_preserve
        return node

    def visit_AsyncWith(self, node):
        return self._visit_container(node)

    def visit_If(self, node):
        if self._is_main_block(node):
            self.generic_visit(node)
            return node
        return self._visit_container(node)

    def _is_main_block(self, node: ast.If) -> bool:
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
        ):
            left = test.left
            right = test.comparators[0] if test.comparators else None
            if (
                isinstance(left, ast.Name)
                and left.id == "__name__"
                and (
                    isinstance(right, ast.Constant)
                    and right.value == "__main__"
                )
            ):
                return True
            if isinstance(right, ast.Name) and right.id == "__name__":
                if isinstance(left, ast.Constant) and left.value == "__main__":
                    return True
        return False

    def visit_Try(self, node):
        self.generic_visit(node)
        node.body = self._optimize_sequence(node.body)
        node.finalbody = self._optimize_sequence(node.finalbody)
        return node

    def visit_While(self, node):
        return self._visit_container(node)

    def _get_loop_body_writes(self, body: List[ast.AST]) -> Set[str]:
        """Get all variables written to in the loop body (recursive)."""
        return self._get_all_assigned_names(body)

    def _get_loop_outer_reads(self, node: ast.For) -> Set[str]:
        """
        Get variables that are read from outer scope (not defined in the loop).
        CRITICAL FIX: Uses improved _get_loop_local_vars that finds nested assignments.
        """
        reads = set()
        loop_locals = self._get_loop_local_vars(node)

        for stmt in node.body:
            for child in ast.walk(stmt):
                if (
                    isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and child.id not in loop_locals
                ):
                    reads.add(child.id)

        return reads

    def _get_accumulations(self, node: ast.For) -> Dict[str, ast.AugAssign]:
        accumulations = {}
        loop_locals = self._get_loop_local_vars(node)
        for stmt in node.body:
            if isinstance(stmt, ast.AugAssign) and isinstance(
                stmt.target, ast.Name
            ):
                var_name = stmt.target.id
                if var_name not in loop_locals and isinstance(
                    stmt.op,
                    (
                        ast.Add,
                        ast.Sub,
                        ast.Mult,
                        ast.BitOr,
                        ast.BitAnd,
                        ast.BitXor,
                    ),
                ):
                    accumulations[var_name] = stmt
        return accumulations

    def _has_complex_outer_writes(self, node: ast.For) -> bool:
        loop_locals = self._get_loop_local_vars(node)
        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name):
                                if target.value.id not in loop_locals:
                                    return True
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Subscript):
                            if isinstance(target.value, ast.Name):
                                if target.value.id not in loop_locals:
                                    if isinstance(target.slice, ast.Slice):
                                        return True

        return False

    def _has_loop_var_indexed_assignment(self, node: ast.For) -> bool:
        """
        Detect loops where the loop variable is used as an index in an assignment.

        Example patterns that should NOT be parallelized:
            for i in range(n):
                frames[i] = compute(i)  # i used as index in assignment

            for i in range(n):
                start = i * hop
                result[i] = data[start:end]  # i used as index, body has multiple statements

            for t in range(n_frames):
                mag[:, t] = something  # 2D array with loop var as column index

        These patterns are problematic because:
        1. The current slice assignment (result[:] = ...) loses index information
        2. Multi-statement bodies with indexed assignments can't be correctly parallelized
        3. Multi-dimensional subscripts (array[:, i]) can't be correctly reassembled
        """
        if not isinstance(node.target, ast.Name):
            return False
        loop_var = node.target.id

        # Check if loop variable is used in any subscript assignment target
        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Subscript):
                            # Check if the slice uses the loop variable
                            slice_node = target.slice
                            if self._uses_name(slice_node, loop_var):
                                # Found: array[...i...] = ... where i is loop var

                                # CRITICAL FIX: Multi-dimensional subscripts are ALWAYS unsafe
                                # e.g., array[:, i] or array[i, :] or array[a, b, i]
                                # These create Tuple slices that can't be correctly reassembled
                                if isinstance(slice_node, ast.Tuple):
                                    return True

                                # Also check for Slice objects - indicates complex indexing
                                # e.g., array[i:j] where i is loop var
                                if isinstance(slice_node, ast.Slice):
                                    return True

                                # Multiple statements with indexed assignment - always unsafe
                                if len(node.body) > 1:
                                    return True

                                # Single statement - check if value has complex slicing
                                value = child.value
                                for val_child in ast.walk(value):
                                    if isinstance(val_child, ast.Subscript):
                                        if isinstance(
                                            val_child.slice, ast.Slice
                                        ):
                                            # Slicing in the value - complex pattern
                                            return True
                                        # Also catch multi-dimensional reads
                                        if isinstance(
                                            val_child.slice, ast.Tuple
                                        ):
                                            return True
        return False

    def _has_index_offset_pattern(self, node: ast.For) -> bool:
        if not isinstance(node.iter, ast.Call):
            return False
        if not isinstance(node.iter.func, ast.Name):
            return False
        if node.iter.func.id != "range":
            return False

        has_offset = False
        if len(node.iter.args) >= 2:
            start_arg = node.iter.args[0]
            if (
                (isinstance(start_arg, ast.Constant) and start_arg.value != 0)
                or (isinstance(start_arg, ast.Num) and start_arg.n != 0)
                or isinstance(start_arg, ast.Name)
            ):
                has_offset = True

        if not has_offset:
            return False

        if not isinstance(node.target, ast.Name):
            return False
        loop_var = node.target.id

        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Subscript):
                    slice_node = child.slice
                    if self._uses_name(slice_node, loop_var):
                        return True

        return False

    def _uses_name(self, node: ast.AST, name: str) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Tuple):
            return any(self._uses_name(elt, name) for elt in node.elts)
        if isinstance(node, ast.Index):  # Python 3.8 compat
            return self._uses_name(node.value, name)
        return any(
            self._uses_name(child, name) for child in ast.iter_child_nodes(node)
        )

    def _has_multidim_subscript_assignment(self, node: ast.For) -> bool:
        """
        Detect loops that assign to multi-dimensional array subscripts.

        Patterns detected:
            for t in range(n):
                array[:, t] = ...      # Tuple slice with loop var
                array[t, :] = ...      # Tuple slice with loop var
                array[a, b, t] = ...   # Any multi-dimensional with loop var

        These CANNOT be safely parallelized with the current implementation
        because the result[:] = ... reassembly would create wrong shapes.
        """
        if not isinstance(node.target, ast.Name):
            return False
        loop_var = node.target.id

        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Subscript):
                            slice_node = target.slice
                            # Check if it's a multi-dimensional subscript (Tuple)
                            if isinstance(slice_node, ast.Tuple):
                                # Check if any element uses the loop variable
                                if self._uses_name(slice_node, loop_var):
                                    return True
        return False

    def visit_For(self, node):
        node = self._visit_container(node)

        # Check for control flow that breaks parallelization
        for child in ast.walk(node):
            if isinstance(
                child,
                (ast.Break, ast.Continue, ast.Return, ast.Yield, ast.YieldFrom),
            ):
                return node

        # Skip loops where loop variable is used as array index
        # AND range doesn't start from 0
        if self._has_index_offset_pattern(node):
            return node

        # Skip loops where loop variable is used as index in complex assignments
        # These can't be correctly parallelized with the current implementation
        if self._has_loop_var_indexed_assignment(node):
            return node

        # CRITICAL FIX: Also check for multi-dimensional subscript assignments
        # This catches patterns like array[:, t] = ... that _has_loop_var_indexed_assignment might miss
        if self._has_multidim_subscript_assignment(node):
            return node

        # Check if this is a parallelizable iteration pattern
        is_candidate = False
        if isinstance(node.iter, ast.Call) and isinstance(
            node.iter.func, ast.Name
        ):
            if node.iter.func.id in ["range", "enumerate", "zip"]:
                is_candidate = True
        elif isinstance(node.iter, ast.Name):
            is_candidate = True

        if not is_candidate:
            return node

        if self._has_complex_outer_writes(node):
            return node

        accumulations = self._get_accumulations(node)
        outer_reads = self._get_loop_outer_reads(node)

        # CRITICAL FIX: Check for problematic patterns before parallelizing
        # Get variables from outer scope to detect potential issues
        outer_scope_vars = set()
        if self.scope_stack:
            locals_created, arguments = self.scope_stack[-1]
            outer_scope_vars = locals_created | arguments

        # Pass accumulations keys to be exempted from "unsafe outer write" check
        if self._has_problematic_loop_patterns(
            node, outer_scope_vars, set(accumulations.keys())
        ):
            return node

        if not accumulations and self._is_vectorizable_loop(node):
            return self._create_vectorized_loop(node)

        has_subscript_assign = False
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Subscript):
                        has_subscript_assign = True

        backend_choice = "threads" if has_subscript_assign else "auto"
        return self._create_loop_worker(
            node, backend_choice, outer_reads, accumulations
        )
