"""Integrated execution engine with audit, AST optimization, and JIT compilation."""

import ast
import builtins
import sys
import time
from pathlib import Path
from typing import Any, Dict

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAIL = True
    console = Console()
except ImportError:
    RICH_AVAIL = False
    console = None
try:
    from numba import jit, njit, prange, vectorize

    NUMBA_AVAIL = True
except ImportError:
    NUMBA_AVAIL = False
    njit = jit = vectorize = prange = None
try:
    import cloudpickle

    CLOUDPICKLE_AVAIL = True
except ImportError:
    CLOUDPICKLE_AVAIL = False
from turboscan.ast_transform.parallelizer import HyperAutoParallelizer
from turboscan.auditor.auditor import HyperAuditor
from turboscan.cache.hyper_cache import HYPER_CACHE
from turboscan.execution.hyper_boost import GPU_COUNT, HyperBoost
from turboscan.gpu.accelerator import GPU_ACCELERATOR
from turboscan.hardware.config import HARDWARE
from turboscan.io.file_reader import FAST_READER
from turboscan.jit.injector import JITInjector


class HyperExecutor:
    def __init__(
        self,
        script_path: str,
        audit_first: bool = True,
        optimize: bool = True,
        inject_jit: bool = True,
    ) -> None:
        self.script_path = Path(script_path).resolve()
        self.audit_first = audit_first
        self.optimize = optimize
        self.inject_jit = inject_jit
        self.stats = {
            "parallelized_loops": 0,
            "parallelized_tasks": 0,
            "jit_functions": 0,
            "vectorized_functions": 0,
            "audit_errors": 0,
            "audit_warnings": 0,
        }

    def _run_audit(self) -> bool:
        project_root = self.script_path.parent
        auditor = HyperAuditor(
            project_root,
            excludes={
                ".git",
                "node_modules",
                "venv",
                "__pycache__",
                "env",
                ".venv",
                "build",
                "dist",
            },
            check_unused=True,
        )
        errors, warnings = auditor.run()
        self.stats["audit_errors"] = errors
        self.stats["audit_warnings"] = warnings
        if errors > 0:
            print(
                f"\n⚠️  Found {errors} errors. Continue anyway? (y/n): ", end=""
            )
            try:
                response = input().strip().lower()
                return response in ("y", "yes")
            except Exception:
                return True
        return True

    def _optimize_ast(self, tree: ast.Module) -> ast.Module:
        parallelizer = HyperAutoParallelizer()
        tree = parallelizer.visit(tree)
        self.stats["parallelized_loops"] = parallelizer.loop_counter
        self.stats["parallelized_tasks"] = parallelizer.task_counter
        if self.inject_jit and NUMBA_AVAIL:
            jit_injector = JITInjector()
            tree = jit_injector.inject(tree)
            self.stats["jit_functions"] = jit_injector.jit_count
            self.stats["vectorized_functions"] = jit_injector.vectorize_count
        ast.fix_missing_locations(tree)
        return tree

    def _prepare_globals(self) -> Dict[str, Any]:
        global_env = {
            "__name__": "__main__",
            "__file__": str(self.script_path),
            "__builtins__": builtins,
            "HyperBoost": HyperBoost,
            "Boost": HyperBoost,
            "GPU_ACCELERATOR": GPU_ACCELERATOR,
            "HYPER_CACHE": HYPER_CACHE,
            "HARDWARE": HARDWARE,
            "prange": prange,
        }
        if NUMBA_AVAIL:
            global_env["njit"] = njit
            global_env["jit"] = jit
            global_env["vectorize"] = vectorize
            global_env["prange"] = prange
        return global_env

    def execute(self) -> None:
        if not self.script_path.exists():
            print(f"Error: File '{self.script_path}' not found.")
            sys.exit(1)
        if RICH_AVAIL:
            serializer = "cloudpickle" if CLOUDPICKLE_AVAIL else "pickle"
            console.print(
                Panel(
                    f"[bold green]⚡ TurboScan HYPER Execution[/bold green]\n[dim]Script: {self.script_path.name}[/dim]\n[dim]CPUs: {HARDWARE.cpu_count} | GPUs: {GPU_COUNT}[/dim]\n[dim]Serializer: {serializer}[/dim]",
                    border_style="green",
                )
            )
        else:
            serializer = (
                "cloudpickle ✓"
                if CLOUDPICKLE_AVAIL
                else "pickle (consider: pip install cloudpickle)"
            )
            print(f"⚡ TurboScan HYPER: Executing {self.script_path.name}")
            print(f"   CPUs: {HARDWARE.cpu_count} | GPUs: {GPU_COUNT}")
            print(f"   Serializer: {serializer}")
        if self.audit_first:
            print("\n[Phase 1/3] Running integrated audit...")
            if not self._run_audit():
                print("Execution aborted.")
                sys.exit(1)
        print("\n[Phase 2/3] Optimizing code...")
        source_code = FAST_READER.read_file(self.script_path)
        try:
            tree = ast.parse(source_code)
            if self.optimize:
                tree = self._optimize_ast(tree)
            compiled_code = compile(
                tree, filename=str(self.script_path), mode="exec"
            )
            if RICH_AVAIL:
                table = Table(title="Optimization Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Count", style="green")
                table.add_row(
                    "Parallelized Loops", str(self.stats["parallelized_loops"])
                )
                table.add_row(
                    "Parallelized Tasks", str(self.stats["parallelized_tasks"])
                )
                table.add_row("JIT Functions", str(self.stats["jit_functions"]))
                table.add_row(
                    "Vectorized Functions",
                    str(self.stats["vectorized_functions"]),
                )
                console.print(table)
            else:
                print(
                    f"   Parallelized: {self.stats['parallelized_loops']} loops, {self.stats['parallelized_tasks']} tasks"
                )
                print(
                    f"   JIT: {self.stats['jit_functions']} functions, {self.stats['vectorized_functions']} vectorized"
                )
        except SyntaxError as e:
            print(f"Syntax error at line {e.lineno}: {e.msg}")
            sys.exit(1)
        except Exception as e:
            print(f"⚠️  Optimization failed ({e}). Running unoptimized.")
            compiled_code = compile(
                source_code, filename=str(self.script_path), mode="exec"
            )
        print("\n[Phase 3/3] Executing...")
        print("=" * 60)
        sys.path.insert(0, str(self.script_path.parent))
        global_env = self._prepare_globals()
        start_time = time.perf_counter()
        try:
            exec(compiled_code, global_env)
        except KeyboardInterrupt:
            print("\n\nExecution interrupted.")
        except SystemExit:
            pass
        except Exception:
            import traceback

            traceback.print_exc()
        elapsed = time.perf_counter() - start_time
        print("=" * 60)
        print(f"\n✅ Execution completed in {elapsed:.3f}s")
        cache_stats = HYPER_CACHE.stats
        if cache_stats["hits"] + cache_stats["misses"] > 0:
            print(
                f"   Cache: {cache_stats['hit_rate']:.1%} hit rate ({cache_stats['hits']} hits, {cache_stats['misses']} misses)"
            )
