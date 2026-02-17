import argparse
import sys
from pathlib import Path

from turboscan import (
    GPU_AVAIL,
    HARDWARE,
    NUMBA_AVAIL,
    NUMPY_AVAIL,
    RAY_AVAIL,
    RICH_AVAIL,
    HyperAuditor,
    HyperBoost,
    HyperExecutor,
    console,
)

if RICH_AVAIL:
    from rich.table import Table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TurboScan - Maximum Performance Python Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\nExamples:\n  turboscan.py run script.py          Run with full optimization\n  turboscan.py run script.py --fast   Skip audit for faster startup\n  turboscan.py audit .                Audit current directory\n  turboscan.py audit /path/to/project Audit specific project\n        ",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser(
        "run", help="Run a Python script with HYPER optimization"
    )
    run_parser.add_argument("script", help="Script to execute")
    run_parser.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="Script arguments"
    )
    run_parser.add_argument(
        "--fast", action="store_true", help="Skip audit for faster startup"
    )
    run_parser.add_argument(
        "--no-optimize", action="store_true", help="Disable AST optimization"
    )
    run_parser.add_argument(
        "--no-jit", action="store_true", help="Disable JIT injection"
    )
    run_parser.add_argument(
        "--debug", action="store_true", help="Enable Debug Mode"
    )
    audit_parser = subparsers.add_parser(
        "audit", help="Audit a project directory"
    )
    audit_parser.add_argument(
        "root", nargs="?", default=".", help="Project root directory"
    )
    audit_parser.add_argument(
        "--exclude",
        nargs="+",
        default=[
            ".git",
            "node_modules",
            "venv",
            "__pycache__",
            "tests",
            "docs",
            "env",
            ".venv",
            "build",
            "dist",
        ],
        help="Directories to exclude",
    )
    audit_parser.add_argument(
        "--no-unused", action="store_true", help="Skip unused import check"
    )
    subparsers.add_parser("info", help="Show system information")
    args = parser.parse_args()
    if args.command == "info" or (args.command is None and len(sys.argv) == 1):
        if RICH_AVAIL:
            table = Table(title="TurboScan - System Info")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_row(
                "CPU Cores",
                f"{HARDWARE.cpu_count} ({HARDWARE.cpu_count_physical} physical)",
            )
            table.add_row(
                "Memory", f"{HARDWARE.memory_total / 1024**3:.1f} GB total"
            )
            table.add_row("GPU Available", "Yes" if GPU_AVAIL else "No")
            if GPU_AVAIL:
                for i, name in enumerate(HARDWARE.gpu_names):
                    table.add_row(
                        f"  GPU {i}",
                        f"{name} ({HARDWARE.gpu_memory[i] / 1024**3:.1f} GB)",
                    )
            table.add_row("Ray", "Available" if RAY_AVAIL else "Not installed")
            table.add_row(
                "Numba JIT", "Available" if NUMBA_AVAIL else "Not installed"
            )
            table.add_row(
                "NumPy", "Available" if NUMPY_AVAIL else "Not installed"
            )
            console.print(table)
        else:
            print("TurboScan - System Info")
            print(
                f"  CPUs: {HARDWARE.cpu_count} ({HARDWARE.cpu_count_physical} physical)"
            )
            print(f"  Memory: {HARDWARE.memory_total / 1024**3:.1f} GB")
            print(f"  GPU: {('Yes' if GPU_AVAIL else 'No')}")
            print(f"  Ray: {('Yes' if RAY_AVAIL else 'No')}")
            print(f"  Numba: {('Yes' if NUMBA_AVAIL else 'No')}")
        return
    elif args.command == "audit":
        root_path = Path(args.root).resolve()
        auditor = HyperAuditor(
            root_path, set(args.exclude), check_unused=not args.no_unused
        )
        errors, _warnings = auditor.run()
        sys.exit(1 if errors > 0 else 0)
    elif args.command == "run":
        if args.debug:
            HyperBoost.set_debug(True)
        sys.argv = [args.script, *args.script_args]
        executor = HyperExecutor(
            args.script,
            audit_first=not args.fast,
            optimize=not args.no_optimize,
            inject_jit=not args.no_jit,
        )
        executor.execute()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
