import ast
from pathlib import Path
from typing import List, Set, Tuple
try:
    from rich.panel import Panel
    from rich.console import Console
    RICH_AVAIL = True
    console = Console()
except ImportError:
    RICH_AVAIL = False
    console = None
from turboscan.registry.registry import HyperRegistry
from turboscan.resolver.resolver import HyperResolver
from turboscan.validator.validator import HyperValidator
from turboscan.execution.hyper_boost import HyperBoost, GPU_COUNT
from turboscan.io.file_reader import FAST_READER
from turboscan.hardware.config import HARDWARE

class HyperAuditor:
    def __init__(self, root: Path, excludes: Set[str], check_unused: bool=True):
        self.root = root
        self.excludes = excludes
        self.check_unused = check_unused
        self.registry = HyperRegistry(root, excludes)
        self.registry.scan()
        self.registry.first_pass_indexing()
        self.registry.second_pass_star_imports()
        self.resolver = HyperResolver(self.registry)
    def _validate_file(self, file_path: Path) -> List[str]:
        issues = []
        try:
            content = self.registry._file_contents.get(file_path, '')
            if not content:
                content = FAST_READER.read_file(file_path)
            tree = ast.parse(content)
            validator = HyperValidator(self.registry, file_path, self.resolver, self.check_unused)
            validator.visit(tree)
            rel = file_path.relative_to(self.root)
            for line, msg in sorted(validator.errors, key=lambda x: x[0]):
                issues.append(f'[ERROR] {rel}:{line} -> {msg}')
            for line, msg in sorted(validator.warnings, key=lambda x: x[0]):
                issues.append(f'[WARN] {rel}:{line} -> {msg}')
            if self.check_unused:
                unused = validator.get_unused_imports()
                for u in sorted(unused):
                    issues.append(f"[UNUSED] {rel} -> Imported '{u}' is never used")
            return issues
        except SyntaxError as e:
            return [f'[ERROR] {file_path}: Syntax error at line {e.lineno}: {e.msg}']
        except Exception as e:
            return [f'[WARN] Skipped {file_path}: {e}']
    def run(self) -> Tuple[int, int]:
        if RICH_AVAIL:
            console.print(Panel(f'[bold cyan]TurboScan HYPER Audit[/bold cyan]\n[dim]Project: {self.root}[/dim]\n[dim]Modules: {len(self.registry.modules)}[/dim]\n[dim]CPUs: {HARDWARE.cpu_count} | GPUs: {GPU_COUNT}[/dim]', title='⚡ HYPER MODE', border_style='cyan'))
        else:
            print(f'--- TurboScan HYPER Audit: {self.root} ---')
            print(f'Indexed {len(self.registry.modules)} internal modules')
        sorted_files = sorted(self.registry.files_map.keys(), key=str)
        all_results = HyperBoost.run(self._validate_file, sorted_files, quiet=False, backend='auto')
        flat_issues = []
        for file_res in all_results:
            if isinstance(file_res, list):
                flat_issues.extend(file_res)
        for issue in flat_issues:
            print(issue)
        error_count = sum((1 for i in flat_issues if '[ERROR]' in i))
        warn_count = sum((1 for i in flat_issues if '[WARN]' in i))
        if RICH_AVAIL:
            console.print(f'\n[bold]Audit complete.[/bold] Errors: {error_count}, Warnings: {warn_count}')
        else:
            print(f'\nAudit complete. Errors: {error_count}, Warnings: {warn_count}')
        return (error_count, warn_count)
