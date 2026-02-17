import asyncio
import io
import mmap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from turboscan.hardware.config import HARDWARE


class FastFileReader:
    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.max_workers = max_workers or HARDWARE.cpu_count * 2
        self._mmap_cache: Dict[str, mmap.mmap] = {}
        self._file_handles: Dict[str, io.FileIO] = {}

    def read_file(self, path: Path) -> str:
        try:
            size = path.stat().st_size
            # For very small files, use direct read with optimized buffer
            if size < 4096:
                return path.read_text(encoding="utf-8", errors="ignore")
            # For small to medium files, use buffered read
            elif size < 64 * 1024:
                with open(
                    path, encoding="utf-8", errors="ignore", buffering=8192
                ) as f:
                    return f.read()
            # For large files, use memory mapping
            else:
                with open(path, "rb") as f, mmap.mmap(
                    f.fileno(), 0, access=mmap.ACCESS_READ
                ) as mm:
                    content = mm.read().decode("utf-8", errors="ignore")
                return content
        except Exception:
            return ""

    def read_files_parallel(self, paths: List[Path]) -> Dict[Path, str]:
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.read_file, p): p for p in paths}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    results[path] = future.result()
                except Exception:
                    results[path] = ""
        return results

    async def read_files_async(self, paths: List[Path]) -> Dict[Path, str]:
        loop = asyncio.get_event_loop()

        async def read_one(path: Path) -> Tuple[Path, str]:
            content = await loop.run_in_executor(None, self.read_file, path)
            return (path, content)

        tasks = [read_one(p) for p in paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            path: content
            for path, content in results
            if isinstance(content, str) or isinstance(path, Path)
        }


FAST_READER = FastFileReader()
