import hashlib
import multiprocessing
import multiprocessing.pool
from contextlib import contextmanager
from typing import Any, List

try:
    import torch

    TORCH_AVAIL = True
    GPU_AVAIL = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if GPU_AVAIL else 0
except ImportError:
    TORCH_AVAIL = False
    GPU_AVAIL = False
    GPU_COUNT = 0
    torch = None
try:
    import numpy as np

    NUMPY_AVAIL = True
except ImportError:
    np = None
    NUMPY_AVAIL = False


class GPUAccelerator:
    def __init__(self) -> None:
        self.device = None
        self.streams: List[Any] = []
        if GPU_AVAIL and TORCH_AVAIL:
            self.device = torch.device("cuda:0")
            self.streams = [
                torch.cuda.Stream() for _ in range(min(8, GPU_COUNT * 4))
            ]

    def batch_hash(self, items: List[str]) -> List[str]:
        if not items:
            return []

        if len(items) > 1000:
            num_workers = min(4, multiprocessing.cpu_count())
            with multiprocessing.pool.ThreadPool(processes=num_workers) as pool:
                return pool.map(
                    lambda item: hashlib.blake2b(
                        str(item).encode(), digest_size=16
                    ).hexdigest(),
                    items,
                )
        else:
            return [
                hashlib.blake2b(str(item).encode(), digest_size=16).hexdigest()
                for item in items
            ]

    @contextmanager
    def stream_context(self, idx: int = 0):
        if self.streams:
            stream = self.streams[idx % len(self.streams)]
            with torch.cuda.stream(stream):
                yield stream
            stream.synchronize()
        else:
            yield None


GPU_ACCELERATOR = GPUAccelerator()
