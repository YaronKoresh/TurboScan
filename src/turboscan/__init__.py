import contextlib
import multiprocessing as mp
import os
import platform
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")
os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONOPTIMIZE"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "0"
warnings.filterwarnings("ignore")

if platform.system() == "Windows":
    mp.freeze_support()
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn", force=True)

try:
    import cloudpickle

    CLOUDPICKLE_AVAIL = True
except ImportError:
    CLOUDPICKLE_AVAIL = False
    cloudpickle = None

try:
    from loky import get_reusable_executor

    LOKY_AVAIL = True
except ImportError:
    LOKY_AVAIL = False
    get_reusable_executor = None

try:
    import ray

    RAY_AVAIL = True
except ImportError:
    RAY_AVAIL = False

try:
    import torch

    TORCH_AVAIL = True
    GPU_AVAIL = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if GPU_AVAIL else 0
    if GPU_AVAIL:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
except ImportError:
    TORCH_AVAIL = False
    GPU_AVAIL = False
    GPU_COUNT = 0
    torch = None

try:
    from numba import cuda as numba_cuda, jit, njit, prange, vectorize

    NUMBA_AVAIL = True
    NUMBA_CUDA_AVAIL = numba_cuda.is_available()
except ImportError:
    NUMBA_AVAIL = False
    NUMBA_CUDA_AVAIL = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args or not callable(args[0]) else args[0]

    jit = njit
    prange = range
    vectorize = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    RICH_AVAIL = True
    console = Console()
except ImportError:
    RICH_AVAIL = False
    console = None

try:
    from diskcache import Cache, FanoutCache

    CACHE_AVAIL = True
except ImportError:
    CACHE_AVAIL = False

try:
    import numpy as np

    NUMPY_AVAIL = True
except ImportError:
    NUMPY_AVAIL = False
    np = None

from turboscan.analysis import FunctionAnalysis, FunctionAnalyzer
from turboscan.ast_transform import HyperAutoParallelizer, MutationDetector
from turboscan.auditor import HyperAuditor
from turboscan.cache import HYPER_CACHE, BloomFilter, HyperCache
from turboscan.execution import HyperBoost, boost_all
from turboscan.executor import HyperExecutor
from turboscan.gpu import GPU_ACCELERATOR, GPUAccelerator
from turboscan.hardware import HARDWARE, HardwareConfig, detect_hardware
from turboscan.indexing import (
    HyperIndexer,
    ModuleInfo,
    Scope,
    Signature,
    SymbolDef,
)
from turboscan.io import FAST_READER, FastFileReader
from turboscan.jit import JIT_INJECTOR, JITInjector
from turboscan.registry import HyperRegistry
from turboscan.resolver import HyperResolver
from turboscan.validator import HyperValidator

__all__ = [
    "CACHE_AVAIL",
    "CLOUDPICKLE_AVAIL",
    "FAST_READER",
    "GPU_ACCELERATOR",
    "GPU_AVAIL",
    "GPU_COUNT",
    "HARDWARE",
    "HYPER_CACHE",
    "JIT_INJECTOR",
    "LOKY_AVAIL",
    "NUMBA_AVAIL",
    "NUMBA_CUDA_AVAIL",
    "NUMPY_AVAIL",
    "RAY_AVAIL",
    "RICH_AVAIL",
    "TORCH_AVAIL",
    "BloomFilter",
    "FastFileReader",
    "FunctionAnalysis",
    "FunctionAnalyzer",
    "GPUAccelerator",
    "HardwareConfig",
    "HyperAuditor",
    "HyperAutoParallelizer",
    "HyperBoost",
    "HyperCache",
    "HyperExecutor",
    "HyperIndexer",
    "HyperRegistry",
    "HyperResolver",
    "HyperValidator",
    "JITInjector",
    "ModuleInfo",
    "MutationDetector",
    "Scope",
    "Signature",
    "SymbolDef",
    "boost_all",
    "console",
    "detect_hardware",
    "jit",
    "njit",
    "prange",
    "vectorize",
]
