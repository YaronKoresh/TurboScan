import os
import sys
import warnings
import platform
import multiprocessing as mp

sys.stdout.reconfigure(encoding='utf-8')
os.environ['RAY_DISABLE_METRICS_COLLECTION'] = '1'
os.environ['RAY_dedup_logs'] = '0'
os.environ['PYTHONOPTIMIZE'] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '0'
warnings.filterwarnings('ignore')

if platform.system() == 'Windows':
    mp.freeze_support()
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

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
    from numba import jit, njit, prange, vectorize, cuda as numba_cuda
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
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.console import Console
    from rich.panel import Panel
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

from turboscan.hardware import HardwareConfig, detect_hardware, HARDWARE
from turboscan.cache import BloomFilter, HyperCache, HYPER_CACHE
from turboscan.gpu import GPUAccelerator, GPU_ACCELERATOR
from turboscan.io import FastFileReader, FAST_READER
from turboscan.jit import JITInjector, JIT_INJECTOR
from turboscan.analysis import FunctionAnalysis, FunctionAnalyzer
from turboscan.execution import HyperBoost, boost_all
from turboscan.ast_transform import MutationDetector, HyperAutoParallelizer
from turboscan.indexing import Signature, SymbolDef, ModuleInfo, Scope, HyperIndexer
from turboscan.registry import HyperRegistry
from turboscan.resolver import HyperResolver
from turboscan.validator import HyperValidator
from turboscan.auditor import HyperAuditor
from turboscan.executor import HyperExecutor

__all__ = [
    'CLOUDPICKLE_AVAIL',
    'LOKY_AVAIL',
    'RAY_AVAIL',
    'TORCH_AVAIL',
    'GPU_AVAIL',
    'GPU_COUNT',
    'NUMBA_AVAIL',
    'NUMBA_CUDA_AVAIL',
    'RICH_AVAIL',
    'CACHE_AVAIL',
    'NUMPY_AVAIL',
    'console',
    'njit',
    'jit',
    'prange',
    'vectorize',
    'HardwareConfig',
    'detect_hardware',
    'HARDWARE',
    'BloomFilter',
    'HyperCache',
    'HYPER_CACHE',
    'GPUAccelerator',
    'GPU_ACCELERATOR',
    'FastFileReader',
    'FAST_READER',
    'JITInjector',
    'JIT_INJECTOR',
    'FunctionAnalysis',
    'FunctionAnalyzer',
    'HyperBoost',
    'boost_all',
    'MutationDetector',
    'HyperAutoParallelizer',
    'Signature',
    'SymbolDef',
    'ModuleInfo',
    'Scope',
    'HyperIndexer',
    'HyperRegistry',
    'HyperResolver',
    'HyperValidator',
    'HyperAuditor',
    'HyperExecutor',
]
