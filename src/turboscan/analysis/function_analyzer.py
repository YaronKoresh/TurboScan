import os
import sys
import pickle
import inspect
from typing import Callable, Dict, Optional, Set, Tuple
from dataclasses import dataclass
try:
    import cloudpickle
    CLOUDPICKLE_AVAIL = True
except ImportError:
    CLOUDPICKLE_AVAIL = False
    cloudpickle = pickle

class FunctionAnalysis:
    is_picklable: bool = True
    uses_threads: bool = False
    uses_multiprocessing: bool = False
    uses_async: bool = False
    has_closures: bool = False
    is_lambda: bool = False
    is_local_func: bool = False
    estimated_weight: str = 'unknown'
    failure_reason: Optional[str] = None
    confidence: float = 0.5
    @property
    def prefers_processes(self) -> bool:
        if not self.is_picklable:
            return False
        if self.uses_threads:
            return True
        if self.estimated_weight == 'cpu_bound':
            return True
        if self.estimated_weight == 'mixed':
            return True
        return True
    @property
    def should_force_threads(self) -> bool:
        if not self.is_picklable:
            return True
        if self.is_lambda and (not CLOUDPICKLE_AVAIL):
            return True
        if self.is_local_func and (not CLOUDPICKLE_AVAIL):
            return True
        return False

class FunctionAnalyzer:
    THREAD_PATTERNS = frozenset({'ThreadPoolExecutor', 'threading.Thread', 'Thread(', 'concurrent.futures', 'ThreadPool', 'threading.Lock', 'threading.RLock', 'threading.Semaphore', 'threading.Event', 'threading.Condition', 'threading.Barrier'})
    THREAD_NAMES = frozenset({'ThreadPoolExecutor', 'Thread', 'Lock', 'RLock', 'Semaphore', 'Event', 'Condition', 'Barrier', 'threading', 'ThreadPool', 'lock', '_thread'})
    MP_PATTERNS = frozenset({'ProcessPoolExecutor', 'multiprocessing.Process', 'Process(', 'multiprocessing.Pool', 'Pool(', 'mp.Pool', 'mp.Process'})
    ASYNC_PATTERNS = frozenset({'async def', 'await ', 'asyncio.', 'aiohttp', 'aiofiles'})
    CPU_BOUND_PATTERNS = frozenset({'numpy', 'np.', 'scipy', 'sklearn', 'torch', 'tensorflow', 'librosa', 'soundfile', 'cv2', 'PIL', 'numba', 'math.', 'cmath.', 'statistics.'})
    IO_BOUND_PATTERNS = frozenset({'open(', 'read(', 'write(', 'requests.', 'urllib', 'socket.', 'http.', 'ftp.', 'smtp.', 'json.load', 'pickle.load', 'yaml.load'})
    _analysis_cache: Dict[int, FunctionAnalysis] = {}
    _source_cache: Dict[int, str] = {}
    @classmethod
    def analyze(cls, func: Callable) -> FunctionAnalysis:
        func_id = id(func)
        if func_id in cls._analysis_cache:
            return cls._analysis_cache[func_id]
        analysis = FunctionAnalysis()
        func_name = getattr(func, '__name__', '')
        func_qualname = getattr(func, '__qualname__', '')
        analysis.is_lambda = func_name == '<lambda>'
        analysis.is_local_func = '<locals>' in func_qualname
        if hasattr(func, '__closure__') and func.__closure__:
            analysis.has_closures = True
            for cell in func.__closure__:
                try:
                    cell_val = cell.cell_contents
                    cell_type = type(cell_val).__name__
                    cell_module = getattr(type(cell_val), '__module__', '')
                    
                    if (cell_type in cls.THREAD_NAMES or 
                        'threading' in cell_module or 
                        'concurrent' in cell_module or
                        cell_module == '_thread'):
                        analysis.uses_threads = True
                        break
                except ValueError:
                    pass
        if hasattr(func, '__code__'):
            code = func.__code__
            code_names = set(code.co_names) if hasattr(code, 'co_names') else set()
            free_vars = set(code.co_freevars) if hasattr(code, 'co_freevars') else set()
            all_names = code_names | free_vars
            if all_names & cls.THREAD_NAMES:
                analysis.uses_threads = True
        source = cls._get_source(func)
        if source:
            # Optimized pattern matching - check if any pattern exists
            has_thread = any(p in source for p in cls.THREAD_PATTERNS)
            has_mp = any(p in source for p in cls.MP_PATTERNS)
            has_async = any(p in source for p in cls.ASYNC_PATTERNS)
            
            if has_thread:
                analysis.uses_threads = True
            if has_mp:
                analysis.uses_multiprocessing = True
            if has_async:
                analysis.uses_async = True
            
            # Count CPU and IO indicators
            cpu_indicators = sum(1 for p in cls.CPU_BOUND_PATTERNS if p in source)
            io_indicators = sum(1 for p in cls.IO_BOUND_PATTERNS if p in source)
            if cpu_indicators > io_indicators * 2:
                analysis.estimated_weight = 'cpu_bound'
                analysis.confidence = 0.8
            elif io_indicators > cpu_indicators * 2:
                analysis.estimated_weight = 'io_bound'
                analysis.confidence = 0.7
            elif cpu_indicators > 0 or io_indicators > 0:
                analysis.estimated_weight = 'mixed'
                analysis.confidence = 0.6
            if analysis.uses_threads:
                analysis.confidence = min(1.0, analysis.confidence + 0.2)
        elif hasattr(func, '__module__') and (not analysis.uses_threads):
            try:
                import sys
                mod = sys.modules.get(func.__module__)
                if mod:
                    mod_source = getattr(mod, '__file__', '')
                    if mod_source and os.path.exists(mod_source):
                        try:
                            with open(mod_source, 'r', encoding='utf-8', errors='ignore') as f:
                                mod_content = f.read()
                            if any((p in mod_content for p in cls.THREAD_PATTERNS)):
                                analysis.confidence = max(0.3, analysis.confidence - 0.2)
                        except:
                            pass
            except:
                pass
        analysis.is_picklable, analysis.failure_reason = cls._test_pickle(func)
        cls._analysis_cache[func_id] = analysis
        return analysis
    @classmethod
    def _get_source(cls, func: Callable) -> Optional[str]:
        func_id = id(func)
        if func_id in cls._source_cache:
            return cls._source_cache[func_id]
        source = None
        try:
            source = inspect.getsource(func)
        except (TypeError, OSError):
            pass
        if source is None and hasattr(func, '__self__'):
            try:
                source = inspect.getsource(type(func.__self__))
            except (TypeError, OSError):
                pass
        cls._source_cache[func_id] = source
        return source
    @classmethod
    def _test_pickle(cls, func: Callable) -> Tuple[bool, Optional[str]]:
        serializer = cloudpickle if CLOUDPICKLE_AVAIL else pickle
        try:
            serializer.dumps(func)
            return (True, None)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            return (False, str(e))
        except Exception as e:
            return (False, f'Unknown: {type(e).__name__}: {str(e)}')
    @classmethod
    def clear_cache(cls):
        cls._analysis_cache.clear()
        cls._source_cache.clear()
