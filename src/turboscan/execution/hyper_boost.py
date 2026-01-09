"""
HyperBoost

Fixes:
1. @lru_cache pickling errors - Patches classes at runtime before serialization
2. Empty list fingerprint errors - Handles gracefully without crashing
3. Proper multiprocessing execution instead of silent fallback
"""

import os
import sys
import time
import gc
import inspect
import pickle
import platform
import threading
import queue
import multiprocessing as mp
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
from collections.abc import Iterable as AbcIterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    import cloudpickle
    CLOUDPICKLE_AVAIL = True
except ImportError:
    CLOUDPICKLE_AVAIL = False
    cloudpickle = pickle

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
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.console import Console
    RICH_AVAIL = True
    console = Console()
except ImportError:
    RICH_AVAIL = False
    console = None

try:
    import torch
    GPU_AVAIL = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if GPU_AVAIL else 0
except ImportError:
    GPU_AVAIL = False
    GPU_COUNT = 0

from turboscan.hardware.config import HARDWARE
from turboscan.cache.hyper_cache import HYPER_CACHE
from turboscan.gpu.accelerator import GPU_ACCELERATOR
from turboscan.analysis.function_analyzer import FunctionAnalysis, FunctionAnalyzer
from turboscan.execution.utils import (
    _clean_for_pickle, 
    _hyperboost_cloudpickle_worker, 
    _hyperboost_worker_execute,
    patch_module_for_multiprocessing,
    patch_class_for_multiprocessing,
    prepare_for_serialization,
    _is_lru_cache_method,
)


# Mark this as the main TurboScan process to prevent auto-patching in main process
_TURBOSCAN_MAIN_PROCESS = True


def _safe_make_cache_key(func_name: str, item: Any) -> Optional[str]:
    """
    Safely create a cache key, returning None if it fails.
    Handles empty lists and other fingerprint-unfriendly items.
    """
    try:
        return HYPER_CACHE._make_key(func_name, item)
    except (ValueError, TypeError, RecursionError) as e:
        # Common errors: "cannot compute fingerprint of empty list"
        return None
    except Exception:
        return None


class HyperBoost:
    _active_threads: Set[int] = set()
    _lock = threading.RLock()
    _thread_pool: Optional[ThreadPoolExecutor] = None
    _gpu_queue: Optional[queue.Queue] = None
    _main_module_patched: bool = False
    _patched_classes: Set[type] = set()  # Track patched classes
    
    BATCH_SIZE = 32
    MIN_PARALLEL_ITEMS = 2
    PROCESS_MAX_RETRIES = 3
    PROCESS_RETRY_DELAY = 0.1
    TRANSIENT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionRefusedError, EOFError)
    FATAL_PICKLE_ERRORS = (pickle.PicklingError, TypeError, AttributeError)
    DEBUG = False
    
    # Auto-scaling: minimum chunks to maximize resource utilization
    # When work items < available workers, automatically create more chunks
    AUTO_SCALE_MIN_CHUNKS = 'auto'  # 'auto' = use cpu_count, or set to specific int

    @classmethod
    def set_debug(cls, enabled: bool = True):
        cls.DEBUG = enabled
        if enabled:
            print('🔍 HyperBoost DEBUG mode enabled - will show detailed execution info')

    @classmethod
    def _init_thread_pool(cls):
        if cls._thread_pool is None:
            cls._thread_pool = ThreadPoolExecutor(max_workers=HARDWARE.cpu_count)

    @classmethod
    def _init_gpu_queue(cls):
        if GPU_AVAIL and cls._gpu_queue is None:
            cls._gpu_queue = queue.Queue()
            for i in range(GPU_COUNT):
                cls._gpu_queue.put(i)

    @classmethod
    def _get_gpu(cls) -> Optional[int]:
        if cls._gpu_queue:
            try:
                return cls._gpu_queue.get_nowait()
            except queue.Empty:
                return None
        return None

    @classmethod
    def _release_gpu(cls, device_id: int):
        if cls._gpu_queue:
            cls._gpu_queue.put(device_id)

    @classmethod
    def _get_effective_min_chunks(cls, min_chunks: Union[int, str, None], num_items: int) -> int:
        """
        Calculate the effective minimum number of chunks to use.
        
        Args:
            min_chunks: User-specified min_chunks or None
            num_items: Number of work items
            
        Returns:
            Effective minimum number of chunks
        """
        if min_chunks is None:
            # Default: no subdivision, use one chunk per item
            return num_items
        elif min_chunks == 'auto':
            # Auto: scale to use all CPUs, at least 2x for better distribution
            return max(HARDWARE.cpu_count * 2, num_items)
        elif isinstance(min_chunks, int) and min_chunks > 0:
            # Use the larger of min_chunks or num_items to ensure we have enough chunks
            return max(min_chunks, num_items)
        else:
            return num_items

    @classmethod
    def _is_chunkable_item(cls, item: Any) -> bool:
        try:
            import numpy as np
            if isinstance(item, np.ndarray):
                return item.ndim >= 3 # Only auto-chunk 3D+ (e.g., [Batch, Height, Width])
        except ImportError:
            np = None

        if isinstance(item, (list, tuple)) and len(item) > 1:
            first = item[0]
            if isinstance(first, (int, float, complex)):
                return False
            # Check for numpy scalar types if numpy is available
            if np is not None and isinstance(first, np.number):
                return False

            if hasattr(first, '__dict__') or isinstance(first, str):
                return True

        return False

    @classmethod
    def _get_item_length(cls, item: Any) -> int:
        """Get the length/size of an item for chunking purposes."""
        try:
            import numpy as np
            if isinstance(item, np.ndarray):
                return item.shape[0] if item.ndim > 0 else 1
        except ImportError:
            pass
        
        if isinstance(item, (list, tuple)):
            return len(item)
        
        return 1

    @classmethod
    def _subdivide_items(cls, items: List[Any], target_chunks: int, quiet: bool = False) -> Tuple[List[Any], List[Tuple[int, int, int]]]:
        """
        Subdivide items into smaller chunks to maximize parallelization.
        
        Args:
            items: List of work items
            target_chunks: Target number of chunks to create
            quiet: If True, suppress debug output
            
        Returns:
            Tuple of (subdivided_items, chunk_map)
            chunk_map: List of (original_index, start_offset, end_offset) for reassembly
        """
        if not any(cls._is_chunkable_item(item) for item in items):
            return items, [(i, 0, 1) for i in range(len(items))]

        if len(items) >= target_chunks:
            return items, [(i, 0, 1) for i in range(len(items))]
        
        chunks_per_item = max(1, target_chunks // len(items))
        remainder = target_chunks % len(items)
        
        subdivided = []
        chunk_map = []
        
        for orig_idx, item in enumerate(items):
            item_chunks = chunks_per_item + (1 if orig_idx < remainder else 0)
            
            if cls._is_chunkable_item(item) and item_chunks > 1:
                item_len = cls._get_item_length(item)
                actual_chunks = min(item_chunks, item_len)
                
                if actual_chunks > 1:
                    try:
                        import numpy as np
                        if isinstance(item, np.ndarray):
                            if item.ndim == 2:
                                chunk_arrays = np.array_split(item, actual_chunks, axis=1) # Split Time, keep Freq
                            else:
                                chunk_arrays = np.array_split(item, actual_chunks)
                            for chunk_idx, chunk in enumerate(chunk_arrays):
                                subdivided.append(chunk)
                                chunk_map.append((orig_idx, chunk_idx, len(chunk_arrays)))
                            if cls.DEBUG and not quiet:
                                print(f'  [DEBUG] Split item {orig_idx} (array shape {item.shape}) into {actual_chunks} chunks')
                            continue
                    except ImportError:
                        pass
                    
                    if isinstance(item, (list, tuple)):
                        chunk_size = max(1, len(item) // actual_chunks)
                        for chunk_idx in range(actual_chunks):
                            start = chunk_idx * chunk_size
                            end = start + chunk_size if chunk_idx < actual_chunks - 1 else len(item)
                            # Preserve original type
                            chunk = item[start:end] if isinstance(item, list) else tuple(item[start:end])
                            subdivided.append(chunk)
                            chunk_map.append((orig_idx, chunk_idx, actual_chunks))
                        if cls.DEBUG and not quiet:
                            print(f'  [DEBUG] Split item {orig_idx} (length {len(item)}) into {actual_chunks} chunks')
                        continue
            
            # Item not chunkable or chunking not beneficial
            subdivided.append(item)
            chunk_map.append((orig_idx, 0, 1))
        
        if cls.DEBUG and not quiet and len(subdivided) > len(items):
            print(f'  [DEBUG] Subdivided {len(items)} items into {len(subdivided)} chunks for better CPU utilization')
        
        return subdivided, chunk_map

    @classmethod
    def _reassemble_results(cls, results: List[Any], chunk_map: List[Tuple[int, int, int]], 
                           original_count: int, quiet: bool = False) -> List[Any]:
        """
        Reassemble subdivided results back into original structure.
        
        Args:
            results: Results from subdivided work items
            chunk_map: Map from _subdivide_items showing how to reassemble
            original_count: Original number of items
            quiet: If True, suppress debug output
            
        Returns:
            List of results matching original item order
        """
        if len(results) == original_count:
            # No subdivision was done
            return results
        
        # Group results by original index
        grouped = {}
        for result, (orig_idx, chunk_idx, total_chunks) in zip(results, chunk_map):
            if orig_idx not in grouped:
                grouped[orig_idx] = [None] * total_chunks
            grouped[orig_idx][chunk_idx] = result
        
        # Reassemble each original item's results
        reassembled = []
        for orig_idx in range(original_count):
            if orig_idx in grouped:
                chunks = grouped[orig_idx]
                if len(chunks) == 1:
                    reassembled.append(chunks[0])
                else:
                    # Try to concatenate results
                    try:
                        import numpy as np
                        if all(isinstance(c, np.ndarray) for c in chunks):
                            # For 2D arrays, split was done along axis=1, so concatenate along axis=1
                            if chunks[0].ndim == 2:
                                reassembled.append(np.concatenate(chunks, axis=1))
                            else:
                                reassembled.append(np.concatenate(chunks))
                            continue
                    except ImportError:
                        pass
                    
                    # For lists
                    if all(isinstance(c, list) for c in chunks):
                        combined = []
                        for c in chunks:
                            combined.extend(c)
                        reassembled.append(combined)
                        continue
                    
                    # For other types, return as list of chunk results
                    reassembled.append(chunks)
            else:
                reassembled.append(None)
        
        if cls.DEBUG and not quiet:
            print(f'  [DEBUG] Reassembled {len(results)} chunk results into {len(reassembled)} original items')
        
        return reassembled

    @classmethod
    def _ensure_main_module_patched(cls):
        """Patch all classes in __main__ that have @lru_cache methods."""
        if cls._main_module_patched:
            return
        
        try:
            main_mod = sys.modules.get('__main__')
            if main_mod is None:
                return
            
            patched_count = patch_module_for_multiprocessing(main_mod)
            cls._main_module_patched = True
            
            if patched_count > 0 and cls.DEBUG:
                print(f'[DEBUG] Patched {patched_count} classes in __main__ for multiprocessing')
        except Exception as e:
            if cls.DEBUG:
                print(f'[DEBUG] Error patching __main__: {e}')

    @classmethod
    def _find_problematic_classes(cls, obj, path='item', seen=None) -> List[Tuple[str, type, str]]:
        """Find classes with @lru_cache methods in an object tree."""
        if seen is None:
            seen = set()
        
        problems = []
        obj_id = id(obj)
        if obj_id in seen:
            return problems
        seen.add(obj_id)
        
        try:
            if hasattr(obj, '__class__') and obj.__class__.__module__ == '__main__':
                cls_type = type(obj)
                for name in dir(cls_type):
                    try:
                        attr = getattr(cls_type, name, None)
                        if attr is not None and _is_lru_cache_method(attr):
                            problems.append((path, cls_type, name))
                    except:
                        pass
            
            # Check nested objects
            if hasattr(obj, '__dict__'):
                for attr_name, attr_val in obj.__dict__.items():
                    if not attr_name.startswith('_'):
                        problems.extend(cls._find_problematic_classes(
                            attr_val, f'{path}.{attr_name}', seen
                        ))
        except:
            pass
        
        return problems

    @classmethod
    def _deep_patch_for_serialization(cls, obj, seen=None):
        """
        Recursively find and patch all __main__ classes that have lru_cache methods.
        This is more aggressive than prepare_for_serialization.
        """
        if seen is None:
            seen = set()
        
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        
        try:
            # If this object is from __main__, patch its class
            if hasattr(obj, '__class__'):
                cls_type = type(obj)
                if cls_type.__module__ == '__main__' and cls_type not in cls._patched_classes:
                    # Check if class has lru_cache methods
                    for name in dir(cls_type):
                        try:
                            attr = getattr(cls_type, name, None)
                            if attr is not None and _is_lru_cache_method(attr):
                                # Patch the entire class
                                patch_class_for_multiprocessing(cls_type, use_instance_cache=False)
                                cls._patched_classes.add(cls_type)
                                if cls.DEBUG:
                                    print(f'  [DEBUG] Patched class {cls_type.__name__} for multiprocessing')
                                break
                        except:
                            pass
            
            # Recursively check nested objects
            if hasattr(obj, '__dict__'):
                for attr_val in obj.__dict__.values():
                    cls._deep_patch_for_serialization(attr_val, seen)
            
            # Check tuple/list elements
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    cls._deep_patch_for_serialization(item, seen)
            
            # Check dict values
            if isinstance(obj, dict):
                for val in obj.values():
                    cls._deep_patch_for_serialization(val, seen)
                    
        except Exception:
            pass

    @classmethod
    def _execute_with_processes(cls, func: Callable, todos: List[Tuple[int, Any]], 
                                 results: List[Any], items: List[Any], func_name: str, 
                                 use_cache: bool, progress, task_id, quiet: bool,
                                 analysis: FunctionAnalysis) -> Tuple[bool, Optional[str]]:
        """Execute tasks using multiprocessing with comprehensive serialization handling."""
        
        if not todos:
            return (True, None)
        
        # Ensure __main__ is patched FIRST
        cls._ensure_main_module_patched()
        
        # Determine serializer
        use_cloudpickle_worker = CLOUDPICKLE_AVAIL and (
            analysis.is_picklable is None or analysis.is_picklable
        )
        serializer = cloudpickle if use_cloudpickle_worker else pickle
        
        # Test serialization with samples
        if use_cloudpickle_worker:
            # Test function serialization
            try:
                func_bytes = serializer.dumps(func)
                if cls.DEBUG:
                    print(f"  [DEBUG] Function '{func_name}' serialized OK ({len(func_bytes)} bytes)")
            except Exception as e:
                reason = f'Function cannot be serialized: {type(e).__name__}: {str(e)[:100]}'
                if cls.DEBUG:
                    print(f'  [DEBUG] Function serialization failed: {e}')
                return (False, reason)
            
            # Test work item serialization with aggressive patching
            sample_indices = [0]
            if len(todos) > 1:
                sample_indices.append(len(todos) - 1)
            if len(todos) > 10:
                sample_indices.append(len(todos) // 2)
            
            need_cleaning = False
            
            for sample_idx in sample_indices:
                test_item = todos[sample_idx][1]
                
                # Aggressively patch any __main__ classes in the item
                cls._deep_patch_for_serialization(test_item)
                
                # Find and report any remaining problematic classes
                if cls.DEBUG and sample_idx == 0:
                    problems = cls._find_problematic_classes(test_item)
                    if problems:
                        print(f'  [DEBUG] Found {len(problems)} lru_cache methods in work items:')
                        for path, _, method in problems[:5]:
                            print(f'    - {path}.{method}')
                        if len(problems) > 5:
                            print(f'    ... and {len(problems) - 5} more')
                
                # Try serialization
                test_work = (func, test_item, None, {})
                try:
                    work_bytes = serializer.dumps(test_work)
                    if cls.DEBUG and sample_idx == 0:
                        print(f'  [DEBUG] Work item serialized OK ({len(work_bytes)} bytes)')
                except Exception as e:
                    error_str = str(e)[:200]
                    if cls.DEBUG:
                        print(f'  [DEBUG] Serialization failed: {type(e).__name__}: {error_str}')
                    
                    # Try cleaning the item
                    try:
                        cleaned_item = _clean_for_pickle(prepare_for_serialization(test_item))
                        test_work = (func, cleaned_item, None, {})
                        work_bytes = serializer.dumps(test_work)
                        need_cleaning = True
                        if cls.DEBUG:
                            print(f'  [DEBUG] Work item serialized OK after cleaning')
                    except Exception as e2:
                        reason = f'Work item serialization failed: {type(e).__name__}: {error_str}'
                        if cls.DEBUG:
                            print(f'  [DEBUG] Still failed after cleaning: {e2}')
                            print(f'  [DEBUG] Item type: {type(test_item)}')
                            if hasattr(test_item, '__class__'):
                                print(f'  [DEBUG] Item class module: {test_item.__class__.__module__}')
                        return (False, reason)
            
            if cls.DEBUG:
                cleaning_note = ' (with cleaning)' if need_cleaning else ''
                print(f'  [DEBUG] All {len(sample_indices)} samples serializable{cleaning_note}')
        else:
            need_cleaning = False
        
        # Windows + cloudpickle without loky check
        if platform.system() == 'Windows' and use_cloudpickle_worker and not LOKY_AVAIL:
            if cls.DEBUG:
                print(f'  [DEBUG] Windows + cloudpickle without loky - falling back to threads')
            return (False, 'Windows requires loky for cloudpickle workers (pip install loky)')
        
        max_workers = max(HARDWARE.cpu_count, HARDWARE.cpu_count_physical * 2)
        last_error = None
        last_error_tb = None
        use_loky_native = LOKY_AVAIL and use_cloudpickle_worker
        
        # Prepare all work items with patching
        def prepare_item(item):
            cls._deep_patch_for_serialization(item)
            if need_cleaning:
                return _clean_for_pickle(prepare_for_serialization(item))
            return prepare_for_serialization(item)
        
        prepared_todos = [(idx, prepare_item(item)) for idx, item in todos]
        
        # Pre-serialize work items
        if use_loky_native or use_cloudpickle_worker:
            try:
                work_items = [serializer.dumps((func, item, None, {})) for _, item in prepared_todos]
                worker_func = _hyperboost_cloudpickle_worker
                if cls.DEBUG:
                    print(f'  [DEBUG] Pre-serialized {len(work_items)} work items')
            except Exception as e:
                if cls.DEBUG:
                    print(f'  [DEBUG] Pre-serialization failed: {e}')
                return (False, f'Pre-serialization failed: {type(e).__name__}')
        else:
            work_items = [(func, item, None, {}) for _, item in prepared_todos]
            worker_func = _hyperboost_worker_execute
        
        if cls.DEBUG:
            backend_name = 'loky' if use_loky_native else 'ProcessPoolExecutor'
            print(f'  [DEBUG] Using {backend_name} with {max_workers} workers')
        
        # Execute with retries
        for attempt in range(cls.PROCESS_MAX_RETRIES):
            try:
                if use_loky_native:
                    pool = get_reusable_executor(
                        max_workers=max_workers,
                        timeout=300,
                        kill_workers=True,
                    )
                    if cls.DEBUG:
                        print(f'  [DEBUG] Attempt {attempt + 1}: submitting {len(work_items)} items via loky')
                    
                    future_objs = [pool.submit(worker_func, item) for item in work_items]
                    completed = [None] * len(future_objs)
                    
                    for future in as_completed(future_objs):
                        fut_idx = future_objs.index(future)
                        completed[fut_idx] = future.result()
                    
                    futures = completed
                    if cls.DEBUG:
                        print(f'  [DEBUG] Got {len(futures)} results')
                    
                    # Process results
                    for i, raw_result in enumerate(futures):
                        idx = todos[i][0]
                        result = serializer.loads(raw_result)
                        
                        if isinstance(result, tuple) and len(result) == 2:
                            status, data = result
                            if status == 'success':
                                results[idx] = data
                                if use_cache:
                                    # SAFE cache key creation
                                    key = _safe_make_cache_key(func_name, items[idx])
                                    if key is not None:
                                        HYPER_CACHE.set(key, data)
                            elif status == 'error':
                                error_msg = f"{data['type']}: {data['message']}"
                                if cls.DEBUG:
                                    print(f'  [DEBUG] Worker error at item {idx}:')
                                    print(f"  {data['traceback']}")
                                raise RuntimeError(error_msg)
                        else:
                            results[idx] = result
                        
                        if progress:
                            progress.advance(task_id)
                    
                    if cls.DEBUG:
                        print(f'  [DEBUG] ✓ Multiprocessing completed successfully!')
                    return (True, None)
                
                else:
                    # Standard ProcessPoolExecutor path
                    if platform.system() == 'Windows':
                        ctx = mp.get_context('spawn')
                    else:
                        try:
                            ctx = mp.get_context('fork')
                        except ValueError:
                            ctx = mp.get_context('spawn')
                    
                    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
                        if cls.DEBUG:
                            print(f'  [DEBUG] Attempt {attempt + 1}: submitting {len(work_items)} items via ProcessPoolExecutor')
                        
                        future_objs = [pool.submit(worker_func, item) for item in work_items]
                        completed = [None] * len(future_objs)
                        
                        for future in as_completed(future_objs):
                            fut_idx = future_objs.index(future)
                            try:
                                completed[fut_idx] = future.result()
                            except Exception as e:
                                last_error = e
                                last_error_tb = traceback.format_exc()
                                raise
                        
                        # Process results
                        for i, raw_result in enumerate(completed):
                            idx = todos[i][0]
                            if use_cloudpickle_worker:
                                result = serializer.loads(raw_result)
                            else:
                                result = raw_result
                            
                            if isinstance(result, tuple) and len(result) == 2:
                                status, data = result
                                if status == 'success':
                                    results[idx] = data
                                    if use_cache:
                                        key = _safe_make_cache_key(func_name, items[idx])
                                        if key is not None:
                                            HYPER_CACHE.set(key, data)
                                elif status == 'error':
                                    error_msg = f"{data['type']}: {data['message']}"
                                    raise RuntimeError(error_msg)
                            else:
                                results[idx] = result
                            
                            if progress:
                                progress.advance(task_id)
                        
                        if cls.DEBUG:
                            print(f'  [DEBUG] ✓ Multiprocessing completed successfully!')
                        return (True, None)
            
            except cls.TRANSIENT_ERRORS as e:
                last_error = e
                last_error_tb = traceback.format_exc()
                if cls.DEBUG:
                    print(f'  [DEBUG] Transient error on attempt {attempt + 1}: {e}')
                if attempt < cls.PROCESS_MAX_RETRIES - 1:
                    time.sleep(cls.PROCESS_RETRY_DELAY * (attempt + 1))
                    continue
                break
            
            except Exception as e:
                last_error = e
                last_error_tb = traceback.format_exc()
                if cls.DEBUG:
                    print(f'  [DEBUG] Error on attempt {attempt + 1}: {type(e).__name__}: {e}')
                break
        
        reason = f'Process pool failed: {type(last_error).__name__}: {str(last_error)[:100]}'
        return (False, reason)

    @classmethod
    def run(cls, task: Union[Callable, List[Callable]], data: Iterable[Any], quiet: bool = False, 
            backend: str = 'auto', use_cache: bool = True, batch_size: int = None, 
            force_processes: bool = False, min_chunks: Union[int, str] = None) -> List[Any]:
        """
        Run a task on data items in parallel.
        
        Args:
            task: Function or list of functions to apply to each item
            data: Iterable of items to process
            quiet: If True, suppress progress output
            backend: Execution backend ('auto', 'threads', 'processes', 'gpu', 'ray')
            use_cache: If True, cache results for repeated calls
            batch_size: Size of batches for processing
            force_processes: If True, prefer multiprocessing over threads
            min_chunks: Minimum number of parallel chunks to create.
                       - None: use default behavior (one chunk per item)
                       - 'auto': automatically scale to use all available CPUs
                       - int: create at least this many chunks by subdividing work items
                       
        FIXES:
        - Handles empty data gracefully
        - Safe cache key creation for items with empty lists
        - Aggressive patching of @lru_cache classes
        - Auto-scaling: when few items exist, automatically subdivides for better CPU utilization
        """
        items = list(data)
        
        # Handle empty input
        if not items:
            return []
        
        # Prevent recursive parallelization
        thread_id = threading.get_ident()
        if thread_id in cls._active_threads:
            if isinstance(task, list):
                current = items
                for t in task:
                    current = [t(x) for x in current]
                return current
            return [task(x) for x in items]
        
        with cls._lock:
            cls._active_threads.add(thread_id)
        
        try:
            # Handle task chaining
            if isinstance(task, list):
                current_data = items
                for func in task:
                    current_data = cls.run(func, current_data, quiet=quiet, backend=backend, 
                                          use_cache=use_cache, force_processes=force_processes)
                return current_data
            
            cls._init_thread_pool()
            cls._init_gpu_queue()
            
            func = task
            func_name = getattr(func, '__name__', 'task')
            min_batches = HARDWARE.cpu_count * 4
            batch_size = batch_size or max(1, min(cls.BATCH_SIZE, len(items) // min_batches))
            
            # Analyze function
            analysis = FunctionAnalyzer.analyze(func)
            if cls.DEBUG:
                print(f"\n[DEBUG] Function analysis for '{func_name}':")
                print(f'  is_picklable: {analysis.is_picklable}')
                print(f'  uses_threads: {analysis.uses_threads}')
                print(f'  uses_multiprocessing: {analysis.uses_multiprocessing}')
                print(f'  prefers_processes: {analysis.prefers_processes}')
                print(f'  should_force_threads: {analysis.should_force_threads}')
                print(f'  failure_reason: {analysis.failure_reason}')
            
            if not quiet and analysis.uses_threads:
                print(f"\n🧵 Detected internal threading in '{func_name}' - multiprocessing will give each call its own GIL!")
            
            # Auto-scaling: subdivide items if min_chunks is specified and we have fewer items than desired
            original_items = items
            chunk_map = None
            effective_min_chunks = cls._get_effective_min_chunks(min_chunks, len(items))
            
            if min_chunks is not None and len(items) < effective_min_chunks:
                # Check if items can be subdivided
                can_subdivide = any(cls._is_chunkable_item(item) for item in items)
                
                if can_subdivide:
                    if cls.DEBUG and not quiet:
                        print(f'[DEBUG] Auto-scaling: {len(items)} items < {effective_min_chunks} target chunks')
                        print(f'[DEBUG] Attempting to subdivide items for better CPU utilization...')
                    
                    items, chunk_map = cls._subdivide_items(items, effective_min_chunks, quiet)
                    
                    if not quiet and len(items) > len(original_items):
                        print(f"🚀 Auto-scaled: {len(original_items)} items → {len(items)} chunks (using {min(len(items), HARDWARE.cpu_count)} CPUs)")
            
            # Prepare results array
            results = [None] * len(items)
            todos = []
            
            # Check cache - with SAFE key creation
            if use_cache:
                for i, item in enumerate(items):
                    try:
                        key = _safe_make_cache_key(func_name, item)
                        if key is not None:
                            found, cached = HYPER_CACHE.get(key)
                            if found:
                                results[i] = cached
                            else:
                                todos.append((i, item))
                        else:
                            # Can't create cache key, skip caching for this item
                            todos.append((i, item))
                    except Exception:
                        todos.append((i, item))
            else:
                todos = [(i, x) for i, x in enumerate(items)]
            
            if not todos:
                return results
            
            # Setup progress display
            progress = None
            task_id = None
            show_progress = RICH_AVAIL and not quiet and len(todos) > 5
            if show_progress:
                progress = Progress(
                    SpinnerColumn(), BarColumn(), 
                    TextColumn('[progress.percentage]{task.percentage:>3.0f}%'), 
                    TextColumn('{task.description}'), 
                    TimeRemainingColumn(), 
                    console=console
                )
                progress.start()
                task_id = progress.add_task(f'[cyan]{func_name}', total=len(items), completed=len(items) - len(todos))
            
            # Determine backend
            selected_backend = backend
            if backend == 'auto':
                if force_processes and analysis.is_picklable:
                    backend = 'processes'
                    if cls.DEBUG:
                        print(f'[DEBUG] Backend: processes (force_processes=True)')
                elif analysis.should_force_threads:
                    backend = 'threads'
                    if cls.DEBUG:
                        print(f'[DEBUG] Backend: threads (should_force_threads=True)')
                    if not quiet:
                        reason = analysis.failure_reason or 'unpicklable function'
                        print(f"\n⚠️  '{func_name}' requires threads: {reason}")
                elif GPU_AVAIL and len(todos) > 100:
                    backend = 'gpu'
                    if cls.DEBUG:
                        print(f'[DEBUG] Backend: gpu (GPU available, large workload)')
                elif analysis.prefers_processes:
                    backend = 'processes'
                    if cls.DEBUG:
                        print(f'[DEBUG] Backend: processes (prefers_processes=True)')
                    if not quiet and analysis.uses_threads:
                        print(f'   → Using separate processes for true parallelism')
                else:
                    backend = 'processes'
                    if cls.DEBUG:
                        print(f'[DEBUG] Backend: processes (default)')
            elif cls.DEBUG:
                print(f'[DEBUG] Backend: {backend} (explicitly specified)')
            
            success = False
            fallback_reason = None
            
            # Try Ray for large workloads
            if RAY_AVAIL and backend in ['auto', 'ray', 'gpu'] and len(todos) > 100:
                try:
                    if not ray.is_initialized():
                        ray.init(ignore_reinit_error=True, logging_level='ERROR', 
                                log_to_driver=False, include_dashboard=False, 
                                num_cpus=HARDWARE.cpu_count, num_gpus=GPU_COUNT)
                    
                    @ray.remote(num_gpus=1 if GPU_AVAIL else 0, num_cpus=1)
                    def ray_worker(f, item):
                        return f(item)
                    
                    func_ref = ray.put(func)
                    futures = [ray_worker.remote(func_ref, item) for _, item in todos]
                    for i, result in enumerate(ray.get(futures)):
                        idx = todos[i][0]
                        results[idx] = result
                        if use_cache:
                            key = _safe_make_cache_key(func_name, items[idx])
                            if key is not None:
                                HYPER_CACHE.set(key, result)
                        if progress:
                            progress.advance(task_id)
                    success = True
                except Exception as e:
                    if not quiet:
                        print(f'\n⚠️  Ray failed ({type(e).__name__}), trying process pool...')
            
            # Try GPU acceleration
            if not success and backend in ['auto', 'gpu'] and GPU_AVAIL:
                try:
                    completed = 0
                    for batch_start in range(0, len(todos), batch_size):
                        batch = todos[batch_start:batch_start + batch_size]
                        with GPU_ACCELERATOR.stream_context(batch_start // batch_size):
                            for idx, item in batch:
                                device_id = cls._get_gpu()
                                if device_id is not None:
                                    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
                                try:
                                    result = func(item)
                                    results[idx] = result
                                    if use_cache:
                                        key = _safe_make_cache_key(func_name, items[idx])
                                        if key is not None:
                                            HYPER_CACHE.set(key, result)
                                finally:
                                    if device_id is not None:
                                        cls._release_gpu(device_id)
                                if progress:
                                    progress.advance(task_id)
                    success = True
                except Exception:
                    pass
            
            # Try multiprocessing
            if not success and backend in ['auto', 'processes']:
                success, fallback_reason = cls._execute_with_processes(
                    func=func, todos=todos, results=results, items=items, 
                    func_name=func_name, use_cache=use_cache, progress=progress, 
                    task_id=task_id, quiet=quiet, analysis=analysis
                )
                if not success and not quiet and fallback_reason:
                    print(f'\n⚠️  Multiprocessing unavailable: {fallback_reason}')
                    print(f'   → Falling back to threads (still parallel, but shared GIL)')
            
            # Fallback to threads
            if not success:
                def thread_worker(args):
                    idx, item = args
                    try:
                        result = func(item)
                        if use_cache:
                            key = _safe_make_cache_key(func_name, items[idx])
                            if key is not None:
                                HYPER_CACHE.set(key, result)
                        return (idx, result, None)
                    except Exception as e:
                        return (idx, None, e)
                
                future_objs = [cls._thread_pool.submit(thread_worker, todo) for todo in todos]
                for future in as_completed(future_objs):
                    idx, result, error = future.result()
                    if error:
                        raise error
                    results[idx] = result
                    if progress:
                        progress.advance(task_id)
            
            if progress:
                progress.stop()
            
            # Reassemble results if items were subdivided
            if chunk_map is not None and len(items) > len(original_items):
                results = cls._reassemble_results(results, chunk_map, len(original_items), quiet)
            
            return results
        
        finally:
            with cls._lock:
                cls._active_threads.discard(thread_id)

    @classmethod
    def boost_all(cls, func_name: str, tasks_iter: Iterable[Callable], 
                  parallel: bool = True, backend: str = 'auto') -> List[Any]:
        """Execute multiple independent tasks in parallel."""
        
        # Ensure __main__ is patched
        cls._ensure_main_module_patched()
        
        tasks = list(tasks_iter)
        if not tasks:
            return []
        
        if not parallel or len(tasks) < cls.MIN_PARALLEL_ITEMS:
            return [task() for task in tasks]
        
        def execute_task(task):
            return task()
        
        return cls.run(execute_task, tasks, quiet=True, backend=backend)

    @classmethod
    def map(cls, func: Callable, *iterables, **kwargs) -> List[Any]:
        """Parallel map operation."""
        items = list(zip(*iterables))
        
        def wrapper(args):
            return func(*args)
        
        return cls.run(wrapper, items, **kwargs)

    @classmethod
    def starmap(cls, func: Callable, iterable: Iterable, **kwargs) -> List[Any]:
        """Parallel starmap operation."""
        items = list(iterable)
        
        def wrapper(args):
            return func(*args)
        
        return cls.run(wrapper, items, **kwargs)

    @classmethod 
    def shutdown(cls):
        """Shutdown thread pool and cleanup."""
        if cls._thread_pool:
            cls._thread_pool.shutdown(wait=False)
            cls._thread_pool = None
        gc.collect()


# Module-level convenience functions
def boost_run(task, data, **kwargs):
    return HyperBoost.run(task, data, **kwargs)

def boost_map(func, *iterables, **kwargs):
    return HyperBoost.map(func, *iterables, **kwargs)

def boost_all(func_name, tasks, **kwargs):
    return HyperBoost.boost_all(func_name, tasks, **kwargs)


# Pre-warm thread pool and GPU queue on import for faster first execution
def _prewarm_pools():
    """
    Pre-initialize thread pool and GPU queue to avoid cold start overhead.
    Only runs when not in a test environment (pytest, unittest, etc.) or subprocess.
    """
    try:
        HyperBoost._init_thread_pool()
        HyperBoost._init_gpu_queue()
    except Exception:
        # Silently ignore initialization errors
        pass

# Only pre-warm if not in a test environment (to avoid issues with testing)
# This checks for pytest specifically, but could be extended for other test runners if needed
if 'pytest' not in sys.modules and not os.environ.get('_TURBOSCAN_SUBPROCESS'):
    _prewarm_pools()