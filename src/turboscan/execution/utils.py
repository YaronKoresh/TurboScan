"""
TurboScan Execution Utilities
"""

import functools
import os
import pickle
import sys
import tempfile
import threading
import traceback
import weakref
from typing import Any, Callable, Dict, Optional, Set, Tuple

try:
    import cloudpickle

    CLOUDPICKLE_AVAIL = True
except ImportError:
    CLOUDPICKLE_AVAIL = False
    cloudpickle = pickle


_PATCHED_CLASSES: weakref.WeakSet = weakref.WeakSet()
_ORIGINAL_METHODS: Dict[Tuple[int, str], Any] = {}
_PATCH_LOCK = threading.Lock()
IPC_OFFLOAD_THRESHOLD = 10 * 1024 * 1024


def _is_lru_cache_method(obj) -> bool:
    """
    Check if an object is an lru_cache wrapped method.

    Detection methods:
    1. Has __wrapped__ attribute (functools wrapper pattern)
    2. Has cache_info and cache_clear methods (lru_cache signature)
    3. Is instance of functools._lru_cache_wrapper
    """
    if obj is None:
        return False

    try:
        if type(obj).__name__ == "_lru_cache_wrapper":
            return True

        has_cache_info = hasattr(obj, "cache_info") and callable(
            getattr(obj, "cache_info", None)
        )
        has_cache_clear = hasattr(obj, "cache_clear") and callable(
            getattr(obj, "cache_clear", None)
        )
        has_wrapped = hasattr(obj, "__wrapped__")

        if has_cache_info and has_cache_clear and has_wrapped:
            return True

        if has_wrapped and has_cache_info:
            return True

    except Exception:
        pass

    return False


def _find_lru_cache_methods(cls: type) -> Dict[str, Any]:
    """
    Find all methods in a class that are decorated with @lru_cache.
    Returns dict of {method_name: lru_cache_wrapper}
    """
    lru_methods = {}

    try:
        for name in dir(cls):
            if name.startswith("__") and name.endswith("__"):
                continue

            try:
                attr = getattr(cls, name, None)
                if attr is not None and _is_lru_cache_method(attr):
                    lru_methods[name] = attr
            except Exception:
                pass
    except Exception:
        pass

    return lru_methods


def _create_simple_method(original_func: Callable) -> Callable:
    """
    Create a simple method from an lru_cache wrapped function.
    This extracts the original function and creates a new method without caching.
    """

    unwrapped = original_func

    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__

    @functools.wraps(unwrapped)
    def simple_method(*args, **kwargs):
        return unwrapped(*args, **kwargs)

    return simple_method


def patch_class_for_multiprocessing(
    cls: type, use_instance_cache: bool = False
) -> int:
    """
    Patch a class to remove @lru_cache decorators from its methods.
    This makes the class picklable for multiprocessing.

    Args:
        cls: The class to patch
        use_instance_cache: If True, create per-instance cache (not supported yet)

    Returns:
        Number of methods patched
    """
    with _PATCH_LOCK:
        if cls in _PATCHED_CLASSES:
            return 0

        lru_methods = _find_lru_cache_methods(cls)

        if not lru_methods:
            _PATCHED_CLASSES.add(cls)
            return 0

        patched_count = 0
        cls_id = id(cls)

        for method_name, lru_wrapper in lru_methods.items():
            try:
                _ORIGINAL_METHODS[(cls_id, method_name)] = lru_wrapper

                simple_method = _create_simple_method(lru_wrapper)

                setattr(cls, method_name, simple_method)
                patched_count += 1

            except Exception:
                pass

        _PATCHED_CLASSES.add(cls)
        return patched_count


def patch_module_for_multiprocessing(module) -> int:
    """
    Patch all classes in a module that have @lru_cache methods.

    Args:
        module: The module to patch (typically __main__)

    Returns:
        Total number of classes patched
    """
    patched_classes = 0

    try:
        for name in dir(module):
            try:
                obj = getattr(module, name, None)
                if obj is None:
                    continue

                if isinstance(obj, type):
                    if getattr(obj, "__module__", None) == getattr(
                        module, "__name__", None
                    ):
                        methods_patched = patch_class_for_multiprocessing(obj)
                        if methods_patched > 0:
                            patched_classes += 1
            except Exception:
                pass
    except Exception:
        pass

    return patched_classes


def _get_all_referenced_classes(
    obj, seen: Optional[Set[int]] = None
) -> Set[type]:
    """
    Recursively find all classes referenced by an object.
    """
    if seen is None:
        seen = set()

    classes = set()
    obj_id = id(obj)

    if obj_id in seen:
        return classes
    seen.add(obj_id)

    try:
        if hasattr(obj, "__class__"):
            cls = type(obj)
            if cls.__module__ == "__main__":
                classes.add(cls)

        if hasattr(obj, "__dict__"):
            for attr_val in obj.__dict__.values():
                classes.update(_get_all_referenced_classes(attr_val, seen))

        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                classes.update(_get_all_referenced_classes(item, seen))

        if isinstance(obj, dict):
            for val in obj.values():
                classes.update(_get_all_referenced_classes(val, seen))

    except Exception:
        pass

    return classes


def prepare_for_serialization(obj: Any) -> Any:
    """
    Prepare an object for serialization by patching any __main__ classes
    it references that have @lru_cache methods.

    This is the key function that should be called before serializing work items.
    """
    try:
        classes = _get_all_referenced_classes(obj)

        for cls in classes:
            patch_class_for_multiprocessing(cls)
    except Exception:
        pass

    return obj


def _clean_for_pickle(obj: Any, seen: Optional[Set[int]] = None) -> Any:
    """
    Clean an object for pickle serialization.

    Creates COPIES of objects that need cleaning, preserving the original
    objects intact for thread fallback scenarios. Unpicklable attributes
    are omitted from the copy rather than replaced with None.

    Args:
        obj: The object to clean for serialization
        seen: Set of object IDs already processed (for cycle detection)

    Returns:
        Either the original object if already picklable, or a cleaned copy
        with unpicklable attributes omitted.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return obj
    seen.add(obj_id)

    prepare_for_serialization(obj)

    try:
        pickle.dumps(obj)
        return obj
    except Exception:
        pass

    if isinstance(obj, (type(lambda: None), type(print))):
        try:
            cloudpickle.dumps(obj)
            return obj
        except Exception:
            return obj

    if hasattr(obj, "__dict__") and hasattr(obj, "__class__"):
        try:
            cls = type(obj)

            try:
                cleaned_obj = object.__new__(cls)
            except TypeError:
                return obj

            for key, val in list(obj.__dict__.items()):
                try:
                    pickle.dumps(val)
                    cleaned_obj.__dict__[key] = val
                except Exception:
                    try:
                        cloudpickle.dumps(val)
                        cleaned_obj.__dict__[key] = val
                    except Exception:
                        pass
            return cleaned_obj
        except Exception:
            return obj

    if isinstance(obj, list):
        return [_clean_for_pickle(item, seen) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_clean_for_pickle(item, seen) for item in obj)
    if isinstance(obj, dict):
        return {k: _clean_for_pickle(v, seen) for k, v in obj.items()}

    return obj


def _offload_large_result(data: Any, serializer) -> str:
    """
    Save large data to a temporary file and return the path.
    """
    fd, path = tempfile.mkstemp(prefix="turbo_ipc_", suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            serializer.dump(data, f)
        return path
    except Exception:
        try:
            os.close(fd)
            os.remove(path)
        except OSError:
            pass
        raise


def _hyperboost_cloudpickle_worker(serialized_work: bytes) -> bytes:
    """
    Worker function for cloudpickle-based multiprocessing.

    Args:
        serialized_work: Cloudpickle-serialized tuple of (func, item, key, kwargs)

    Returns:
        Cloudpickle-serialized result tuple: ('success', result) or ('error', error_info)
    """
    try:
        func, item, _key, kwargs = cloudpickle.loads(serialized_work)

        result = func(item, **kwargs) if kwargs else func(item)

        success_payload = ("success", result)
        blob = cloudpickle.dumps(success_payload)

        if len(blob) > IPC_OFFLOAD_THRESHOLD:
            try:
                path = _offload_large_result(result, cloudpickle)

                return cloudpickle.dumps(("offloaded", path))
            except Exception:
                return blob

        return blob

    except Exception as e:
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        return cloudpickle.dumps(("error", error_info))


def _hyperboost_worker_execute(work: Tuple) -> Tuple:
    """
    Worker function for standard pickle-based multiprocessing.

    Args:
        work: Tuple of (func, item, key, kwargs)

    Returns:
        Result tuple: ('success', result) or ('error', error_info)
    """
    try:
        func, item, _key, kwargs = work
        result = func(item, **kwargs) if kwargs else func(item)

        try:
            serializer = pickle
            blob = serializer.dumps(result)

            if len(blob) > IPC_OFFLOAD_THRESHOLD:
                path = _offload_large_result(result, serializer)
                return ("offloaded", path)

        except Exception:
            pass

        return ("success", result)

    except Exception as e:
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        return ("error", error_info)


def restore_class_methods(cls: type) -> int:
    """
    Restore original @lru_cache methods to a class.

    Args:
        cls: The class to restore

    Returns:
        Number of methods restored
    """
    cls_id = id(cls)
    restored = 0

    with _PATCH_LOCK:
        keys_to_remove = []

        for (stored_cls_id, method_name), original in _ORIGINAL_METHODS.items():
            if stored_cls_id == cls_id:
                try:
                    setattr(cls, method_name, original)
                    restored += 1
                    keys_to_remove.append((stored_cls_id, method_name))
                except Exception:
                    pass

        for key in keys_to_remove:
            del _ORIGINAL_METHODS[key]

        _PATCHED_CLASSES.discard(cls)

    return restored


def is_class_patched(cls: type) -> bool:
    """Check if a class has been patched for multiprocessing."""
    return cls in _PATCHED_CLASSES


def get_patched_class_count() -> int:
    """Get the number of classes that have been patched."""
    return len(_PATCHED_CLASSES)


def _auto_patch_main() -> None:
    """Auto-patch __main__ when running in a worker process."""

    if hasattr(
        sys.modules.get("turboscan.execution.hyper_boost", None),
        "_TURBOSCAN_MAIN_PROCESS",
    ):
        return

    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        patch_module_for_multiprocessing(main_mod)
