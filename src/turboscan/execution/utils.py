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


# Track which classes have been patched to avoid double-patching
# Use weakref.WeakSet to allow garbage collection and prevent id() reuse issues
_PATCHED_CLASSES: weakref.WeakSet = weakref.WeakSet()
_ORIGINAL_METHODS: Dict[
    Tuple[int, str], Any
] = {}  # (id(cls), method_name) -> original method
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

    # Check for the lru_cache wrapper type
    try:
        # Direct type check
        if type(obj).__name__ == "_lru_cache_wrapper":
            return True

        # Check for cache_info and cache_clear (unique to lru_cache)
        has_cache_info = hasattr(obj, "cache_info") and callable(
            getattr(obj, "cache_info", None)
        )
        has_cache_clear = hasattr(obj, "cache_clear") and callable(
            getattr(obj, "cache_clear", None)
        )
        has_wrapped = hasattr(obj, "__wrapped__")

        if has_cache_info and has_cache_clear and has_wrapped:
            return True

        # Check for functools.cache (Python 3.9+)
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
        # Get all attributes once instead of repeatedly calling dir
        for name in dir(cls):
            if name.startswith("__") and name.endswith("__"):
                continue  # Skip dunder methods

            try:
                # Get the attribute from the class (not instance)
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
    # Get the original unwrapped function
    unwrapped = original_func

    # Follow the __wrapped__ chain
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__

    # Create a new function that calls the unwrapped version
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
            return 0  # Already patched

        lru_methods = _find_lru_cache_methods(cls)

        if not lru_methods:
            _PATCHED_CLASSES.add(cls)
            return 0

        patched_count = 0
        cls_id = id(cls)

        for method_name, lru_wrapper in lru_methods.items():
            try:
                # Store original for potential restoration
                _ORIGINAL_METHODS[(cls_id, method_name)] = lru_wrapper

                # Create unwrapped version
                simple_method = _create_simple_method(lru_wrapper)

                # Replace on the class
                setattr(cls, method_name, simple_method)
                patched_count += 1

            except Exception:
                # If we fail to patch one method, continue with others
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

                # Check if it's a class defined in this module
                if isinstance(obj, type):
                    # Only patch classes from __main__ or the target module
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
        # Get the object's class
        if hasattr(obj, "__class__"):
            cls = type(obj)
            if cls.__module__ == "__main__":
                classes.add(cls)

        # Check object's __dict__
        if hasattr(obj, "__dict__"):
            for attr_val in obj.__dict__.values():
                classes.update(_get_all_referenced_classes(attr_val, seen))

        # Check sequences
        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                classes.update(_get_all_referenced_classes(item, seen))

        # Check dicts
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
        # Find all __main__ classes referenced by this object
        classes = _get_all_referenced_classes(obj)

        # Patch each class
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

    # First, prepare for serialization (patch classes)
    prepare_for_serialization(obj)

    try:
        # Try to pickle as-is
        pickle.dumps(obj)
        return obj
    except Exception:
        pass

    # Handle common unpicklable types
    if isinstance(obj, (type(lambda: None), type(print))):
        try:
            cloudpickle.dumps(obj)
            return obj
        except Exception:
            # Return original unchanged - serialization will fail but thread fallback
            # will work correctly since the original lambda is preserved
            return obj

    # For objects with __dict__, try to clean attributes by creating a copy
    # This applies to objects from any module to ensure consistent behavior
    if hasattr(obj, "__dict__") and hasattr(obj, "__class__"):
        try:
            cls = type(obj)
            # Create a new instance without calling __init__
            # This preserves the original object intact for thread fallback
            try:
                cleaned_obj = object.__new__(cls)
            except TypeError:
                # Some classes have custom __new__ that requires arguments,
                # or inherit from types that don't support object.__new__
                # In this case, return the original object unmodified
                # to avoid corrupting it - serialization will fail but fallback works
                return obj

            # Copy and clean attributes to the new object
            for key, val in list(obj.__dict__.items()):
                try:
                    pickle.dumps(val)
                    cleaned_obj.__dict__[key] = val
                except Exception:
                    try:
                        cloudpickle.dumps(val)
                        cleaned_obj.__dict__[key] = val
                    except Exception:
                        # Skip unpicklable attributes - they are omitted from the copy
                        # The original object retains all its attributes
                        pass
            return cleaned_obj
        except Exception:
            # If we can't create a copy, return original unmodified
            return obj

    # Handle sequences by creating new containers with cleaned elements
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
        # If writing fails, close/delete and re-raise
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
        # Deserialize the work item
        func, item, _key, kwargs = cloudpickle.loads(serialized_work)

        # Execute the function
        result = func(item, **kwargs) if kwargs else func(item)

        # Serialize result to check size
        success_payload = ("success", result)
        blob = cloudpickle.dumps(success_payload)

        # Check size - if too big, offload the RESULT (not the tuple)
        if len(blob) > IPC_OFFLOAD_THRESHOLD:
            try:
                path = _offload_large_result(result, cloudpickle)
                # Return a special status indicating offload
                return cloudpickle.dumps(("offloaded", path))
            except Exception:
                # If offload fails, fall back to returning the blob (might deadlock, but we tried)
                return blob

        return blob

    except Exception as e:
        # Return error info
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
            # We use the available pickle module
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


# Initialize by patching __main__ on import in child processes
def _auto_patch_main() -> None:
    """Auto-patch __main__ when running in a worker process."""
    # Only run in child processes, not the main TurboScan process
    if hasattr(
        sys.modules.get("turboscan.execution.hyper_boost", None),
        "_TURBOSCAN_MAIN_PROCESS",
    ):
        return

    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        patch_module_for_multiprocessing(main_mod)


# Don't auto-patch on import - let HyperBoost control when patching happens
# _auto_patch_main()
