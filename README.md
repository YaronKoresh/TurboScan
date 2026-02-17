# TurboScan - Maximum Performance Python Execution Engine
A massively parallelized, GPU-accelerated, JIT-optimized execution framework
that uses ALL available system resources for lightning-fast Python execution.

---

**Features:**
- Multi-GPU acceleration with intelligent load balancing
- Numba JIT injection for hot paths
- Memory-mapped file I/O
- Async parallel scanning
- Shared memory IPC with enhanced serialization (cloudpickle)
- CPU affinity optimization
- Vectorized batch processing
- Integrated audit-on-run
- Aggressive caching with bloom filters
- SIMD-style operations where possible
- Transparent handling of custom classes and dataclasses

---

**SMART MULTIPROCESSING:**
- Intelligently detects when functions use threads internally
- Functions using ThreadPoolExecutor, threading.Thread, etc. are
  PRIORITIZED for multiprocessing (each process gets its own GIL)
- Retries transient errors before falling back to threads
- Classifies errors as fatal (pickle) vs transient (pipe/connection)
- Pre-flight function analysis predicts optimal execution strategy
- Use HyperBoost.analyze_function(fn) to see why a function runs
  with processes vs threads
- force_processes=True bypasses automatic detection when you KNOW better
