"""
Parallel execution utilities for SGPyKit.

Provides context-based parallelization for expensive function evaluations
on sparse grids.
"""

import os
import pickle
import numpy as np
import multiprocessing
import concurrent.futures
from typing import Callable, Any

from sgpykit.util.log import logger


class ParallelContext:
    """
    Context manager for parallel function evaluation.

    Enables selective parallelization of expensive operations with
    configurable thresholds and execution strategies.

    Parameters
    ----------
    min_workload_size : int, optional
        Minimum number of items required to use parallelization.
        Default: None (uses environment variable or default of 10)
    max_workers : int, optional
        Maximum number of parallel workers.
        Default: None (uses the number of available CPUs, i.e.
        ``len(os.sched_getaffinity(0))`` on Linux / ``os.cpu_count()``
        elsewhere)
    backend : str, optional
        Parallelization backend: 'process' (multiprocessing) or 'thread' (threading).
        Default: 'process'
    parallel_condition : Callable, optional
        Lambda function for custom parallelization conditions.
        Signature: parallel_condition(num_points: int, points: np.ndarray) -> bool
        If provided, this overrides the default threshold check.
        Default: None (use min_workload_size threshold)
    min_items_per_worker : int, optional
        Minimum number of items assigned to each worker per chunk.
        Default: 1 (one item per worker)
        Higher values reduce parallelism overhead by batching items.
        Note: this is a lower bound — chunk sizes may be larger, and fewer
        chunks/workers may be created if the workload does not divide evenly.
    enabled : bool, optional
        Whether parallelization is active. Default: True

    Fallback behavior
    -----------------
    ``execute()`` falls back to serial execution only when the parallel
    *infrastructure* fails (pool creation, pickling of func/args, broken
    pipes). In that case a warning is logged via the ``sgpykit`` logger and
    the computation is redone serially — so ``f`` may be evaluated twice
    (relevant for side-effecting functions). Errors raised by ``f`` itself
    inside a chunk are always propagated to the caller.

    Examples
    --------
    >>> # Basic usage with threshold
    >>> with ParallelContext(min_workload_size=20, max_workers=4) as ctx:
    ...     f_eval, *_ = evaluate_on_sparse_grid(f, S, Sr, parallel_ctx=ctx)
    >>>
    >>> # Custom parallelization condition
    >>> with ParallelContext(
    ...     parallel_condition=lambda n, pts: n > 50 or (n > 20 and pts.shape[0] > 3)
    ... ) as ctx:
    ...     f_eval, *_ = evaluate_on_sparse_grid(f, S, Sr, parallel_ctx=ctx)
    >>>
    >>> # Control chunk size for batching
    >>> with ParallelContext(min_workload_size=100, min_items_per_worker=10) as ctx:
    ...     f_eval, *_ = evaluate_on_sparse_grid(f, S, Sr, parallel_ctx=ctx)

    Backend selection guide
    -----------------------
    Choose 'process' (default) or 'thread' based on the nature of f:

    - 'process': f is pure-Python CPU-bound. Separate interpreters bypass the
      GIL and use multiple cores, but points/results are pickled over pipes
      and f must be picklable. Best for heavy per-point computation.
    - 'thread': f is I/O-bound or high-latency (remote calls, HTTP, file/db
      access, time.sleep) or calls GIL-releasing C code (NumPy/BLAS kernels).
      Workers share memory (zero-copy, no picklability requirement) and
      startup is ~us instead of ~100ms for a process pool. Best for expensive
      remote evaluations and large point matrices where pickling dominates.

    Note: for pure-Python CPU-bound f, threads give no speedup (GIL).
    The worker pool is created lazily on first execute() call and reused
    for the lifetime of the context, so repeated calls within one context
    do not pay process-spawn overhead.
    """

    def __init__(
        self,
        min_workload_size=None,
        max_workers=None,
        backend='process',
        parallel_condition=None,
        min_items_per_worker=1,
        enabled=True
    ):
        # Validate explicitly provided parameters first
        if min_workload_size is not None and min_workload_size <= 0:
            raise ValueError(f"min_workload_size must be positive, got {min_workload_size}")
        if max_workers is not None and max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}")
        if min_items_per_worker <= 0:
            raise ValueError(f"min_items_per_worker must be positive, got {min_items_per_worker}")
        if backend not in ['process', 'thread']:
            raise ValueError(f"backend must be 'process' or 'thread', got {backend}")

        # Set values with fallbacks
        self.min_workload_size = min_workload_size if min_workload_size is not None else self._get_default_threshold()
        self.max_workers = max_workers if max_workers is not None else self._get_default_max_workers()
        self.backend = backend
        self.parallel_condition = parallel_condition
        self.min_items_per_worker = min_items_per_worker
        self.enabled = enabled
        # Worker pool cached for the lifetime of this context, so that
        # repeated execute() calls do not pay process-spawn overhead each time
        self._pool = None

    @staticmethod
    def _get_default_max_workers():
        """Get the number of usable CPUs.

        Uses len(os.sched_getaffinity(0)) on Linux so that CPU-affinity /
        cgroup-constrained environments (containers, HSL clusters) report
        the actually available core count rather than the host total.
        """
        try:
            return len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            return os.cpu_count() or 1

    def _get_default_threshold(self):
        """Get default threshold from environment or fallback"""
        return int(os.getenv('SGPYKIT_PARALLEL_THRESHOLD', 10))

    def should_use_parallel(self, num_points, points=None):
        """
        Determine if parallelization should be used for a given workload.

        Parameters
        ----------
        num_points : int
            Number of points to evaluate
        points : np.ndarray, optional
            Actual points array (shape: n_dimensions, n_points)
            Required when using parallel_condition lambda

        Returns
        -------
        bool
            True if parallelization should be used
        """
        if not self.enabled:
            return False

        # Use custom parallel_condition if provided
        if self.parallel_condition is not None:
            if points is None:
                raise ValueError(
                    "points parameter required when parallel_condition is provided"
                )
            return self.parallel_condition(num_points, points)

        # Use default threshold check
        return num_points >= self.min_workload_size

    def execute(self, func, points, *args, **kwargs):
        """
        Execute a function evaluation with automatic parallel/serial selection.

        Parameters
        ----------
        func : Callable
            Function to evaluate. Must accept a points array as first argument.
            Signature: func(points: np.ndarray, *args, **kwargs) -> np.ndarray
        points : np.ndarray
            Points array where each column is a point to evaluate
            Shape: (n_dimensions, n_points)
        *args, **kwargs
            Additional arguments passed to func

        Returns
        -------
        np.ndarray
            Function evaluation results
            Shape: function_output_shape (usually (1, n_points) or (n_outputs, n_points))

        Notes
        -----
        If the parallel infrastructure fails (pool creation, pickling,
        broken pipes), a warning is logged and the computation falls back to
        serial execution — meaning ``f`` may be evaluated twice (relevant for
        side-effecting functions). Errors raised by ``f`` inside a chunk are
        propagated to the caller.
        """
        num_points = points.shape[1]

        # Check if we should use parallelization
        if not self.should_use_parallel(num_points, points):
            return func(points, *args, **kwargs)

        # Execute with appropriate backend; only infrastructure errors
        # (pool creation, pickling, broken pipes) trigger the serial
        # fallback — errors raised by func inside a chunk propagate.
        try:
            if self.backend == 'process':
                return self._execute_multiprocess(func, points, *args, **kwargs)
            else:
                return self._execute_multithread(func, points, *args, **kwargs)
        except _INFRASTRUCTURE_ERRORS as e:
            logger.warning(
                "Parallel execution failed (%s: %s), falling back to serial. "
                "Note: func may be evaluated twice.",
                type(e).__name__, e
            )
            return func(points, *args, **kwargs)

    def _execute_multiprocess(self, func, points, *args, **kwargs):
        """Execute with multiprocessing. The pool is created lazily on first
        use and reused for all subsequent execute() calls of this context.

        Note: creating the pool from a multithreaded parent process (e.g.
        Jupyter with BLAS/OpenMP threads) can deadlock with the default
        'fork' start method, since fork only replicates the calling thread.
        """
        if self._pool is None:
            self._pool = multiprocessing.Pool(self.max_workers)
        chunks = self._split_points(points, self.max_workers, self.min_items_per_worker)
        chunk_args = [(func, chunk, args, kwargs) for chunk in chunks]
        results = self._pool.starmap(self._execute_chunk, chunk_args)
        return self._combine_chunk_results(results)

    def _execute_multithread(self, func, points, *args, **kwargs):
        """Execute with threading."""
        with concurrent.futures.ThreadPoolExecutor(self.max_workers) as executor:
            chunks = self._split_points(points, self.max_workers, self.min_items_per_worker)
            chunk_args = [(func, chunk, args, kwargs) for chunk in chunks]

            # Submit all chunks and collect results in submission order,
            # so that hstack concatenates columns in the correct order
            # (as_completed() would yield chunks out of completion order).
            futures = [executor.submit(self._execute_chunk, *cargs) for cargs in chunk_args]

            # Chunk exceptions (errors raised by func) are propagated,
            # consistent with the process backend where starmap re-raises.
            results = [future.result() for future in futures]

            return self._combine_chunk_results(results)

    @staticmethod
    def _combine_chunk_results(results):
        """Concatenate per-chunk results along the point (column) axis.

        atleast_2d normalizes 1-D chunk results so hstack yields
        (n_outputs, n_points); if all chunks returned 1-D arrays the
        combined result is squeezed back to (n_points,) so the parallel
        path matches the output shape of the serial path.
        """
        if all(np.ndim(r) == 1 for r in results):
            return np.concatenate(results)
        return np.hstack([np.atleast_2d(r) for r in results])

    def _split_points(self, points, n_chunks, min_items_per_worker):
        """
        Split points into chunks for parallel processing.

        Parameters
        ----------
        points : np.ndarray
            Points array to split (shape: n_dimensions, n_points)
        n_chunks : int
            Number of chunks/workers to create
        min_items_per_worker : int
            Minimum number of items per worker per chunk

        Returns
        -------
        list
            List of point chunks
        """
        n_points = points.shape[1]

        # Avoid creating more chunks than points
        n_chunks = min(n_chunks, n_points)

        # Guarantee that every chunk receives at least min_items_per_worker
        # points: reduce the number of chunks if necessary, so no trailing
        # remainder chunk falls below the minimum.
        if min_items_per_worker > 1:
            n_chunks = max(1, min(n_chunks, n_points // min_items_per_worker))

        # Balanced chunk sizes via ceil division: the remainder is spread
        # across the chunks instead of forming a tiny trailing chunk
        # (e.g. 102 points / 4 chunks -> 26, 26, 26, 24)
        chunk_size = -(-n_points // n_chunks)  # ceil(n_points / n_chunks)
        chunks = []

        for i in range(0, n_points, chunk_size):
            chunk_end = min(i + chunk_size, n_points)
            chunks.append(points[:, i:chunk_end])

        return chunks

    @staticmethod
    def _execute_chunk(func, chunk, args, kwargs):
        """
        Execute function on a chunk of points.

        This is a static method to make it picklable for multiprocessing.
        """
        return func(chunk, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Shut down the cached worker pool, if one was created.
        # On normal exit, close() + join() lets pending work finish gracefully;
        # on error paths, terminate() kills workers immediately.
        if self._pool is not None:
            if exc_type is None:
                self._pool.close()
                self._pool.join()
            else:
                self._pool.terminate()
            self._pool = None
        return False


# Exception types that indicate a parallelization-infrastructure failure
# (as opposed to an error raised by the user-supplied function itself)
_INFRASTRUCTURE_ERRORS = (
    pickle.PicklingError,
    ImportError,
    AttributeError,   # e.g. can't pickle local functions/ lambdas
    BrokenPipeError,
    OSError,          # includes multiprocessing pool creation failures
)


# Convenience function for backward compatibility.
# Kept as a documented public factory: it provides a single, stable entry
# point for constructing contexts from configuration dictionaries (e.g.
# **kwargs loaded from user config) and for easier mocking in tests.
def create_parallel_context(
    min_workload_size=None,
    max_workers=None,
    backend='process',
    parallel_condition=None,
    min_items_per_worker=1,
    enabled=True
):
    """
    Helper function to create a ParallelContext.

    This function exists as a stable public factory: it allows constructing
    a context from a configuration mapping (``create_parallel_context(**cfg)``)
    and gives tests a single seam to patch.

    Parameters
    ----------
    min_workload_size : int, optional
        Minimum number of items required to use parallelization
    max_workers : int, optional
        Maximum number of parallel workers
    backend : str, optional
        Parallelization backend
    parallel_condition : Callable, optional
        Lambda function for custom parallelization conditions
    min_items_per_worker : int, optional
        Minimum number of items per worker per chunk
    enabled : bool, optional
        Whether parallelization is active

    Returns
    -------
    ParallelContext
        Configured parallelization context
    """
    return ParallelContext(
        min_workload_size=min_workload_size,
        max_workers=max_workers,
        backend=backend,
        parallel_condition=parallel_condition,
        min_items_per_worker=min_items_per_worker,
        enabled=enabled
    )
