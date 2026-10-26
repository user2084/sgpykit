"""
Benchmark tests for ParallelContext and evaluate_on_sparse_grid.

Compares sequential vs parallel execution using pytest-benchmark.
Sequential and parallel variants of the same workload are parametrizations
of the same test, so they appear together in the benchmark results table
(grouped by test name) without needing explicit groups:

    pytest tests/sgpykit/util/test_parallel_benchmark.py --benchmark-autosave
    pytest-benchmark compare  # shows sequential vs parallel side by side

Workloads:
- low latency  : cheap vectorized function (parallel overhead dominates)
- high latency : function with per-call delay, simulating expensive/remote
                 evaluations typical for sparse grid function evaluation
- evaluate_on_sparse_grid : the actual sgpykit entry point, sequential vs parallel

All point sets are deterministic (lattice-based), no RNG involved.
"""

import pytest
import numpy as np
import time
from sgpykit import (
    ParallelContext,
    create_sparse_grid,
    reduce_sparse_grid,
    evaluate_on_sparse_grid,
    knots_uniform,
    lev2knots_lin,
)


# ==================== TEST FUNCTIONS ====================


def _low_latency_func(x):
    """Cheap vectorized computation; parallel overhead dominates."""
    return np.sum(x ** 2, axis=0)


def _high_latency_func(x):
    """Expensive evaluation simulation (e.g., high-latency/remote function).

    Deterministic busy work proportional to the number of points, so that
    latency scales with the workload (like a remote/expensive per-point call).
    Roughly 2ms of work per point, so parallelization can amortize the
    per-call IPC overhead.
    """
    n = x.shape[1]
    acc = 0.0
    for i in range(n * 40_000):  # ~2ms per point
        acc += (i % 7) ** 0.5
    result = np.sum(x ** 2, axis=0) + acc * 0.0
    return result * (1.0 + 0.1 * np.sin(result))


def _io_bound_func(x):
    """I/O-bound / high-latency evaluation simulation (releases the GIL).

    time.sleep() releases the GIL, so the thread backend runs chunks truly
    concurrently — this is the scenario where parallel-thread beats
    parallel-process (no pickling) and sequential (no concurrency at all).
    Simulates expensive remote function evaluations on a sparse grid.
    """
    time.sleep(0.002 * x.shape[1])  # 2ms per point, GIL released
    result = np.sum(x ** 2, axis=0)
    return result * (1.0 + 0.1 * np.sin(result))


def _points(num_points):
    """Deterministic lattice of points, shape (4, num_points)."""
    # Offset each dimension slightly so points differ across dimensions
    offsets = np.arange(4)[:, None] / (4.0 * num_points)
    return np.linspace(0.0, 1.0, num_points)[None, :] + offsets


# ==================== ParallelContext: LOW LATENCY ====================


@pytest.mark.parametrize("mode", ["sequential", "parallel-thread", "parallel-process"])
def test_benchmark_low_latency(benchmark, mode):
    """Sequential vs parallel (thread/process) with a cheap function."""
    points = _points(50)
    kwargs = dict(min_workload_size=5, max_workers=2)
    if mode == "sequential":
        kwargs["enabled"] = False
    else:
        kwargs["backend"] = mode.split("-")[1]

    # Warm up the worker pool outside of the timed region
    # (must exceed min_workload_size so the pool is actually created)
    with ParallelContext(**kwargs) as ctx:
        ctx.execute(_low_latency_func, points[:, :5])

        def run():
            return ctx.execute(_low_latency_func, points)

        result = benchmark(run)
    np.testing.assert_allclose(result, _low_latency_func(points))


# ==================== ParallelContext: HIGH LATENCY ====================


@pytest.mark.parametrize("num_points", [10, 100])
@pytest.mark.parametrize("mode", ["sequential", "parallel-thread-2", "parallel-process-2", "parallel-process-4"])
def test_benchmark_high_latency(benchmark, num_points, mode):
    """Sequential vs parallel with an expensive function, various sizes/workers."""
    points = _points(num_points)
    parts = mode.split("-")  # e.g. parallel-process-4
    kwargs = dict(min_workload_size=5)
    if parts[0] == "sequential":
        kwargs["enabled"] = False
    else:
        kwargs["backend"] = parts[1]
        kwargs["max_workers"] = int(parts[2])

    # Warm up the worker pool outside of the timed region
    # (must exceed min_workload_size so the pool is actually created)
    with ParallelContext(**kwargs) as ctx:
        ctx.execute(_high_latency_func, points[:, :5])

        def run():
            return ctx.execute(_high_latency_func, points)

        result = benchmark(run)
    np.testing.assert_allclose(result, _high_latency_func(points))


@pytest.mark.parametrize("num_points", [10, 100])
@pytest.mark.parametrize("mode", ["sequential", "parallel-thread-2", "parallel-thread-4", "parallel-process-2"])
def test_benchmark_io_bound_gil_released(benchmark, num_points, mode):
    """Sequential vs parallel with a GIL-releasing (I/O-bound) function.

    time.sleep() releases the GIL, so the thread backend shows a real
    speedup (~n_workers) while the process backend pays pickling overhead
    on top of the sleep time.
    """
    points = _points(num_points)
    parts = mode.split("-")
    kwargs = dict(min_workload_size=5)
    if parts[0] == "sequential":
        kwargs["enabled"] = False
    else:
        kwargs["backend"] = parts[1]
        kwargs["max_workers"] = int(parts[2])

    # Warm up the worker pool outside of the timed region
    with ParallelContext(**kwargs) as ctx:
        ctx.execute(_io_bound_func, points[:, :5])

        def run():
            return ctx.execute(_io_bound_func, points)

        result = benchmark(run)
    np.testing.assert_allclose(result, _io_bound_func(points))


@pytest.mark.parametrize("items_per_worker", [1, 25])
def test_benchmark_batching_high_latency(benchmark, items_per_worker):
    """Effect of min_items_per_worker chunk batching (2 process workers)."""
    points = _points(50)
    with ParallelContext(min_workload_size=5, max_workers=2,
                         min_items_per_worker=items_per_worker) as ctx:
        ctx.execute(_high_latency_func, points[:, :5])  # pool warmup

        def run():
            return ctx.execute(_high_latency_func, points)

        result = benchmark(run)
    np.testing.assert_allclose(result, _high_latency_func(points))


# ==================== evaluate_on_sparse_grid ====================


def _make_grids(n_dim, level):
    T, _ = create_sparse_grid(n_dim, level, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    return T, reduce_sparse_grid(T)


@pytest.mark.parametrize("parallel", [False, True])
def test_benchmark_evaluate_on_sparse_grid(benchmark, parallel):
    """evaluate_on_sparse_grid() sequential vs parallel (2 process workers)."""
    N, w = 2, 4
    T, Tr = _make_grids(N, w)
    f = _high_latency_func  # expensive evaluation: parallelization should pay off

    ctx_kwargs = dict(min_workload_size=10, max_workers=2)
    if not parallel:
        ctx_kwargs["enabled"] = False
    with ParallelContext(**ctx_kwargs) as ctx:
        # Warm the pool once before timing (>= min_workload_size=10)
        ctx.execute(f, Tr.knots[:, :10])

        def run():
            evals, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr, parallel_ctx=ctx)
            return evals

        result = benchmark(run)
    expected = f(Tr.knots)
    np.testing.assert_allclose(np.asarray(result).reshape(expected.shape), expected, atol=1e-15)


@pytest.mark.parametrize("parallel", [False, True])
def test_benchmark_evaluate_on_sparse_grid_recycling(benchmark, parallel):
    """evaluate_on_sparse_grid() with old-grid recycling, sequential vs parallel."""
    N = 2
    S, Sr = _make_grids(N, 3)
    T, Tr = _make_grids(N, 4)
    f = _high_latency_func

    # Old evaluations on the smaller grid (recycling source)
    evals_old, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Sr)

    ctx_kwargs = dict(min_workload_size=10, max_workers=2)
    if not parallel:
        ctx_kwargs["enabled"] = False
    with ParallelContext(**ctx_kwargs) as ctx:
        # Warm the pool once before timing (>= min_workload_size=10)
        ctx.execute(f, Tr.knots[:, :10])

        def run():
            evals, *_ = evaluate_on_sparse_grid(
                f, T, Sr=Tr, evals_old=evals_old, S_old=S, Sr_old=Sr, parallel_ctx=ctx)
            return evals

        result = benchmark(run)
    expected = f(Tr.knots)
    np.testing.assert_allclose(np.asarray(result).reshape(expected.shape), expected, atol=1e-15)
