# ParallelContext Usage Guide

[`ParallelContext`](sgpykit/util/parallel.py:18) provides a Python context manager. It can be used for parallel function evaluation on sparse grids. It decides **whether** to parallelize (based on workload size or a custom condition) and **how** (separate processes or threads).

## Quick start

```python
from sgpykit.util.parallel import ParallelContext
from sgpykit.main import evaluate_on_sparse_grid
#
# ...
#
# only parallel when >=20 grid points
with ParallelContext(min_workload_size=20, max_workers=4) as ctx:
    # entering context
    # evals grid with f, might happen in parallel depending on workload
    f_eval, *_ = evaluate_on_sparse_grid(f, S, Sr, parallel_ctx=ctx)
# context exit
```

If no context object is passed → serial execution (safe default).

## Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `min_workload_size` | 10 (or environment variable `SGPYKIT_PARALLEL_THRESHOLD`) | Parallelize only if `num_points >= threshold` |
| `max_workers` | usable CPUs | Worker count cap |
| `backend` | `'process'` | `'process'` (multiple processes) or `'thread'` |
| `parallel_condition` | `None` | Lambda `(num_points, points) -> bool`, to customize threshold behaviour |
| `min_items_per_worker` | `1` | Minimum number of points each worker receives per chunk; higher values mean fewer, larger chunks. Useful when starting workers costs more than evaluating points |
| `enabled` | `True` | Master switch; `False` → always serial |

All invalid values raise `ValueError` at construction.

## When is parallelization worth it?

Consider the latency and nature of your evaluation function `f`:

- **Local, low-latency `f`** (e.g. NumPy formulas): parallelization often **not** needed. Parallelisation overhead can make things slower.
- **Local, compute-heavy `f`** (e.g. per-point simulations, slow per-point math): can help with `backend='process'`, which runs chunks in separate Python processes on multiple cores. Note `f` must be defined at module level (importable, "picklable"), since chunks are shipped to other processes.
- **External / high-latency `f`** (remote solver calls, network, database, file access): benefits most from parallelization. Try `backend='thread'`. No process startup cost, and `f` can be defined anywhere. However, when the Python's global interpreter lock (GIL) prevents parallelisation, then try `backend='process'`.

## Chunking strategy

- Points (columns of the `(d, n)` array) are split into balanced chunks via ceil division; no tiny trailing chunk (102 pts / 4 workers → 26, 26, 26, 24).
- `min_items_per_worker` is a **lower bound** per chunk; the chunk count shrinks if the workload doesn't divide evenly.
- Results are re-assembled in submission order, so column order matches the input.

## Fallback and error behavior

- Only **infrastructure failures** (e.g. pool could not be created, `f` could not be shipped to a worker process, broken communication) fall back to serial execution, with a warning via the `sgpykit` logger. Because the parallel attempt may have partially run before failing, `f` can be evaluated **twice**. Relevant if `f` has side effects (writes files, mutates state). For pure evaluation functions this is harmless.
- Errors raised by `f` itself inside a chunk **propagate** to the caller (no silent swallowing).
- Lambdas / functions defined inside another function or notebook cell cannot be shipped to worker processes → they trigger the serial fallback under `backend='process'`. Define `f` at module level, or use `backend='thread'`.

## Pitfalls

- **Jupyter / notebooks**: process pools created from a multithreaded parent can deadlock. Try `backend='thread'` in notebooks, or define `f` at module level in an imported file.
- **Don't share one context across processes**. Create the context inside each worker scope.

