import pytest
import numpy as np

from sgpykit import create_sparse_grid, reduce_sparse_grid, evaluate_on_sparse_grid, knots_uniform, lev2knots_lin, ParallelContext


@pytest.mark.parametrize(
    "N, w1, w2",
    [
        pytest.param(2,3,4, id="1"),
        pytest.param(4,1,2, id="2"),
    ]
)
def test_evaluate_on_sparse_grid(N, w1, w2):
    f = lambda x: sum(x)
    S, _ = create_sparse_grid(N, w1, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Sr = reduce_sparse_grid(S)
    T, _ = create_sparse_grid(N, w2, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)
    evals_non_rec, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr)
    evals_old, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Sr)
    evals_rec, *_ = evaluate_on_sparse_grid(f, T,
                                            Sr=Tr,
                                            evals_old=evals_old,
                                            S_old=S,
                                            Sr_old=Sr)
    np.testing.assert_allclose(evals_non_rec, evals_rec, atol=1e-15)


@pytest.mark.parametrize(
    "N, w1, w2",
    [
        pytest.param(2,3,4, id="1"),
        pytest.param(4,1,2, id="2"),
    ]
)
def test_evaluate_on_sparse_grid_from_vector(N, w1, w2):
    f = lambda x: 2*x
    S, _ = create_sparse_grid(N, w1, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Sr = reduce_sparse_grid(S)
    T, _ = create_sparse_grid(N, w2, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)
    evals_non_rec, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr)
    evals_old, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Sr)
    evals_rec, *_ = evaluate_on_sparse_grid(f, T,
                                            Sr=Tr,
                                            evals_old=evals_old,
                                            S_old=S,
                                            Sr_old=Sr)
    np.testing.assert_allclose(evals_non_rec, evals_rec, atol=1e-15)


def test_evaluate_on_sparse_grid_benchmark(benchmark):
    f = lambda x: sum(x)
    N = 8
    w = 6
    T, _ = create_sparse_grid(N, w, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)

    def func(T, Tr):
        evals_1, *_ = evaluate_on_sparse_grid(f, T, Sr=Tr,
                                              evals_old=[], S_old=[], Sr_old=[],
                                              # SGMK interface, so []s have to be provided here
                                              )
        return evals_1[0]

    result = benchmark(func, T=T, Tr=Tr)
    np.testing.assert_almost_equal(sum(result), 0.0)


def _sum_squared_test_func(x):
    """Helper function for parallel tests."""
    return np.sum(x**2, axis=0)

def test_evaluate_on_sparse_grid_with_parallel():
    """Test evaluate_on_sparse_grid with parallel context."""
    f = _sum_squared_test_func
    N = 2
    w = 4
    
    # Create grids
    T, _ = create_sparse_grid(N, w, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)
    
    # Serial execution
    f_eval_serial, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr)
    
    # Parallel execution with threshold met
    with ParallelContext(min_workload_size=10, max_workers=2) as ctx:
        f_eval_parallel, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr, parallel_ctx=ctx)
    
    # Results should be identical
    np.testing.assert_allclose(f_eval_serial, f_eval_parallel, atol=1e-15)


def test_evaluate_on_sparse_grid_with_parallel_below_threshold():
    """Test evaluate_on_sparse_grid with parallel context below threshold."""
    f = lambda x: np.sum(x**2, axis=0)
    N = 2
    w = 2  # Small grid
    
    # Create small grid
    T, _ = create_sparse_grid(N, w, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)
    
    # Serial execution
    f_eval_serial, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr)
    
    # Parallel execution with threshold not met (high threshold)
    with ParallelContext(min_workload_size=1000, max_workers=2) as ctx:
        f_eval_parallel, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr, parallel_ctx=ctx)
    
    # Results should be identical (both executed serially)
    np.testing.assert_allclose(f_eval_serial, f_eval_parallel, atol=1e-15)


def test_evaluate_on_sparse_grid_parallel_with_recycling():
    """Test evaluate_on_sparse_grid with parallel context and evaluation recycling."""
    f = _sum_squared_test_func
    N = 2
    w1 = 3
    w2 = 4
    
    # Create grids
    S, _ = create_sparse_grid(N, w1, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Sr = reduce_sparse_grid(S)
    T, _ = create_sparse_grid(N, w2, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)
    
    # Serial execution with recycling
    evals_old, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Sr)
    f_eval_serial, *_ = evaluate_on_sparse_grid(f, T, Sr=Tr, evals_old=evals_old, S_old=S, Sr_old=Sr)
    
    # Parallel execution with recycling
    with ParallelContext(min_workload_size=5, max_workers=2) as ctx:
        evals_old_parallel, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Sr, parallel_ctx=ctx)
        f_eval_parallel, *_ = evaluate_on_sparse_grid(f, T, Sr=Tr, evals_old=evals_old_parallel, S_old=S, Sr_old=Sr, parallel_ctx=ctx)
    
    # Results should be identical
    np.testing.assert_allclose(f_eval_serial, f_eval_parallel, atol=1e-15)


def test_evaluate_on_sparse_grid_parallel_disabled():
    """Test evaluate_on_sparse_grid with disabled parallel context."""
    f = lambda x: np.sum(x**2, axis=0)
    N = 2
    w = 4
    
    # Create grid
    T, _ = create_sparse_grid(N, w, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)
    
    # Serial execution
    f_eval_serial, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr)
    
    # Parallel execution but disabled
    with ParallelContext(min_workload_size=10, max_workers=2, enabled=False) as ctx:
        f_eval_parallel, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr, parallel_ctx=ctx)
    
    # Results should be identical (both executed serially)
    np.testing.assert_allclose(f_eval_serial, f_eval_parallel, atol=1e-15)


def test_evaluate_on_sparse_grid_parallel_custom_condition():
    """Test evaluate_on_sparse_grid with custom parallel condition."""
    f = _sum_squared_test_func
    N = 2
    w = 4
    
    # Create grid
    T, _ = create_sparse_grid(N, w, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)
    
    # Serial execution
    f_eval_serial, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr)
    
    # Parallel execution with custom condition (always true)
    with ParallelContext(
        parallel_condition=lambda n_points, points: True,
        max_workers=2
    ) as ctx:
        f_eval_parallel, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr, parallel_ctx=ctx)
    
    # Results should be identical
    np.testing.assert_allclose(f_eval_serial, f_eval_parallel, atol=1e-15)
