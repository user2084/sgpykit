import pytest
import numpy as np

from sgpykit import create_sparse_grid, reduce_sparse_grid, lev2knots_doubling, knots_CC, \
    quadrature_on_sparse_grid, ParallelContext


def test_quadrature_on_sparse_grid():
    f = lambda x, b: np.prod(1.0 / np.sqrt(x + b), axis=0, keepdims=True)
    b = 3
    N = 4
    w = 4
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)
    result,_ = quadrature_on_sparse_grid(lambda x: f(x, b), S=None, Sr=Sr)
    assert np.isclose(result, 1.883984044753591)


def test_quadrature_on_sparse_grid_recycle():
    f = lambda x, b: np.prod((x + b) ** -0.5, axis=0)
    b = 5
    N = 2
    # the starting grid
    w = 3
    knots = lambda n: knots_CC(n, -2, 1, 'prob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)
    IS, evals_S = quadrature_on_sparse_grid(lambda x: f(x, b), S=None, Sr=Sr)
    assert np.isclose(IS, 0.228763833671746)
    # the new grid
    w = w+1
    T, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Tr = reduce_sparse_grid(T)
    IT_rec, _ = quadrature_on_sparse_grid(lambda x: f(x, b), S=T, Sr=Tr, evals_old=evals_S, S_old=S, Sr_old=Sr)
    assert np.isclose(IT_rec, 0.228763833671746)


def _sum_squared_keepdims_func(x):
    """Helper function for quadrature parallel tests."""
    return np.sum(x**2, axis=0, keepdims=True)

def _prod_power_func(x):
    """Helper function for quadrature recycling tests."""
    return np.prod((x + 5) ** -0.5, axis=0)

def test_quadrature_on_sparse_grid_with_parallel():
    """Test quadrature_on_sparse_grid with parallel context."""
    f = _sum_squared_keepdims_func
    N = 2
    w = 4
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    
    # Create grid
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)
    
    # Serial execution
    result_serial, evals_serial = quadrature_on_sparse_grid(f, S=None, Sr=Sr)
    
    # Parallel execution with threshold met
    with ParallelContext(min_workload_size=10, max_workers=2) as ctx:
        result_parallel, evals_parallel = quadrature_on_sparse_grid(f, S=None, Sr=Sr, parallel_ctx=ctx)
    
    # Results should be identical
    np.testing.assert_allclose(result_serial, result_parallel)
    np.testing.assert_allclose(evals_serial, evals_parallel)


def test_quadrature_on_sparse_grid_parallel_with_recycling():
    """Test quadrature_on_sparse_grid with parallel context and evaluation recycling."""
    f = _prod_power_func
    N = 2
    
    # Starting grid
    w = 3
    knots = lambda n: knots_CC(n, -2, 1, 'prob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)
    
    # Serial execution
    IS_serial, evals_S_serial = quadrature_on_sparse_grid(f, S=None, Sr=Sr)
    
    # Parallel execution
    with ParallelContext(min_workload_size=5, max_workers=2) as ctx:
        IS_parallel, evals_S_parallel = quadrature_on_sparse_grid(f, S=None, Sr=Sr, parallel_ctx=ctx)
    
    # Results should be identical
    np.testing.assert_allclose(IS_serial, IS_parallel)
    np.testing.assert_allclose(evals_S_serial, evals_S_parallel)
    
    # New grid with recycling
    w = w + 1
    T, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Tr = reduce_sparse_grid(T)
    
    # Serial execution with recycling
    IT_rec_serial, _ = quadrature_on_sparse_grid(f, S=T, Sr=Tr, evals_old=evals_S_serial, S_old=S, Sr_old=Sr)
    
    # Parallel execution with recycling
    with ParallelContext(min_workload_size=5, max_workers=2) as ctx:
        IT_rec_parallel, _ = quadrature_on_sparse_grid(f, S=T, Sr=Tr, evals_old=evals_S_parallel, S_old=S, Sr_old=Sr, parallel_ctx=ctx)
    
    # Results should be identical
    np.testing.assert_allclose(IT_rec_serial, IT_rec_parallel)


def test_quadrature_on_sparse_grid_parallel_below_threshold():
    """Test quadrature_on_sparse_grid with parallel context below threshold."""
    f = lambda x: np.sum(x**2, axis=0, keepdims=True)
    N = 2
    w = 2  # Small grid
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    
    # Create small grid
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)
    
    # Serial execution
    result_serial, evals_serial = quadrature_on_sparse_grid(f, S=None, Sr=Sr)
    
    # Parallel execution with threshold not met (high threshold)
    with ParallelContext(min_workload_size=1000, max_workers=2) as ctx:
        result_parallel, evals_parallel = quadrature_on_sparse_grid(f, S=None, Sr=Sr, parallel_ctx=ctx)
    
    # Results should be identical (both executed serially)
    np.testing.assert_allclose(result_serial, result_parallel)
    np.testing.assert_allclose(evals_serial, evals_parallel)
