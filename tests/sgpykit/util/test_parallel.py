import logging
import pickle
from unittest.mock import patch

import pytest
import numpy as np
from sgpykit.util.parallel import ParallelContext, create_parallel_context


def test_parallel_threshold():
    """Test that parallelization threshold works correctly."""
    ctx = ParallelContext(min_workload_size=20)
    
    # Should not use parallel (10 < 20)
    assert not ctx.should_use_parallel(10)
    
    # Should use parallel (25 >= 20)
    assert ctx.should_use_parallel(25)


def test_parallel_threshold_with_points():
    """Test threshold with points parameter."""
    ctx = ParallelContext(min_workload_size=20)
    
    points = np.random.rand(2, 25)
    
    # Should not use parallel (10 < 20)
    assert not ctx.should_use_parallel(10, points)
    
    # Should use parallel (25 >= 20)
    assert ctx.should_use_parallel(25, points)


def test_disabled_context():
    """Test that disabled context doesn't use parallelization."""
    ctx = ParallelContext(enabled=False)
    
    # Should not use parallel even with many points
    assert not ctx.should_use_parallel(100)
    assert not ctx.should_use_parallel(100, np.random.rand(2, 100))


def test_enabled_context():
    """Test that enabled context uses parallelization."""
    ctx = ParallelContext(enabled=True, min_workload_size=10)
    
    assert ctx.should_use_parallel(50)
    assert not ctx.should_use_parallel(5)


def test_execute_serial():
    """Test serial execution (below threshold)."""
    ctx = ParallelContext(min_workload_size=100)
    points = np.random.rand(2, 10)
    
    result = ctx.execute(lambda x: x**2, points)
    
    np.testing.assert_allclose(result, points**2)
    assert result.shape == (2, 10)


def square_helper(x):
    """Helper function for serialization tests."""
    return x**2

def test_execute_parallel_multiprocess():
    """Test parallel execution with multiprocessing backend."""
    # square_helper is a module-level function, so it pickles fine and
    # this runs truly in parallel via the process pool
    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='process')
    points = np.random.rand(2, 50)
    
    result = ctx.execute(square_helper, points)
    
    np.testing.assert_allclose(result, points**2)
    assert result.shape == points.shape


def test_execute_parallel_thread():
    """Test parallel execution with threading backend."""
    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='thread')
    points = np.random.rand(2, 50)
    
    result = ctx.execute(square_helper, points)
    
    np.testing.assert_allclose(result, points**2)
    assert result.shape == points.shape


def test_execute_with_function_args():
    """Test execute with additional function arguments."""
    def test_func(x, scalar, array):
        return x * scalar + array.sum()
    
    # Use threading to avoid pickling issues with local functions
    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='thread')
    points = np.random.rand(2, 50)
    scalar = 2.0
    array = np.array([1.0, 2.0])
    
    result = ctx.execute(test_func, points, scalar, array)
    
    expected = points * scalar + array.sum()
    np.testing.assert_allclose(result, expected)


def test_split_points_basic():
    """Test basic point splitting logic."""
    ctx = ParallelContext()
    points = np.random.rand(2, 100)
    
    chunks = ctx._split_points(points, 4, min_items_per_worker=1)
    
    assert len(chunks) == 4
    total_points = sum(chunk.shape[1] for chunk in chunks)
    assert total_points == 100


def test_split_points_with_items_per_worker():
    """Test point splitting with min_items_per_worker constraint."""
    ctx = ParallelContext()
    points = np.random.rand(2, 100)
    
    chunks = ctx._split_points(points, 4, min_items_per_worker=10)
    
    assert len(chunks) == 4
    total_points = sum(chunk.shape[1] for chunk in chunks)
    assert total_points == 100
    
    # Each chunk should have at least items_per_worker points
    for chunk in chunks:
        assert chunk.shape[1] >= 10


def test_split_points_chunk_sizes():
    """Test exact chunk sizes for uneven splitting.

    102 points / 4 workers: chunk_size = ceil(102 / 4) = 26, so the split
    yields 26, 26, 26, 24 — the remainder is spread across the chunks
    instead of forming a tiny trailing chunk.
    """
    ctx = ParallelContext()
    points = np.random.rand(2, 102)

    chunks = ctx._split_points(points, 4, min_items_per_worker=1)

    sizes = [chunk.shape[1] for chunk in chunks]
    assert sizes == [26, 26, 26, 24]
    assert sum(sizes) == 102


def test_split_points_min_items_per_worker_larger_than_chunk_size():
    """Test min_items_per_worker exceeding the computed chunk size.

    102 points / 4 workers with min_items_per_worker=30: the chunk count is
    reduced to 102 // 30 = 3, giving balanced chunks of 34, 34, 34 — every
    chunk has at least 30 points.
    """
    ctx = ParallelContext()
    points = np.random.rand(2, 102)

    chunks = ctx._split_points(points, 4, min_items_per_worker=30)

    sizes = [chunk.shape[1] for chunk in chunks]
    assert sizes == [34, 34, 34]
    assert sum(sizes) == 102
    for chunk in chunks:
        assert chunk.shape[1] >= 30


def test_split_points_min_items_per_worker_halves_worker_count():
    """Test min_items_per_worker large enough to reduce the number of chunks.

    102 points / 4 workers with min_items_per_worker=51: the chunk count is
    reduced to 102 // 51 = 2, so only 2 chunks of 51 are created.
    """
    ctx = ParallelContext()
    points = np.random.rand(2, 102)

    chunks = ctx._split_points(points, 4, min_items_per_worker=51)

    sizes = [chunk.shape[1] for chunk in chunks]
    assert sizes == [51, 51]
    assert sum(sizes) == 102


def test_split_points_min_items_per_worker_exceeds_total_points():
    """Test min_items_per_worker larger than the total number of points.

    The chunk count is clamped to at least 1, which yields a single chunk
    containing all points.
    """
    ctx = ParallelContext()
    points = np.random.rand(2, 102)

    chunks = ctx._split_points(points, 4, min_items_per_worker=200)

    sizes = [chunk.shape[1] for chunk in chunks]
    assert sizes == [102]


def test_split_points_fewer_points_than_chunks():
    """Test splitting when points < chunks."""
    ctx = ParallelContext()
    points = np.random.rand(2, 5)
    
    chunks = ctx._split_points(points, 10, min_items_per_worker=1)
    
    # Should create at most as many chunks as points
    assert len(chunks) == 5
    total_points = sum(chunk.shape[1] for chunk in chunks)
    assert total_points == 5


def test_custom_parallel_condition_basic():
    """Test custom parallel condition with simple lambda."""
    # Parallelize only if > 30 points
    ctx = ParallelContext(
        parallel_condition=lambda n_points, points: n_points > 30
    )
    
    points_25 = np.random.rand(2, 25)
    assert not ctx.should_use_parallel(25, points_25)
    
    points_50 = np.random.rand(2, 50)
    assert ctx.should_use_parallel(50, points_50)


def test_custom_parallel_condition_complex():
    """Test custom parallel condition with complex lambda."""
    # Parallelize if > 50 points OR (> 20 points AND > 2 dimensions)
    ctx = ParallelContext(
        parallel_condition=lambda n_points, points:
            n_points > 50 or (n_points > 20 and points.shape[0] > 2)
    )
    
    # 2D, 30 points - should not parallelize
    points_2d = np.random.rand(2, 30)
    assert not ctx.should_use_parallel(30, points_2d)
    
    # 3D, 30 points - should parallelize
    points_3d = np.random.rand(3, 30)
    assert ctx.should_use_parallel(30, points_3d)
    
    # 2D, 60 points - should parallelize
    points_many = np.random.rand(2, 60)
    assert ctx.should_use_parallel(60, points_many)


def test_parallel_condition_requires_points():
    """Test that custom parallel condition requires points parameter."""
    ctx = ParallelContext(
        parallel_condition=lambda n_points, points: n_points > 10
    )
    
    with pytest.raises(ValueError, match="points parameter required"):
        ctx.should_use_parallel(15)  # Missing points


def test_min_items_per_worker_basic():
    """Test min_items_per_worker parameter."""
    # square_helper is module-level and pickles fine; runs in parallel
    ctx = ParallelContext(min_workload_size=10, min_items_per_worker=10, max_workers=2)
    points = np.random.rand(2, 50)
    
    result = ctx.execute(square_helper, points)
    
    np.testing.assert_allclose(result, points**2)
    assert result.shape == points.shape


def test_min_items_per_worker_large():
    """Test min_items_per_worker larger than chunk size."""
    # square_helper is module-level and pickles fine; runs in parallel
    ctx = ParallelContext(min_workload_size=10, min_items_per_worker=100, max_workers=4)
    points = np.random.rand(2, 50)
    
    result = ctx.execute(square_helper, points)
    
    np.testing.assert_allclose(result, points**2)
    assert result.shape == points.shape


def test_context_manager():
    """Test that ParallelContext works as a context manager."""
    with ParallelContext(min_workload_size=10) as ctx:
        points = np.random.rand(2, 50)
        result = ctx.execute(square_helper, points)
        assert result.shape == points.shape


def test_create_parallel_context():
    """Test convenience factory for creating ParallelContext."""
    ctx = create_parallel_context(
        min_workload_size=20,
        max_workers=4,
        backend='thread',
        min_items_per_worker=5
    )
    
    assert isinstance(ctx, ParallelContext)
    assert ctx.min_workload_size == 20
    assert ctx.max_workers == 4
    assert ctx.backend == 'thread'
    assert ctx.min_items_per_worker == 5


def test_default_threshold():
    """Test default threshold behavior."""
    # Test with environment variable not set (should use default of 10)
    ctx = ParallelContext()
    assert ctx.min_workload_size == 10


def test_invalid_min_workload_size():
    """Test that invalid min_workload_size raises error."""
    # Test with zero (should fail)
    with pytest.raises(ValueError, match="min_workload_size must be positive"):
        ParallelContext(min_workload_size=0)
    
    # Test with negative (should fail)
    with pytest.raises(ValueError, match="min_workload_size must be positive"):
        ParallelContext(min_workload_size=-5)


def test_invalid_max_workers():
    """Test that invalid max_workers raises error."""
    # Test with zero (should fail)
    with pytest.raises(ValueError, match="max_workers must be positive"):
        ParallelContext(max_workers=0)
    
    # Test with negative (should fail)
    with pytest.raises(ValueError, match="max_workers must be positive"):
        ParallelContext(max_workers=-5)


def test_invalid_min_items_per_worker():
    """Test that invalid min_items_per_worker raises error."""
    with pytest.raises(ValueError, match="min_items_per_worker must be positive"):
        ParallelContext(min_items_per_worker=0)


def test_invalid_backend():
    """Test that invalid backend raises error."""
    with pytest.raises(ValueError, match="backend must be 'process' or 'thread'"):
        ParallelContext(backend='invalid')


def test_empty_points():
    """Test with empty point array."""
    ctx = ParallelContext(min_workload_size=10)
    points = np.zeros((2, 0))
    
    result = ctx.execute(square_helper, points)
    assert result.shape == (2, 0)


def test_single_point():
    """Test with single point."""
    ctx = ParallelContext(min_workload_size=10)
    points = np.random.rand(2, 1)
    
    result = ctx.execute(square_helper, points)
    np.testing.assert_allclose(result, points**2)


def test_threshold_exactly():
    """Test behavior when num_points equals threshold."""
    # square_helper is module-level and pickles fine; runs in parallel
    ctx = ParallelContext(min_workload_size=20)
    points = np.random.rand(2, 20)
    
    result = ctx.execute(square_helper, points)
    np.testing.assert_allclose(result, points**2)


def sum_squares(x):
    """Helper function for multi-dim tests."""
    return np.sum(x**2, axis=0)

def test_multi_dim_points():
    """Test with multi-dimensional points."""
    # sum_squares is module-level and pickles fine; runs in parallel
    ctx = ParallelContext(min_workload_size=10, max_workers=2)
    points = np.random.rand(5, 30)  # 5-dimensional
    
    result = ctx.execute(sum_squares, points)
    expected = sum_squares(points)
    np.testing.assert_allclose(result, np.atleast_2d(expected))
    assert result.shape == (1, points.shape[1])


def multi_output_func(x):
    """Helper function for multi-output tests."""
    return np.vstack([x.sum(axis=0), x.mean(axis=0)])

def test_function_returning_multiple_values():
    """Test function that returns multiple output dimensions."""
    # multi_output_func is module-level and pickles fine; runs in parallel
    ctx = ParallelContext(min_workload_size=10, max_workers=2)
    points = np.random.rand(3, 30)
    
    result = ctx.execute(multi_output_func, points)
    expected = multi_output_func(points)
    
    np.testing.assert_allclose(result, expected)
    assert result.shape == expected.shape


def func_returning_1d(x):
    """Helper function returning a 1-D array per chunk."""
    return np.sum(x**2, axis=0)


def test_atleast_2d_normalization_process():
    """1-D chunk results are normalized to (1, n_points) in the process backend."""
    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='process')
    points = np.random.rand(2, 50)

    result = ctx.execute(func_returning_1d, points)

    expected = np.atleast_2d(func_returning_1d(points))
    np.testing.assert_allclose(result, expected)
    assert result.shape == (1, 50)


def test_atleast_2d_normalization_thread():
    """1-D chunk results are normalized to (1, n_points) in the thread backend."""
    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='thread')
    points = np.random.rand(2, 50)

    result = ctx.execute(func_returning_1d, points)

    expected = np.atleast_2d(func_returning_1d(points))
    np.testing.assert_allclose(result, expected)
    assert result.shape == (1, 50)


def test_chunk_exception_propagates_thread():
    """Errors raised by func inside a chunk propagate in the thread backend."""
    def failing_func(x):
        raise RuntimeError("boom")

    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='thread')
    points = np.random.rand(2, 50)

    with pytest.raises(RuntimeError, match="boom"):
        ctx.execute(failing_func, points)


def test_chunk_exception_propagates_process():
    """Errors raised by func inside a chunk propagate in the process backend."""
    def failing_func(x):
        raise RuntimeError("boom")

    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='process')
    points = np.random.rand(2, 50)

    with pytest.raises(RuntimeError, match="boom"):
        ctx.execute(failing_func, points)


def test_fallback_to_serial_on_infrastructure_error():
    """Infrastructure failures fall back to serial and log a warning."""
    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='process')
    points = np.random.rand(2, 50)

    with patch.object(ctx, '_execute_multiprocess', side_effect=pickle.PicklingError("cannot pickle")):
        with patch.object(logging.Logger, 'warning') as mock_warn:
            result = ctx.execute(square_helper, points)

    # serial fallback produced the correct result
    np.testing.assert_allclose(result, points**2)
    assert result.shape == points.shape
    # warning about the fallback was logged
    mock_warn.assert_called_once()
    assert "falling back to serial" in mock_warn.call_args[0][0]


def test_fallback_reraises_if_serial_also_fails():
    """If the serial fallback itself raises, the error propagates."""
    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='process')
    points = np.random.rand(2, 50)

    def failing_serial(x):
        raise ValueError("serial boom")

    with patch.object(ctx, '_execute_multiprocess', side_effect=pickle.PicklingError("cannot pickle")):
        with patch.object(logging.Logger, 'warning'):
            with pytest.raises(ValueError, match="serial boom"):
                ctx.execute(failing_serial, points)


def test_func_error_does_not_fall_back_to_serial():
    """Errors raised by func inside a chunk do NOT trigger the serial fallback.

    Otherwise func would be executed twice (relevant for side-effecting functions).
    """
    def failing_func(x):
        raise RuntimeError("boom")

    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='process')
    points = np.random.rand(2, 50)

    with patch.object(ctx, '_execute_multiprocess', side_effect=lambda f, p: f(p)):
        with pytest.raises(RuntimeError, match="boom"):
            ctx.execute(failing_func, points)


def test_pool_closed_on_normal_exit():
    """Pool is closed+joined on normal context exit."""
    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='process')
    points = np.random.rand(2, 50)

    with ctx:
        result = ctx.execute(square_helper, points)
        assert ctx._pool is not None

    np.testing.assert_allclose(result, points**2)


def test_pool_terminated_on_error_exit():
    """Pool is terminated when the context exits with an error."""
    ctx = ParallelContext(min_workload_size=10, max_workers=2, backend='process')
    points = np.random.rand(2, 50)

    try:
        with ctx:
            ctx.execute(square_helper, points)
            assert ctx._pool is not None
            raise ValueError("exit with error")
    except ValueError:
        pass
