import numpy as np

from sgpykit.src.compare_sparse_grids import compare_sparse_grids
from sgpykit.src.lookup_merge_and_diff import lookup_merge_and_diff
from sgpykit.tools.type_and_property_check_functions.is_sparse_grid import is_sparse_grid
from sgpykit.tools.type_and_property_check_functions.isreduced import isreduced
from sgpykit.util.struct import Struct
from sgpykit.util.parallel import ParallelContext


def evaluate_on_sparse_grid(f, S, Sr, evals_old=None, S_old=None, Sr_old=None, tol=1e-14, parallel_ctx=None):
    """
    Evaluate a function on a sparse grid, possibly recycling previous calls.

    Several input combinations are possible, every input from the 3rd on can be set to [].

    Parameters
    ----------
    f : callable
        Function to evaluate. Takes a column vector point and returns a scalar or column vector.
    S : sparse grid or None
        Sparse grid structure. If None, Sr must be provided.
    Sr : reduced sparse grid or None
        Reduced sparse grid structure. Must be provided if S is None.
    evals_old : ndarray, optional
        Matrix storing previous evaluations of f on Sr_old.
    S_old : sparse grid or None, optional
        Previous sparse grid structure.
    Sr_old : reduced sparse grid or matrix, optional
        Previous reduced sparse grid or matrix of points.
    tol : float, optional
        Tolerance for testing point equality (default 1e-14).
    parallel_ctx : ParallelContext, optional
        Context manager for parallel execution configuration.
        If provided and conditions are met, function evaluations will be parallelized.

    Returns
    -------
    f_eval : ndarray
        Matrix storing evaluations of f on Sr.
    new_points : ndarray
        Points where f has been evaluated (new points w.r.t. previous grid).
    tocomp_list : ndarray
        Position of new_points in Sr.knots.
    discard_points : ndarray
        Points of Sr_old that have been discarded.
    discard_list : ndarray
        Index vector s.t. Sr_old.knots[:, discard_list] == discard_points.

    Notes
    -----
    Possible calls:
    - res, *_ = evaluate_on_sparse_grid(f, S=None, Sr)
    - res, *_ = evaluate_on_sparse_grid(f, S, Sr, evals_old, S_old, Sr_old)
    - res, *_ = evaluate_on_sparse_grid(f, S, Sr)
    - res, *_ = evaluate_on_sparse_grid(f, S, Sr, evals_old, None, Sr_old)
    - res, *_ = evaluate_on_sparse_grid(f, S, Sr, evals_old, S_old, Sr_old, tol)

    API is like in evaluate_on_sparse_grid(). S or Sr have to be provided explicitly,
    opposed to SGMK.
    """
    assert callable(f)

    # evaluate_on_sparse_grid(f,S=[],Sr=Sr)
    if Sr and not S:
        if not isreduced(Sr):
            raise ValueError('When evaluate_on_sparse_grid is called with two inputs, the second one must be a reduced sparse grid.')
        f_eval = simple_evaluate(f, Sr, parallel_ctx=parallel_ctx)
        new_points = Sr.knots
        tocomp_list = np.arange(0, len(Sr.weights))
        discard_points = []
        discard_list = []
        return f_eval, new_points, tocomp_list, discard_points, discard_list

    # evaluate_on_sparse_grid(f,S=S,Sr=Sr) # may works but keep it consistent with matlab SGMK interface
    if evals_old is None or Sr_old is None:
        raise ValueError(f'evals_old and Sr_old are required when S and Sr are given.')

    # evaluate_on_sparse_grid(f,S=S,Sr=Sr, evals_old, S_old, Sr_old)
    if isinstance(Sr_old, Struct) and not isreduced(Sr_old):
        raise ValueError('Sr_old must be a reduced sparse grid.')
    if S_old and len(S_old)!=0 and not is_sparse_grid(S_old):
        raise ValueError('S_old must be a sparse grid.')

    # evaluate_on_sparse_grid(f, S, Sr, [], [], [])

    assert Sr and S

    if len(evals_old)==0 and len(Sr_old)==0 and (S_old is None or len(S_old)==0):
        f_eval = simple_evaluate(f, Sr, parallel_ctx=parallel_ctx)
        new_points = Sr.knots
        tocomp_list = np.arange(0, len(Sr.weights))
        discard_points = []
        discard_list = []
        return f_eval, new_points, tocomp_list, discard_points, discard_list

    if S_old is None or len(S_old)==0: # here SR_OLD is matrix with points stored as columns, and we go for the slow code
        pts_list = Sr.knots
        pts_list_old = Sr_old
        tocomp_list, recycle_list, recycle_list_old, discard_list = lookup_merge_and_diff(pts_list.T, pts_list_old.T, tol)
    else: # here S_OLD is a sparse grid and SR_OLD is its reduced version and we go for the faster alternative
        pts_list = Sr.knots
        pts_list_old = Sr_old.knots
        tocomp_list, recycle_list, recycle_list_old, discard_list = compare_sparse_grids(S, Sr, S_old, Sr_old, tol)

    new_points = pts_list[:, tocomp_list]
    discard_points = [] if np.size(discard_list)==0 else pts_list_old[:, discard_list]

    N = pts_list.shape[1]
    s = evals_old.shape[0]

    f_eval = np.zeros((s, N))

    # parallel_ctx.execute() falls back to serial when conditions are not met
    # or no context is given, so a single call suffices here
    points = pts_list[:, tocomp_list]
    if parallel_ctx is not None:
        evals_new = np.atleast_2d(parallel_ctx.execute(f, points))
    else:
        evals_new = np.atleast_2d(f(points))

    f_eval[:, tocomp_list] = evals_new
    f_eval[:, recycle_list] = evals_old[:, recycle_list_old]

    return f_eval, new_points, tocomp_list, discard_points, discard_list


def simple_evaluate(f, Sr, parallel_ctx=None):
    """
    Evaluate a function on a reduced sparse grid without recycling previous evaluations.

    Parameters
    ----------
    f : callable
        Function to evaluate.
    Sr : reduced sparse grid
        Reduced sparse grid structure.
    parallel_ctx : ParallelContext, optional
        Context for parallel execution configuration.

    Returns
    -------
    output : ndarray
        Evaluations of f on Sr.knots.
    """
    # function output = simple_evaluate(f,Sr)
    #
    # does not exploit the previous sparse grids evaluations

    points = Sr.knots
    
    # Use parallelization if context provided and conditions met
    if parallel_ctx is not None and parallel_ctx.should_use_parallel(points.shape[1], points):
        output = np.atleast_2d(parallel_ctx.execute(f, points))
    else:
        output = np.atleast_2d(f(points))

    return output
