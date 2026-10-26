import numpy as np

from sgpykit.main.evaluate_on_sparse_grid import evaluate_on_sparse_grid
from sgpykit.tools.type_and_property_check_functions.isreduced import isreduced
from sgpykit.util import matlab
from sgpykit.util.parallel import ParallelContext


def quadrature_on_sparse_grid(f, S, Sr, evals_old = None, S_old = None, Sr_old = None, tol = None, parallel_ctx=None):
    """
    Compute the integral of a function using a sparse grid.

    This function behaves similarly to `evaluate_on_sparse_grid`, but returns the
    approximated integral of the function. See `evaluate_on_sparse_grid` for more
    information on inputs.

    Parameters
    ----------
    f : callable or array_like
        Function handle or vector/matrix containing the evaluations of the function.
    S : object, optional
        Sparse grid object.
    Sr : object
        Reduced sparse grid object.
    evals_old : array_like, optional
        Previous evaluations of the function.
    S_old : object, optional
        Previous sparse grid object.
    Sr_old : object, optional
        Previous reduced sparse grid object.
    tol : float, optional
        Tolerance for incremental updates.
    parallel_ctx : ParallelContext, optional
        Context for parallel execution configuration.

    Returns
    -------
    res : float
        Approximated integral of the function.
    evals : array_like
        Evaluations of the function over the points of the sparse grid.

    Notes
    -----
    Possible calls:
    - res, evals = quadrature_on_sparse_grid(f, S=None, Sr)
    - res, evals = quadrature_on_sparse_grid(f, S, Sr, evals_old, S_old, Sr_old)
    - res, evals = quadrature_on_sparse_grid(f, S, Sr)
    - res, evals = quadrature_on_sparse_grid(f, S, Sr, evals_old, None, Sr_old)
    - res, evals = quadrature_on_sparse_grid(f, S, Sr, evals_old, S_old, Sr_old, tol)

    API is like in evaluate_on_sparse_grid(). S or Sr have to be provided explicitly,
    opposed to SGMK.
    """
    evals = None
    if S is None and Sr is None:
        raise ValueError('not enough input arguments')
    elif S is None and Sr:
        # res = QUADRATURE_ON_SPARSE_GRID(f,S), S being a reduced sparse grid.
        if callable(f) and isreduced(Sr):
            evals,*_ = evaluate_on_sparse_grid(f, S=None, Sr=Sr, parallel_ctx=parallel_ctx)
            res = evals @ np.transpose(Sr.weights)
        else:
            if matlab.isnumeric(f) and isreduced(Sr):
                res = f @ np.transpose(Sr.weights)
            else:
                raise ValueError('when quadrature_on_sparse_grid is called with two inputs, the second one must be a reduced sparse grid')
        return res,evals
    elif Sr_old is None:
        raise ValueError('QUADRATURE_ON_SPARSE_GRID does not accept inputs.')
    elif tol is None:
        evals,*_ = evaluate_on_sparse_grid(f,S=S,Sr=Sr,evals_old=evals_old,S_old=S_old,Sr_old=Sr_old,parallel_ctx=parallel_ctx)
    else:
        evals,*_ = evaluate_on_sparse_grid(f,S=S,Sr=Sr,evals_old=evals_old,S_old=S_old,Sr_old=Sr_old,tol=tol,parallel_ctx=parallel_ctx)

    res = evals @ np.transpose(Sr.weights)
    return res,evals
