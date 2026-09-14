import numpy as np


def GK_lev_table(row_idx, col_idx):
    """
    Return entries from the Genz-Keister level table.

    This function returns the (row, col) submatrix of the pre-defined
    Genz-Keister level table. The table maps between Genz-Keister level
    indices, the number of knots, and the corresponding quadrature order.

    Parameters
    ----------
    row_idx : int or array_like
        Row index or indices of the table.
    col_idx : int or array_like
        Column index or indices of the table.

    Returns
    -------
    A : ndarray
        The submatrix of the Genz-Keister level table at the specified
        row and column indices.

    Notes
    -----
    In the tabulated GK knots, many levels have the same knots (with weights
    identical up to 10th-11th digit). For each group of levels with the same
    number of knots, one representative is selected, and a map is defined as
    follows:

    MAP=
    i:   l:     nb_knots:   order:
    0     0     0             0
    1     1     1             1
    2     3     3             3
    3     8     9             15
    4    15    19             29
    5    25    35             51
    """
    M = np.array([[0, 0, 0, 0,],
                  [1, 1, 1, 1,],
                  [2, 3, 3, 3,],
                  [3, 8, 9, 15,],
                  [4, 15, 19, 29],
                  [5, 25, 35, 51]
                  ])
    A = M[row_idx, col_idx]
    return A
