from __future__ import annotations

import copy
import logging
import types
import numpy as np

from sgpykit.main.create_sparse_grid_add_multiidx import create_sparse_grid_add_multiidx
from sgpykit.main.create_sparse_grid_multiidx_set import create_sparse_grid_multiidx_set
from sgpykit.main.evaluate_on_sparse_grid import evaluate_on_sparse_grid
from sgpykit.main.interpolate_on_sparse_grid import interpolate_on_sparse_grid
from sgpykit.main.reduce_sparse_grid import reduce_sparse_grid
from sgpykit.src.plot_idx_status import plot_idx_status
from sgpykit.src.tensor_grid import tensor_grid
from sgpykit.tools.idxset_functions import check_index_admissibility
from sgpykit.util.misc import matlab_to_python_index

logger = logging.getLogger(__name__)


def adapt_sparse_grid(f, N_full, knots, lev2knots, prev_adapt, controls):
    """
    Compute an adaptive sparse grid approximation of a function f.

    This function computes a sparse grid adapted to the approximation of a function F: Gamma subset R^N_full -> R^V
    according to some profit indicator. The implementation closely resembles the algorithm proposed in

    T. Gerstner, M. Griebel, Dimension-Adaptive Tensor-Product Quadrature, Computing 71, 65-87 (2003)

    and the various definitions of profits are discussed in

    F. Nobile, L. Tamellini, F. Tesei, and R. Tempone, An Adaptive Sparse Grid Algorithm for Elliptic PDEs
    with Lognormal Diffusion Coefficient, Sparse Grids and Applications - Stuttgart 2014,
    Lecture Notes in Computational Science and Engineering 109, p. 191-220,
    Springer International Publishing Switzerland 2016

    Roughly speaking, at each iteration the algorithm considers the multiidx with the highest profit
    among those already explored, adds it to the sparse grid, computes the profits of all its neighbouring
    indices (see controls.prof) and adds them to the list of explored indices.

    Parameters
    ----------
    f : callable
        A function-handle, taking as input a column vector (N_full components) and returning a column-vector value (V components).
    N_full : int
        A scalar, denotes the dimension of the space Gamma over which F and the sparse grid are defined.
        Note that this is the full dimension, yet the algorithm might explore a smaller subset of variables to
        start with, N_curr, as specified in CONTROLS.var_buffer_size (see below)
    knots : callable or list of callables
        KNOTS can be a function handle or a cell arrays of function handles to differentiate between directions, but they must be
        either all nested or all non-nested formulae (the code cannot work partially with nested and partially with non-nested points)
        KNOTS cannot be used with the 'nonprob' option if later the "buffer" controls functionality is used, see later
    lev2knots : callable
        LEV2KNOTS must be a single function handle, i.e., it is not possible to use
        e.g., lev2knots_2step in one direction and lev2knots_lin in the next one.

        For instance, in N=2, the following setups are ok:

        - ``knots = [lambda n: knots_uniform(n, 0, 1), lambda n: knots_uniform(n, 3, 5)]``,
          ``lev2knots = lev2knots_lin`` (same family on different intervals, same lev2knots)
        - ``knots = [lambda n: knots_CC(n, 0, 1), lambda n: knots_CC(n, 3, 5)]``,
          ``lev2knots = lev2knots_doubling`` (same family on different intervals, same lev2knots)

        while the following ones are not:

        - ``knots = [lambda n: knots_uniform(n, 0, 1), lambda n: knots_CC(n, 3, 5)]``,
          ``lev2knots = lev2knots_doubling`` (one nested family, one non-nested family)
        - ``knots = [lambda n: knots_uniform(n, 0, 1), lambda n: knots_uniform(n, 3, 5)]``,
          ``lev2knots = [lev2knots_lin, lev2knots_doubling]`` (different lev2knots in different directions)
        - ``knots = [lambda n: knots_CC(n, 0, 1, 'nonprob'), lambda n: knots_CC(n, 3, 5, 'nonprob')]``,
          ``lev2knots = lev2knots_doubling`` (nonprob weights, cannot be used together with buffer)

        See tutorial_adaptive.ipynb for an example that shows this.
    prev_adapt : types.SimpleNamespace or None
        By setting PREV_ADAPT as the output ADAPTED of a previous computation, the new computation
        will resume from where the previous one left. Set PREV_ADAPT = None for a fresh start
    controls : dict or types.SimpleNamespace
        A struct defining several parameters that control the algorithm flow. It is possible to
        define only some (or none) of these parameters, the others will be set to default value. Only
        the parameter 'nested' is mandatory. The parameter 'pdf' is mandatory only if controls.prof is
        set to any of the 'weighted*' choices (see below). For safety, 'pdf' is not set to default and
        an error will be raised if the field is not set
        controls.nested : bool
            (mandatory) true if nested points will be used, false otherwise
        controls.max_pts : int
            is the maximum number of points the adapted sparse grid can have (default 1000).
            Observe that if the max nb of points is reached while checking the neighbours
            of the current idx, the algorithm will not be interrupted.
        controls.prof_tol : float
            the algorithm will stop as soon as profit of the considered idx is smaller
            then prof_tol (default 1e-14)
        controls.recycling : str
            can be set to 'priority_to_evaluation' (default) or 'priority_to_recycling'. This affects
            only when non-nested points are used. Because in this case points enter and exit
            from the sparse grid, we should keep a record of all points that have been
            evaluated and recycle from it whenever possible. However, doing this exactly is **very
            expensive** in terms of computational cost with the current data structure if N is
            large. So, unless your function evaluation is **very** expensive, we recommend to
            leave the default setting, which does a "partial search", i.e. searches for
            recycling only in the previous grid (instead than on the entire sparse grid
            history) and therefore implies that the same point might be evaluated multiple times.
        controls.prof : str
            chooses how to compute the profit of an idx. In general, this entails comparing
            the sparse grids approximation of f *before* adding idx to the sparse grids (call
            this approximation S) and *after* having added it (call this approximation T).
            Note that S, T are actually functions of y in Gamma and return a vector with V components
            (same as f, the function to be approximated). In general, profit takes either of the two forms
            P(idx) = Op_y ( Op_vect(S,T) ) , P(idx) = Op_vect ( Op_y(S,T) )
            where Op_vect acts as a "vector norm" over the V components of S,T (e.g., the euclidean norm for
            fixed y), and Op_y acts as a norm over Gamma (e.g. the maximum over a discrete set of values of y).
            Currently, the choices below are available. In all of these choices, Op_vect can
            be changed by setting the field controls.op_vect (see below)

            'Linf' :
                P(idx) = Op_y( Op_vect(S(y),T(y) ) where
                Op_vect(S(y),T(y)) = euclidean norm of S(y) - T(y) for a fixed y;
                Op_y = max over a discrete set of points of Gamma.
                For nested points, the algorithm considers the points that would
                be added to the grid by the considered idx,
                evalutes the function and the previous sparse grid on such
                points, and takes the max of such difference
                as profit estimate. For non-nested points, the sparse grid approx
                is not interpolant, hence we consider the difference between both the
                *new* and the previous sparse grids on the new points.

            'Linf/new_points' (default) :
                the 'Linf' profit estimate above is further divided
                by the number of new points considered
                (i.e. gain-cost ratio). For non-nested points, this is actually
                the cost of the tensor grid associated to the current midx (see
                Nobile Tamellini Tempone, ``Convergence of quasi optimal sparse
                grid approximation ...''

            'weighted Linf' :
                same as 'Linf', but the difference between
                previous sparse grid and function evaluation (or between new and
                old sparse grids, depending whether nested or non-nested points are used) is
                multiplied by the probability density function (see below for details).
                Useful when considering sparse grids on unbounded domains.

            'weighted Linf/new_points' :
                the 'weighted Linf' profit estimate is divided by the number of new points

            'deltaint' :
                P(idx) = Op_vect ( Op_y(S,T) ) = euclidean_norm( expct(S) - expct(T))
                i.e. the profit estimate is the euclidean norm of the difference of the
                quadratures using the sparse grids with and without the given idx

            'deltaint/new_points' :
                the 'deltaint' profit estimate is divided by the number of new points

        controls.op_vect : callable
            changes the operator op_vect above. It is defined as a handle function
            lambda A, B: some_function_of_(A, B)
            where A B are two matrices with V rows, containing the evaluation of the two operators S T discussed above
            as columns, e.g. A=[S(y1) S(y2) S(y3) ...], B=[T(y1) T(y2) T(y3) ...]
        controls.pts_tol : float
            is the tolerance used by evaluate_on_sparse_grid to check for equal points (default 1e-14)
        controls.pdf : callable
            pdf over Gamma subset R^N_full, the set over which the sparse
            of the random variables, to be used in weighted* profits. It must be provided
            as a function handle that takes as input a matrix with N rows (for generic N), i.e. where each column is a point
            where the pdf must be evaluated. For instance, if the weight is the standard gaussian
            pdf on each direction, controls.pdf = lambda Y: np.prod(norm.pdf(Y, 0, 1), axis=0), with Y matrix with N
            rows. For safety, 'pdf' is not set to default and an error will be raised if the field is not set
        controls.var_buffer_size : int
            The algorithm starts exploring N_curr = <var_buffer_size> dimensions. As soon as
            points are placed in one dimension, a new dimension is added to the set of
            explored variables, i.e., N_curr = N_curr+1. In this way we ensure that there
            are always <var_buffer_size> explored but "non-activated" variables, i.e.,
            along which no point is placed (default N_full).

            **IMPORTANT NOTE**: Use this option with care! Is evaluating the function at a
            subset of variables equivalent to evaluating the function at the entire set of
            variables, where the ones outside the buffer are fixed at the mid-point of the
            interval? If this is not the case, the buffer cannot be used.

            For instance:

            - ``f1 = 1/exp(y1*c1 + y2*c2 + y3*c3)``, with y1,y2 in [-0.5,0.5], y3 in [-0.2,0.2]
              This function is OK because f1 restricted to y1,y2 is y1*c1 + y2*c2 =
              y1*c1 + y2*c2 + 0*c3 = y1*c1 + y2*c2 + y3*c3 with y3 held at midpoint
            - ``f2 = 1/exp(y1*c1 + y2*c2 + y3*c3)``, with y1,y2,y3 in [0,1] (note the interval not centered in 0)
              This function is NOT OK because f1 restricted to y1,y2 is y1*c1 + y2*c2 ≠
              y1*c1 + y2*c2 + 0.5*c3 = y1*c1 + y2*c2 + y3*c3 with y3 held at midpoint
            - ``f3 = y1 * y2 * y3``, with y1,y2,y3 in [0,1]
              Is NOT OK because f1 restricted to y1,y2 is y1*y2 ≠ y1*y2*0.5 =
              y1*y2*y3 with y3 held at midpoint
            - ``f4 = cos(y1) + cos(y2) + cos(y3)``, with y1,y2,y3 in [-1,1]
              Is NOT OK because f1 restricted to y1,y2 is cos(y1) + cos(y2) ≠
              cos(y1) + cos(y2) + 1 = cos(y1) + cos(y2) + cos(y3) with y3 held at midpoint
            - ``f5 = y1 * y2 * y3``, with y1,y2,y3 in [0,2]
              Is OK because f1 restricted to y1,y2 is y1*y2 = y1*y2*1 =
              y1*y2*y3 with y3 held at midpoint

            See tutorial_adaptive.ipynb for an example that shows this.
        controls.plot : bool
            plot multiidx set and pause (default false)

    Returns
    -------
    adapted : types.SimpleNamespace
        A struct containing the results of the algorithms. The same structure can be passed
        as input (PREV_ADAPT) to resume the computation. Its fields are
        adapted.S : the adapted sparse grid (built over both all indices whose profit has been computed,
            even those whose profit hasn't yet been chosen as the best one, i.e. those whose
            neighbours haven't been explored yet).
        adapted.Sr : its reduced version;
        adapted.f_on_Sr : the evaluations of f over the list of points contained in Sr;
        adapted.nb_pts : the number of points in Sr;
        adapted.nested : true if points used are nested
        adapted.nb_pts_visited : the number of points visited while building the sparse grid. For non-nested
            points, this will be larger than nb_pts, because some points enter and then exit
            the grid when the corresponding idx exits from the combination technique.
        adapted.num_evals : the number of function evaluations done to compute the sparse grid. This is not
            necessarily equal to the previous one, because for speed reasons (looking for
            points in expensive for N large) sometimes one point might be recomputed twice
        adapted.N : the current number of dimensions considered for exploration
        adapted.private : a structure contained more detailed information on the status of the adaptive algorithm, that is needed
            to resume the computation. In particular, the needed data structure consists of:
            private.G : the multi-idx set used to build the sparse grid. This is called
                I in Gerstner-Griebel paper.
            private.I : the set of explored indices; G = I plus the neighbours of I.
                This is called O in Gerstner-Griebel paper. I is sorted
                lexicographically
            private.G_log : same as private.G, but sorted in the order with which indices
                are added to G insted of lexicographic
            private.I_log : same as private.I, but sorted in the order with which indices
                are added to I insted of lexicographic
            private.coeff_G : the coefficients of the indices in private.G in the combination technique
            private.N_log : for each iteration, the value of N_curr
            private.idx=idx : the idx whose neighbour is the next to be explored
            private.maxprof : the corresponding profit
            private.profits : the set of idx whose profit has been computed. They have been added to the grid
                but their neighbour is yet to be explored.
            private.idx_bin : the corresponding set of profits. This is called A in
                Gerstner-Griebel paper
            private.var_with_pts : vector of variables in which we have actually put points.
                length(var_with_pts) + controls.var_buffer_size = N_curr
            private.nb_pts_log : for each iteration, the current nb_pts
            private.Hr : for non-nested points, the list of points visited by the
                algorithm. Empty for nested-points
            private.f_on_Hr : for non-nested points, the evaluations of f on Hr. Empty for nested-points
    """

    # set control fields
    controls = default_controls(controls, N_full)

    # init / resume adaptive algorithm.
    #
    # Here's the data structure we need
    #
    # --> idx       : is the idx with the highest profit, whose neighbour we will next explore
    # --> maxprof   : the corresponding profit
    # --> I         : is the set of explored indices (the grid is actually larger, it includes as well their
    #                   neighbours), corresponds to O in Gerstner-Griebel
    # --> I_log     : I must be sorted lexicographically for software reasons. I_log is the same set of indices, but
    #                   sorted in the order in which they are chosen by the algorithm
    # --> idx_bin   : is the set of idx whose profit has been computed. They have been added to the grid but their
    #                 neighbour is yet to be explored. Corresponds to A in Gerstner-Griebel
    # --> profits   : is the corresponding set of profits
    # --> G         : is the set of the grid. Corresponds to I in Gerstner-Griebel
    # --> G_log     : same as I_log,  but for G
    # --> coeff_G   : coefficients of the combination technique applied to G
    # --> nb_pts    : the number of points in the grid
    # --> nb_pts_visited : the number of points visited during the sparse grid construction
    # --> num_evals : the number of function evaluations performed
    # --> nb_pts_log: for each iteration, the current nb_pts
    # --> S         : create_sparse_grid_multiidx_set(G,knots,lev2knots);
    # --> Sr        : reduce_sparse_grid(S);
    # --> f_on_Sr   : evaluate_on_sparse_grid(f,Sr);
    # --> var_with_pts: vector of variables in which we have actually put points.
    #                   length(var_with_pts) + controls.var_buffer_size = N_curr
    # --> N_log     : for every iteration, the value of N_curr

    # we need to distinguish the full dimension of the parameter space (N_tot) and the dimensional of the subspace
    # currently explored (N_curr = length(expl_var)). For consistency with previous code, we actually use N instead of N_curr

    # this is the current number of dimensions activated. Its value might change at the next line if we are
    # actually resuming a previous computation. Also, we make sure that this is actually no larger than N_full
    # already!

    N = controls.var_buffer_size

    (N, N_log, var_with_pts, S, Sr, f_on_Sr, I, I_log, idx, maxprof,
     idx_bin, profits, G, G_log, coeff_G, Hr, f_on_Hr,
     nb_pts, nb_pts_log, num_evals, intf) = start_adapt(f, N, knots, lev2knots,
                                                        prev_adapt, controls)

    # here's the adapt algo
    while nb_pts < controls.max_pts:

        if maxprof < controls.prof_tol:
            break

        # -------------------------------------------------------------
        #   Compute neighbours of the idx with highest profit
        # -------------------------------------------------------------
        # In MATLAB: Ng = ones(N,1)*idx + eye(N);
        # In Python we build a (N, N) array where each row is ``idx`` plus the
        # unit vector e_j.
        Ng = np.tile(idx+1, (N, 1)) + np.eye(N, dtype=np.int64)
        Ng_0 = matlab_to_python_index(Ng)

        # -------------------------------------------------------------
        #   Remove those that are not admissible w.r.t. the current set I
        # -------------------------------------------------------------
        admissible_rows = []
        for row in Ng_0:
            is_adm,*_ = check_index_admissibility(row, I)
            if is_adm:
                admissible_rows.append(row)
        Ng = np.array(admissible_rows, dtype=np.int64)

        # -------------------------------------------
        # for those safe, add them to the grid & compute profit
        # -------------------------------------------
        M = Ng.shape[0]
        Prof_temp = np.zeros(M)

        for m in range(M):
            # the current idx
            jj = Ng[m, :]

            G_log = np.vstack([G_log, jj])
            T, G, coeff_G = create_sparse_grid_add_multiidx(jj, S, G, coeff_G, knots, lev2knots)

            Tr = reduce_sparse_grid(T, controls.pts_tol)

            (nb_pts, num_evals, nb_pts_log, Prof_temp[m],
             f_on_Tr, Hr, f_on_Hr, intnew) = compute_profit_idx(
                Ng[m, :], f, S, T, Tr, Sr, Hr, f_on_Sr, f_on_Hr,
                intf, nb_pts, num_evals, nb_pts_log,
                knots, lev2knots, controls)

            S = T
            Sr = Tr
            f_on_Sr = f_on_Tr
            intf = intnew

        # -------------------------------------------
        # update the list of profits
        # -------------------------------------------

        profits = np.append(profits, Prof_temp)
        # observe that any of the indices in Ng can be already in idx_bin, i.e. there are no duplicates. Indeed, if ii is in Ng, this
        # means that ii is admissible wrt I, therefore all previous indices jj<=ii are already included in I, therefore they have already
        # left idx_bin therefore no jj will be considered and ii will no longer be generated in Ng.
        # Anyway, let's leave a check for the moment

        # idx_bin = np.vstack([idx_bin, Ng]) # will fail if one is empty []
        # NOTE: because numpy cannot have a 2D empty set from np.atleast_2d([]) leading to a shape conflict we use this:
        parts = []
        if idx_bin.size:
            parts.append(np.atleast_2d(idx_bin))
        if Ng.size:
            parts.append(np.atleast_2d(Ng))
        if parts:
            idx_bin = np.vstack(parts)
            if np.unique(idx_bin, axis=0).shape[0] != idx_bin.shape[0]:
                raise RuntimeError('an index is appearing twice in idx_bin')
        else:
            raise RuntimeError('Unexpected empty sets.')

        # -------------------------------------------
        # take the next idx with highest profit. I'd like to remove it but first I need to add (possibly) the new dimension.
        # Note that idx it's already in G
        # -------------------------------------------

        k = np.argmax(profits)
        maxprof = profits[k]
        idx = idx_bin[k, :]

        # -------------------------------------------
        # now we take care of possibly changing the variables buffer, i.e., adding a new variable to be explored
        # -------------------------------------------
        # find the list of variables in which the current choice of idx wants to add points
        to_be_explored = np.where(idx > 0)[0]
        new_var = np.setdiff1d(to_be_explored, var_with_pts, assume_unique=True) # i.e., the variables activated by the new profit in which
                                                        # no points have still been placed. Note that obviously new_var <= N_full

        if new_var.size == 0:
            # do nothing here, we keep working in the same subspace
            logger.debug("keep number of dimensions as is")
        elif new_var.size == 1:
            # in this case, we add the new proposed variable to the ones where we have added points and
            # we also add one variable to the explored one, to maintain the balance
            # length(var_with_pts) + controls.var_buffer_size = N_curr
            logger.debug("adding points a new variable")
            var_with_pts = np.union1d(var_with_pts, new_var)
            # the new variable to be explored is necessarily the N_curr+1, so adding it is just a matter
            # of adding a column to the proper containers, unless the hard limit of N_full,
            # i.e. the total number of variables of f has been reached; then, there are no more variables to explore.
            # Observe though that even if all variables are explored, not all variables necessarily have points,
            # so the previous line is not to be put inside the following if
            if N < N_full:
                logger.debug("adding a new variable to the explored dimensions")

                # let's add one variable and increase the containers. We first add one dimension to the index containers
                N = N + 1
                # fill those index lists with initial one's as in MATLAB
                I = np.hstack([I, np.zeros((I.shape[0], 1), dtype=np.int64)])
                I_log = np.hstack([I_log, np.zeros((I_log.shape[0], 1), dtype=np.int64)])
                G = np.hstack([G, np.zeros((G.shape[0], 1), dtype=np.int64)])
                G_log = np.hstack([G_log, np.zeros((G_log.shape[0], 1), dtype=np.int64)])
                idx_bin = np.hstack([idx_bin, np.zeros((idx_bin.shape[0], 1), dtype=np.int64)])

                # then we add one coordinate to the sparse grid points that we have already generated. The new
                # coordinate is the midpoint of the parameter space along the new direction. Here it's crucial that
                # f(x)==f([x mp]), otherwise I have to reevaluate everything, which I do not want to do. I'll just raise a
                # warning for the time being
                logger.debug("adding a new variable, hence a new coordinate to points, held ad midpoint. Does this change f evaluations? If so, the code does not work because it does not recompute function evaluations")

                if callable(knots):
                    mp,_ = knots(1)
                elif isinstance(knots, list):
                    mp,_ = knots[N - 1](1)
                else:
                    raise ValueError('SparseGKit:WrongInput: knots must be either a function handle or a cell array of function handles')

                # Extend every tensor grid stored in ``S`` (the sparse grid before reduction)
                for j in range(len(S)):
                    # add the new coordinate to the knot matrix
                    S[j].knots = np.vstack([S[j].knots, np.full((1, S[j].size), mp)])
                    # store the midpoint in the per-dimension cell array
                    S[j].knots_per_dim = np.append(S[j].knots_per_dim, [mp])
                    # update the 1-D size vector
                    S[j].m = np.append(S[j].m, 1)
                    # the multi-index for this tensor grid gets a new entry = 0
                    S[j].idx = np.append(S[j].idx, 0)

                # Extend the reduced grid ``Sr`` (only the knot matrix - weights are unchanged)
                Sr.knots = np.vstack([Sr.knots, np.full((1, Sr.knots.shape[1]), mp)])

                # For non-nested points we also have to extend the full list ``Hr``
                if not controls.nested:
                    Hr.knots = np.vstack([Hr.knots, np.full((1, Hr.knots.shape[1]), mp)])

                # because we have added the new variable, we need to add right away the profit of the first index in
                # its direction, as if we had started with Ng=ones(N+1,1)*idx + eye(N+1) in the first place. In this way,
                # we trigger exploration in that direction too

                Ng_new = np.zeros((1, N), dtype=np.int64)
                Ng_new[0, -1] = 1

                G_log = np.vstack([G_log, Ng_new])
                T, G, coeff_G = create_sparse_grid_add_multiidx(
                    Ng_new, S, G, coeff_G, knots, lev2knots
                )
                Tr = reduce_sparse_grid(T, controls.pts_tol)

                (
                    nb_pts,
                    num_evals,
                    nb_pts_log,
                    Prof_new,
                    f_on_Tr,
                    Hr,
                    f_on_Hr,
                    intnew,
                ) = compute_profit_idx(
                    Ng_new[0, :],
                    f,
                    S,
                    T,
                    Tr,
                    Sr,
                    Hr,
                    f_on_Sr,
                    f_on_Hr,
                    intf,
                    nb_pts,
                    num_evals,
                    nb_pts_log,
                    knots,
                    lev2knots,
                    controls,
                )

                # update the current grid with the newly added index
                S = T
                Sr = Tr
                f_on_Sr = f_on_Tr
                intf = intnew

                # add the new profit to the list
                profits = np.append(profits, Prof_new)
                idx_bin = np.vstack([idx_bin, Ng_new])

                if np.unique(idx_bin, axis=0).shape[0] != idx_bin.shape[0]:
                    raise ValueError('an index is appearing twice in idx_bin')

                # now, this new profit might already be the best one, so I need to reconsider my previous decision.
                # I recompute the best profit and now I can also remove it from idx_bin (a few lines below)
                # recompute the best profit (the newly added index might be the best)

                k = np.argmax(profits)
                maxprof = profits[k]
                idx = idx_bin[k, :]

            else:
                logger.debug("maximum number of variables to be explored reached, continuing as is")
        else:
            raise NotImplementedError(
                "still don't know how to deal with the case where 2 or more new variables need to be explored"
            )

        # -------------------------------------------
        # end of the section where we change the variables buffer
        # -------------------------------------------
        I_log = np.vstack([I_log, idx])
        I_combined = np.vstack([I, idx])
        # sort rows lexicographically by all columns
        I = I_combined[np.lexsort(np.rot90(I_combined))] # note that I must be lexicog sorted for check_index_admissibility(Ng(m,:),I) to work

        # remove the just-chosen index from the candidate list
        idx_bin = np.delete(idx_bin, k, axis=0)
        profits = np.delete(profits, k)

        N_log = np.append(N_log, N)

        # optional live plot (only if the user asked for it)
        if controls.plot:
            plot_idx_status(G, I, idx_bin, idx)

    # done with the loop on idx, we can reduce Hr to squeeze out multiple occurrencies of same point (if any) and
    # fix nb_pts_visited
    if not controls.nested and controls.recycling == "priority_to_evaluation":
        # I need to make Hr look like a sequence of tensor grids. I actually only need to add to it a fake weights field
        Hr.weights = np.zeros((1, Hr.knots.shape[1]))
        Hr = reduce_sparse_grid(Hr, controls.pts_tol)
        f_on_Hr = f_on_Hr[:, Hr.m]  # here I remove duplicates in f_on_Hr too

    adapted = types.SimpleNamespace(
        N=N,
        S=S,
        Sr=Sr,
        f_on_Sr=f_on_Sr,
        nb_pts=nb_pts,
        nested=bool(controls.nested),
        nb_pts_visited=nb_pts if controls.nested else Hr.knots.shape[1],
        num_evals=num_evals,
        intf=intf,
    )

    if num_evals > adapted.nb_pts_visited:
        logger.debug(
            f"Some points have been evaluated more than once. Total: {num_evals - adapted.nb_pts_visited} "
            f"extra evaluations over {adapted.nb_pts_visited} function evaluations"
        )

    # -----------------------------------------------------------------
    #   Private data needed to resume a computation
    # -----------------------------------------------------------------
    private = types.SimpleNamespace(
        G=G,
        G_log=G_log,
        coeff_G=coeff_G,
        I=I,
        I_log=I_log,
        maxprof=maxprof,
        idx=idx,
        profits=profits,
        idx_bin=idx_bin,
        Hr=Hr,
        f_on_Hr=f_on_Hr,
        var_with_pts=var_with_pts,
        N_log=N_log,
        nb_pts_log=nb_pts_log,
    )

    adapted.private = private

    return adapted

def default_controls(controls, N_full):
    """
    Fill the ``controls`` dictionary with default values for all optional fields.
    The mandatory fields are ``'nested'`` (always required) and, when a weighted
    profit is requested, ``'pdf'``.

    Parameters
    ----------
    controls : dict or types.SimpleNamespace
        Dictionary of control parameters (will be converted to SimpleNamespace).
    N_full : int
        Full dimension of the parameter space.

    Returns
    -------
    types.SimpleNamespace
        Updated controls as SimpleNamespace with default values for missing fields.
    """
    # Convert to dict if it's a SimpleNamespace
    if isinstance(controls, types.SimpleNamespace):
        controls = vars(controls)
    
    defaults = {
        "pts_tol": 1e-14,
        "max_pts": 1000,
        "prof_tol": 1e-14,
        "prof": "Linf/new_points",
        "plot": False,
        "op_vect": lambda A, B: np.sqrt(np.sum((A - B) ** 2, axis=0)),
        "recycling": "priority_to_evaluation",
    }

    # copy defaults only for missing keys
    for key, val in defaults.items():
        controls.setdefault(key, val)

    # Convert to SimpleNamespace for attribute access
    controls = types.SimpleNamespace(**controls)

    # mandatory fields
    if not hasattr(controls, 'nested'):
        raise KeyError("controls must specify the value of 'nested' field")

    # weighted profits need a pdf
    if controls.prof in {"weighted Linf/new_points", "weighted Linf"}:
        if not hasattr(controls, 'pdf'):
            raise KeyError(
                "you need to set the field 'pdf' to use 'weighted Linf' and 'weighted Linf/new_points' profits"
            )

    # buffer size - cannot be larger than the full dimension
    if not hasattr(controls, 'var_buffer_size'):
        controls.var_buffer_size = N_full
    elif controls.var_buffer_size > N_full:
        controls.var_buffer_size = N_full

        logger.warning(
            "controls.var_buffer_size cannot be greater than N_full. "
            "The code will proceed with controls.var_buffer_size = N_full."
        )

    # check recycling option
    if controls.recycling not in {"priority_to_evaluation", "priority_to_recycling"}:
        raise ValueError("unknown value of field controls.recycling")

    return controls

# -------------------------------------------------------------------------
#   Helper: initialise (or resume) the adaptive algorithm
# -------------------------------------------------------------------------
def start_adapt(f, N, knots, lev2knots, prev_adapt, controls):
    """
    Initialize or resume the adaptive algorithm.

    Parameters
    ----------
    f : callable
        Function to approximate.
    N : int
        Current number of dimensions.
    knots : callable or list of callables
        Knot generation functions.
    lev2knots : callable
        Level-to-knots mapping function.
    prev_adapt : types.SimpleNamespace or None
        Previous adaptation state (None for fresh start).
    controls : types.SimpleNamespace
        Control parameters.

    Returns
    -------
    tuple
        Initialized or resumed state of the adaptive algorithm.
    """
    # --> I         : is the set of explored indices (the grid is actually larger, it includes as well their neighbours)
    # --> I_log     : I must be sorted lexicographically for software reasons. I_log is the same set of indices, but
    #                   sorted in the order in which they are chosen by the algorithm
    # --> idx       : is the idx with the highest profit, whose neighbour we will next explore
    # --> maxprof   : the corresponding profit
    # --> idx_bin   : is the set of idx whose profit has been computed. They have been added to the grid but their neighbour is yet to be explored
    # --> profits   : is the corresponding set of profits
    # --> G         : is the set of the grid.
    # --> G_log     : same as I_log,  but for G
    # --> coeff_G   : coefficients of the combination technique applied to G
    # --> nb_pts    : the number of points in the grid
    # --> num_evals : the number of function evaluations
    # --> nb_pts_log: for each iteration, the current nb_pts
    # --> S         : create_sparse_grid_multiidx_set(G,knots,lev2knots);
    # --> Sr        : reduce_sparse_grid(S);
    # --> f_on_Sr   : evaluate_on_sparse_grid(f,Sr);
    # --> intf      : approx of integral of f using Sr
    # --> Hr        : all the points visited by the algo, stored as a reduced grid to be able to use ; only useful for non-nested grids, where it differs from Sr.
    # --> var_with_pts: vector of variables in which we have actually put points.
    #                   length(var_with_pts) + controls.var_buffer_size = N_curr
    # --> N_log     : for every iteration, the value of N_curr

    if prev_adapt is None:
        # it's a fresh start
        # --------------------------------------------

        var_with_pts = np.empty((0,), dtype=np.int64)   # we have put no points in no variables for now
        N_log = np.array([N], dtype=np.int64)

        I = np.zeros((1, N), dtype=np.int64)            # first explored index = [0,0,...,0]
        I_log = I.copy()
        idx = I[0, :].copy()
        maxprof = np.inf

        idx_bin = np.empty((0, N), dtype=np.int64)
        profits = np.empty((0,), dtype=float)

        G = I.copy()
        G_log = G.copy()
        coeff_G = np.array([1.0])

        S,_ = create_sparse_grid_multiidx_set(G, knots, lev2knots)
        Sr = reduce_sparse_grid(S, controls.pts_tol)
        f_on_Sr,*_ = evaluate_on_sparse_grid(f, None, Sr)  # here we don't need controls.pts_tol, there is no check on new/old points

        Hr = copy.copy(Sr)
        f_on_Hr = f_on_Sr.copy() # it is a matrix of size VxM where M is the number of points and f:R^N->R^V

        intf = f_on_Sr @ Sr.weights.T
        nb_pts = Sr.knots.shape[1]
        nb_pts_log = np.array([nb_pts], dtype=np.int64)
        num_evals = nb_pts

    else:
        # we are resuming from a previous run
        #--------------------------------------------
        logger.debug("adapt--recycling")

        N = prev_adapt.N
        N_log = prev_adapt.private.N_log
        var_with_pts = prev_adapt.private.var_with_pts
        I = prev_adapt.private.I
        I_log = prev_adapt.private.I_log
        idx = prev_adapt.private.idx
        maxprof = prev_adapt.private.maxprof
        idx_bin = prev_adapt.private.idx_bin
        profits = prev_adapt.private.profits
        G = prev_adapt.private.G
        G_log = prev_adapt.private.G_log
        coeff_G = prev_adapt.private.coeff_G
        S = prev_adapt.S
        Sr = prev_adapt.Sr
        f_on_Sr = prev_adapt.f_on_Sr
        Hr = prev_adapt.private.Hr
        f_on_Hr = prev_adapt.private.f_on_Hr
        intf = prev_adapt.intf
        nb_pts = prev_adapt.nb_pts
        nb_pts_log = prev_adapt.private.nb_pts_log
        num_evals = prev_adapt.num_evals

    return (
        N,
        N_log,
        var_with_pts,
        S,
        Sr,
        f_on_Sr,
        I,
        I_log,
        idx,
        maxprof,
        idx_bin,
        profits,
        G,
        G_log,
        coeff_G,
        Hr,
        f_on_Hr,
        nb_pts,
        nb_pts_log,
        num_evals,
        intf,
    )

def compute_profit_idx(ng_idx, f, S, T, Tr, Sr, Hr, f_on_Sr, f_on_Hr, intf, nb_pts, num_evals, nb_pts_log, knots, lev2knots, controls):
    """
    Compute the profit of a given index.

    Parameters
    ----------
    ng_idx : ndarray
        The index to compute the profit for.
    f : callable
        Function to approximate.
    S : list
        Current sparse grid.
    T : list
        New sparse grid with the index added.
    Tr : struct
        Reduced new sparse grid.
    Sr : struct
        Reduced current sparse grid.
    Hr : struct
        All points visited by the algorithm.
    f_on_Sr : ndarray
        Function evaluations on Sr.
    f_on_Hr : ndarray
        Function evaluations on Hr.
    intf : ndarray
        Approximation of the integral of f using Sr.
    nb_pts : int
        Current number of points in the grid.
    num_evals : int
        Current number of function evaluations.
    nb_pts_log : ndarray
        Log of the number of points at each iteration.
    knots : callable or list of callables
        Knot generation functions.
    lev2knots : callable
        Level-to-knots mapping function.
    controls : types.SimpleNamespace
        Control parameters.

    Returns
    -------
    tuple
        Updated number of points, number of function evaluations, log of points, profit, function evaluations on Tr, Hr, f_on_Hr, and integral approximation.
    """
    assert ng_idx.ndim==1
    N = ng_idx.shape[0]
    Tr_on_new_pts = None
    if controls.nested:
        # here we evaluate on new points only. Note that finding which points have been evaluated already
        # relies on multiindex info almost exclusively (because the points are nested) so this is quite efficient
        f_on_Tr, new_points, idx_newp, *_ = evaluate_on_sparse_grid(
            f, T, Tr, f_on_Sr, S, Sr, controls.pts_tol
        )
        intnew = f_on_Tr @ Tr.weights.T

        newp = idx_newp.size
        nb_pts += newp
        nb_pts_log = np.append(nb_pts_log, nb_pts)
        num_evals = nb_pts
        Hr = None
        f_on_Hr = None

    else:
        # in this case, we need to keep track of all the points explored, even those that have been discarded in
        # previous iterations
        if controls.recycling == "priority_to_evaluation":
            # here we allow for multiple evaluations of the same point because we recycle from the previous grid only.
            # if the function evaluation is "cheap" this is much faster, because the search for common points relies
            # on multiindices and not on comparison of coordinates
            f_on_Tr, _, idx_newp, *_ = evaluate_on_sparse_grid(
                f,
                T,
                Tr,
                f_on_Sr,
                S,
                Sr,
                controls.pts_tol,
            )
            intnew = f_on_Tr @ Tr.weights.T

            Hr.knots = np.hstack([Hr.knots, Tr.knots[:, idx_newp]])
            f_on_Hr = np.hstack([f_on_Hr, f_on_Tr[:, idx_newp]])

            #newp = np.prod(lev2knots(ng_idx))
            newp = np.prod([lev2knots(ng_idx[i]+1) for i in range(N)])
            nb_pts = Tr.knots.shape[1]
            num_evals += len(idx_newp)
            nb_pts_log = np.append(nb_pts_log, nb_pts)

        else:
            # (slow)
            # here we want to make sure no multiple evaluations of the same point occur. Thus we look in
            # all points ever visited, but this is expensive because we rely on point coordinates only!
            f_on_Tr, _, idx_newp, *_ = evaluate_on_sparse_grid(
                f,
                T,
                Tr,
                f_on_Hr,
                None,
                Hr.knots,  # this is not an error
                controls.pts_tol,
            )
            intnew = f_on_Tr @ Tr.weights.T

            Hr.knots = np.hstack([Hr.knots, Tr.knots[:, idx_newp]])
            f_on_Hr = np.hstack([f_on_Hr, f_on_Tr[:, idx_newp]])

            #newp = np.prod(lev2knots(ng_idx))
            newp = np.prod([lev2knots(ng_idx[i]+1) for i in range(N)])
            nb_pts = f_on_Tr.shape[1]
            num_evals = Hr.knots.shape[1]
            nb_pts_log = np.append(nb_pts_log, nb_pts)

        # moreover, if profit is of type Linf, we need to evaluate the new grid on the ``nominally new points'',
        if controls.prof in {
            "Linf/new_points",
            "Linf",
            "weighted Linf/new_points",
            "weighted Linf",
        }:
            #Tx = tensor_grid(N, lev2knots(ng_idx), knots)
            Tx = tensor_grid(N, [lev2knots(ng_idx[i]+1) for i in range(N)], knots)
            new_points = Tx.knots
            Tr_on_new_pts = interpolate_on_sparse_grid(T, Tr, f_on_Tr, new_points)
        #elif controls.prof in ['deltaint/new_points', 'deltaint']:
            # no need of new points
        else:
            raise ValueError('do we need new points in this case? fix code here')

    prof_type = controls.prof
    op_vect = controls.op_vect

    if prof_type == "Linf/new_points":
        Sr_on_new_pts = interpolate_on_sparse_grid(S, Sr, f_on_Sr, new_points)
        if controls.nested:
            Prof_temp = np.max(op_vect(f_on_Tr[:, idx_newp], Sr_on_new_pts)) / newp
        else:
            Prof_temp = np.max(op_vect(Tr_on_new_pts, Sr_on_new_pts)) / newp

    elif prof_type == "Linf":
        Sr_on_new_pts = interpolate_on_sparse_grid(S, Sr, f_on_Sr, new_points)
        if controls.nested:
            Prof_temp = np.max(op_vect(f_on_Tr[:, idx_newp], Sr_on_new_pts))
        else:
            Prof_temp = np.max(op_vect(Tr_on_new_pts, Sr_on_new_pts))

    elif prof_type == "deltaint/new_points":
        delta_int = op_vect(intnew, intf)
        Prof_temp = delta_int / newp

    elif prof_type == "deltaint":
        delta_int = op_vect(intnew, intf)
        Prof_temp = delta_int

    elif prof_type == "weighted Linf/new_points":
        Sr_on_new_pts = interpolate_on_sparse_grid(S, Sr, f_on_Sr, new_points)
        if controls.nested:
            Prof_temp = (
                np.max(
                    op_vect(f_on_Tr[:, idx_newp], Sr_on_new_pts)
                    * controls.pdf(new_points)
                )
                / newp
            )
        else:
            Prof_temp = (
                np.max(
                    op_vect(Tr_on_new_pts, Sr_on_new_pts)
                    * controls.pdf(new_points)
                )
                / newp
            )

    elif prof_type == "weighted Linf":
        Sr_on_new_pts = interpolate_on_sparse_grid(S, Sr, f_on_Sr, new_points)
        if controls.nested:
            Prof_temp = np.max(
                op_vect(f_on_Tr[:, idx_newp], Sr_on_new_pts)
                * controls.pdf(new_points)
            )
        else:
            Prof_temp = np.max(
                op_vect(Tr_on_new_pts, Sr_on_new_pts)
                * controls.pdf(new_points)
            )
    else:
        raise ValueError("unknown profit indicator. Check spelling")

    return nb_pts, num_evals, nb_pts_log, Prof_temp, f_on_Tr, Hr, f_on_Hr, intnew
