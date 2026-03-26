import pytest
import numpy as np
from scipy.stats import norm
from types import SimpleNamespace

import sgpykit as sg
from sgpykit.util.checks import is_list_math_equal
from sgpykit.util.misc import matlab_to_python_index


@pytest.mark.parametrize(
    "f, knots_f, exp_bufint, exp_nobufint, exp_oneint",
    [
        pytest.param(lambda x: 1.0 / np.exp(sum(x)),
                     [
                         lambda n: sg.knots_CC(n, -0.5, 0.5),
                         lambda n: sg.knots_CC(n, -0.5, 0.5),
                         lambda n: sg.knots_CC(n, -0.2, 0.2)
                     ], 1.09341684058, 1.09341684058, 1.09341684058
            , id="1"),
        pytest.param(lambda x: 1.0 / np.exp(sum(x)),
                     [lambda n: sg.knots_CC(n, 0, 1)] * 3,
                     0.138807930001, 0.252580457688, 0.252580457688,
                     id="2"),
        pytest.param(lambda x: np.prod(x, axis=0),
                     [lambda n: sg.knots_CC(n, 0, 1)] * 3,
                     0.0367666236905, 0.125, 0.125,
                     id="3"),
        pytest.param(lambda x: sum(np.cos(x)),
                     [lambda n: sg.knots_CC(n, -1, 1)] * 3,
                     3.74763058531, 2.52441295442, 2.52441295442,
                     id="4"),
        pytest.param(lambda x: np.prod(x, axis=0),
                     [lambda n: sg.knots_CC(n, 0, 2)] * 3,
                     1,1,1,
                     id="5"),
    ]
)
def test_adapt_sparse_grid_examples(f, knots_f, exp_bufint, exp_nobufint, exp_oneint):
    N = 3
    lev2knots = sg.lev2knots_doubling

    controls = SimpleNamespace(
        max_pts=200,
        prof_tol=1e-10,
        plot=False,
        var_buffer_size=2,  # buffer version
        nested=True
    )
    prev_adapt = None

    # ---- with buffer ------------------------------------------------
    adapt_buff = sg.adapt_sparse_grid(f, N, knots_f, lev2knots,
                                      prev_adapt, controls)

    # ---- without buffer (var_buffer_size = N) -----------------------
    controls.var_buffer_size = N
    adapt_no_buff = sg.adapt_sparse_grid(f, N, knots_f, lev2knots,
                                         prev_adapt, controls)

    # ---- one‑shot construction (no adapt) ---------------------------
    G = adapt_no_buff.private.G
    S, _ = sg.create_sparse_grid_multiidx_set(G, knots_f, lev2knots)
    Sr = sg.reduce_sparse_grid(S)
    Q, _ = sg.quadrature_on_sparse_grid(f, S=None, Sr=Sr)

    assert np.isclose(adapt_buff.intf[0], exp_bufint)
    assert np.isclose(adapt_no_buff.intf[0], exp_nobufint)
    assert np.isclose(Q[0], exp_oneint)


def test_adapt_sparse_grid():
    # function to integrate / interpolate
    f = lambda x: 1.0 / (x[0] ** 2 + x[1] ** 2 + 0.3)

    N = 2
    a, b = -1.0, 1.0

    # knot generator (Clenshaw‑Curtis on [a,b])
    knots = lambda n: sg.knots_CC(n, a, b)

    # level‑to‑knots mapping (doubling rule)
    lev2knots = sg.lev2knots_doubling

    controls = SimpleNamespace(
        max_pts=20,
        prof_tol=1e-10,
        nested=True,
        plot=False
    )
    prev_adapt = None

    adapt1 = sg.adapt_sparse_grid(f, N, knots, lev2knots, prev_adapt, controls)
    assert adapt1.N == 2
    assert adapt1.nb_pts == 21
    assert adapt1.nb_pts_visited == 21
    assert adapt1.nested == 1

    S = adapt1.S
    assert is_list_math_equal(S.size, [5,9,3,15,5])
    assert is_list_math_equal(S.m, [
        [1, 5],
        [1, 9],
        [3, 1],
        [3, 5],
        [5, 1]
    ])
    assert is_list_math_equal(S.knots, [
        [[0, 0, 0, 0, 0],
         [1.0, 0.707106781186548, 0.0, -0.707106781186547, -1.0]],

        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1.0, 0.923879532511287, 0.707106781186548, 0.382683432365090, 0.0, -0.382683432365090, -0.707106781186547, -0.923879532511287, -1.0]],

        [[1.0, 0.0, -1.0],
         [0, 0, 0]],

        [[1.0, 6.123233995736766e-17, -1.0, 1.0, 6.123233995736766e-17, -1.0, 1.0, 6.123233995736766e-17,
          -1.0, 1.0, 6.123233995736766e-17, -1.0, 1.0, 6.123233995736766e-17, -1.0],
         [1.0, 1.0, 1.0, 0.7071067811865476, 0.7071067811865476, 0.7071067811865476,
          6.123233995736766e-17, 6.123233995736766e-17, 6.123233995736766e-17,
          -0.7071067811865475, -0.7071067811865475, -0.7071067811865475,
          -1.0, -1.0, -1.0]],

        [[1.0, 0.707106781186548, 0.0, -0.707106781186547, -1.0],
         [0, 0, 0, 0, 0]]
    ])
    # NOTE: not testing every field here

    Sr = adapt1.Sr
    knots_expected = [
        [-1.000000000000000, -1.000000000000000, -1.000000000000000, -1.000000000000000, -1.000000000000000,
         -0.707106781186547, 0.000000000000000,
         0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000,
         0.000000000000000, 0.000000000000000,
         0.000000000000000, 0.707106781186548, 1.000000000000000, 1.000000000000000, 1.000000000000000,
         1.000000000000000, 1.000000000000000],
        [-1.000000000000000, -0.707106781186547, 0.000000000000000, 0.707106781186548, 1.000000000000000, 0,
         -1.000000000000000,
         -0.923879532511287, -0.707106781186547, -0.382683432365090, 0.000000000000000, 0.382683432365090,
         0.707106781186548, 0.923879532511287,
         1.000000000000000, 0, -1.000000000000000, -0.707106781186547, 0.000000000000000, 0.707106781186548,
         1.000000000000000]
    ]
    m_expected = matlab_to_python_index([32, 29, 26, 23, 20, 36, 31, 13, 28, 11, 25, 9, 22, 7, 19, 34, 30, 27, 24, 21, 18])

    weights_expected = [5.555555555555559e-03, 4.444444444444445e-02, -6.666666666666665e-02, 4.444444444444445e-02, 5.555555555555559e-03, 2.666666666666667e-01,
    -3.174603174603181e-03, 7.310932460800906e-02, 5.079365079365081e-02, 1.808589293602449e-01, -2.031746031746032e-01, 1.808589293602449e-01,
    5.079365079365081e-02, 7.310932460800906e-02, -3.174603174603181e-03, 2.666666666666667e-01, 5.555555555555559e-03, 4.444444444444445e-02,
    -6.666666666666665e-02, 4.444444444444445e-02, 5.555555555555559e-03]
    size_expected = 21
    n_expected = matlab_to_python_index([15,13,11, 9, 7,15,14,13,12,11,10, 9, 8, 7,19,11, 3,21,15, 5,20,13, 4,19,11, 3,18, 9, 2,17, 7, 1,19,16,11, 6, 3])
    assert is_list_math_equal(Sr.knots, knots_expected)
    assert is_list_math_equal(Sr.weights, weights_expected)
    assert Sr.size == size_expected
    assert is_list_math_equal(Sr.n, n_expected)
    assert is_list_math_equal(Sr.m, m_expected)
    assert np.isclose(adapt1.intf, 1.054351534160174)

    assert np.allclose(adapt1.f_on_Sr,  np.array([
        0.434782608695652, 0.555555555555556, 0.769230769230769, 0.555555555555556, 0.434782608695652, 1.250000000000000, 0.769230769230769, 0.866886620207235, 1.250000000000000,
        2.239909496297619, 3.333333333333333, 2.239909496297618, 1.250000000000000, 0.866886620207235, 0.769230769230769, 1.250000000000000, 0.434782608695652, 0.555555555555556,
        0.769230769230769, 0.555555555555556, 0.434782608695652
    ]))

    private = adapt1.private
    assert np.isclose(private.maxprof, 0.400641025641026)
    assert np.allclose(private.profits, np.array([1.887077294685992e-01, 7.931967117769390e-02]))
    assert is_list_math_equal(private.idx, [2,0])
    assert is_list_math_equal(private.idx_bin, np.array([[1, 2],[0,3]]))
    assert is_list_math_equal(private.I_log, matlab_to_python_index(np.array([[1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [1, 3],
    [3, 1]])))


def test_adapt_sparse_grid_pdf():
    f = lambda x: 1.0/(2 + np.exp(x[0]) + np.exp(x[1]))
    N = 2

    knots = lambda n: sg.knots_GK(n, 0, 1)          # Gauss‑Kronrod on [0,1]
    lev2knots = sg.lev2knots_GK
    controls = SimpleNamespace(
        max_pts=150,
        prof_tol=1e-10,
        prof="weighted Linf/new_points",
        nested=True,
        plot=False,
        # Gaussian pdf (product of 1‑D normals)
        pdf=lambda Y: np.prod(norm.pdf(Y, 0, 1), axis=0)
    )
    prev_adapt = None
    adapt1 = sg.adapt_sparse_grid(f, N, knots, lev2knots, prev_adapt, controls)
    pytest.skip("TODO: implement test assertions")


def test_adapt_sparse_grid_lin_prior_eval():
    f = lambda x: 1.0/(2 + np.exp(x[0]) + np.exp(x[1]))
    N = 2
    controls = SimpleNamespace(
        max_pts=150,
        prof_tol=1e-10,
        prof="weighted Linf/new_points",
        nested=False,
        plot=False,
        # Gaussian pdf (product of 1‑D normals)
        pdf=lambda Y: np.prod(norm.pdf(Y, 0, 1), axis=0)
    )
    knots = lambda n: sg.knots_normal(n, 0, 1)  # Gauss‑Hermite (non‑nested)
    lev2knots = sg.lev2knots_lin
    prev_adapt = None
    adapt2 = sg.adapt_sparse_grid(f, N, knots, lev2knots, prev_adapt, controls)
    pytest.skip("TODO: implement test assertions")


def test_adapt_sparse_grid_lin_prior_recycling():
    f = lambda x: 1.0 / (2 + np.exp(x[0]) + np.exp(x[1]))
    N = 2
    controls = SimpleNamespace(
        max_pts=150,
        prof_tol=1e-10,
        prof="weighted Linf/new_points",
        nested=False,
        plot=False,
        recycling="priority_to_recycling",
        # Gaussian pdf (product of 1‑D normals)
        pdf=lambda Y: np.prod(norm.pdf(Y, 0, 1), axis=0)
    )
    knots = lambda n: sg.knots_normal(n, 0, 1)  # Gauss‑Hermite (non‑nested)
    lev2knots = sg.lev2knots_lin
    prev_adapt = None
    adapt3 = sg.adapt_sparse_grid(f, N, knots, lev2knots, prev_adapt, controls)
    pytest.skip("TODO: implement test assertions")


def test_adapt_sparse_grid_4d():
    f = lambda x: 1.0 / (x[0] ** 2 + x[1] ** 2 + 0.3 + 0.1 * np.sin(x[2]) * np.exp(0.4 * x[3]))
    N = 4
    a, b = -1.0, 1.0
    knots = lambda n: sg.knots_CC(n, a, b)
    lev2knots = sg.lev2knots_doubling

    controls = SimpleNamespace(
        max_pts=400,
        prof_tol=1e-10,
        nested=True
    )
    prev_adapt = None
    adapt1 = sg.adapt_sparse_grid(f, N, knots, lev2knots, prev_adapt, controls)
    pytest.skip("TODO: implement test assertions")

