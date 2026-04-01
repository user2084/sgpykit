# sgpykit

**sgpykit** is a Python library for constructing, manipulating, evaluating and analysing sparse‑grid surrogate models.
It implements a Python version of [Sparse Grids Matlab Kit](https://sites.google.com/view/sparse-grids-kit) (SGMK).

## **Key Features**

### **1. Sparse Grid Construction**
- Supports multiple sparse grid rules (e.g., **Total Degree (TD)**, **Hyperbolic Cross (HC)**, **Smolyak (SM)**).
- Works with various knot types (e.g., **Clenshaw-Curtis, Gauss-Patterson, Leja, Chebyshev, Legendre, Hermite, Laguerre**).
- Allows custom level-to-knot mappings (linear, doubling, tripling, etc.).

### **2. Adaptive Sparse Grids**
- Implements **dimension-adaptive refinement** to dynamically refine grids where needed.
- Optimizes computational efficiency by focusing on the most relevant regions of the domain.

### **3. Function Evaluation & Interpolation**
- Efficiently evaluates functions on sparse grids.
- Supports **interpolation, quadrature, and derivative computations** (gradients, Hessians).
- Handles both **scalar and vector-valued functions**.

### **4. Modal & Polynomial Representations**
- Converts sparse grid interpolants into **orthogonal polynomial expansions** (Legendre, Chebyshev, Hermite, etc.).
- Enables **generalized Polynomial Chaos Expansions (gPCE)** for uncertainty quantification.

### **5. Visualization & Analysis**
- Includes tools for **plotting sparse grids** in 2D and 3D.
- Computes **Sobol indices** for sensitivity analysis.
- Provides utilities for comparing and validating sparse grids.

### **Use Cases**
- **Uncertainty Quantification (UQ)** – Efficiently compute statistics of high-dimensional models.
- **Numerical Integration** – Perform high-dimensional quadrature with fewer evaluations.
- **Interpolation & Surrogate Modeling** – Approximate complex functions with sparse grids.
- **Sensitivity Analysis** – Compute Sobol indices to identify influential parameters.

## sgpykit Qickstart

Import sgpykit (install via github: `pip install git+https://github.com/uncertaintyhub/sgpykit` or see below):

```python
import sgpykit as sg
```

and start working with sparse grids like `sg.create_sparse_grid(...)`.

- Tutorials are provided in [Sparse Grids Tutorial](./docs-examples/sparse_grids_tutorial.ipynb) and [Adaptive Grids Tutorial](./docs-examples/tutorial_adaptive.ipynb).
- sgpykit mainly uses numpy as backend. See [differences.md](./differences.md) for more information (relevant for MATLAB users).
- The online code documentation can be found on: <https://uncertaintyhub.github.io/sgpykit-doc/>


## Installation

The library has no compiled extensions and relies only on standard scientific Python packages (NumPy, SciPy and Matplotlib). 
A typical installation workflow is:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # on Windows use `venv\Scripts\activate`

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install numpy scipy matplotlib
```

- Install sgpykit locally: `pip install -e .` (`-e` just makes a link to the source path instead of copying it)
- Or install via github: `pip install git+https://github.com/uncertaintyhub/sgpykit`
- Or install via PyPI: Not available yet

Installation targets `[doc,test]` exist for installing optional dependencies (`notebook pytest-benchmark sphinx`).

If you prefer to use the package without installation, simply add the top‑level directory to your ``PYTHONPATH`` and import it as shown above.

## Usage example

The following snippet creates a 2‑dimensional Smolyak sparse grid with Clenshaw‑Curtis points, 
evaluates a simple function on the grid, and then interpolates the surrogate at a set of random points.

```python
import numpy as np
import sgpykit as sg

# -------------------------------------------------
# 1. Define the 1‑D knot generator (Clenshaw‑Curtis)
# -------------------------------------------------
knots = lambda n: sg.knots_CC(n, -1, 1, 'nonprob')

# -------------------------------------------------
# 2. Build a sparse grid of level w = 3 in N = 2 dimensions
# -------------------------------------------------
N = 2  # number of variables
w = 3  # level (total degree)
# second return argument is unused (would contain multi-index set for the grid)
S,_ = sg.create_sparse_grid(N=N, w=w, knots=knots,
                            lev2knots=sg.lev2knots_doubling)
Sr = sg.reduce_sparse_grid(S)

# -------------------------------------------------
# 3. Evaluate a target function on the grid points
# -------------------------------------------------
fs = lambda x: np.exp(sum(x)) # or np.sum(x,axis=0) instead of sum(x)

# evaluate on the *reduced* grid (faster, no duplicate points)
y,*_ = sg.evaluate_on_sparse_grid(fs, S=None, Sr=Sr)
function_on_grid = fs(Sr.knots)

# -------------------------------------------------
# 4. Interpolate the surrogate at new locations
# -------------------------------------------------
Mnew = 5
Xnew = np.random.uniform(-1, 1, size=(N, Mnew))
y_pred = sg.interpolate_on_sparse_grid(S, Sr, function_on_grid, Xnew)

print('Interpolated values:', y_pred)
```

Other common workflows (adaptive refinement, Sobol index computation, modal conversion, derivative estimation) are built from the same set of core functions. 
See the Jupyter notebooks [Sparse Grids Tutorial](./docs-examples/sparse_grids_tutorial.ipynb) and [Adaptive Grids Tutorial](./docs-examples/tutorial_adaptive.ipynb) for more examples.


## Jupyter Notebook

### Installation

```bash
git clone https://github.com/uncertaintyhub/sgpykit
cd sgpykit
# Create a virtual environment if needed.
python -m venv .venv
source .venv/bin/activate
# install packages including notebook (Jupyter)
pip install numpy matplotlib scipy notebook
```

### Run

```bash
source .venv/bin/activate # if virtual env is used
jupyter notebook # opens the browser with Jupyter
```

In the Jupyter file-browser navigate to `/docs-examples` and open a notebook like `sparse_grids_tutorial.ipynb`.


## Tensor Grids Layout

Each tensor grid structure contains the following fields (also see [SGMK manual online](https://sites.google.com/view/sparse-grids-kit#h.p_IBOqq8I5k39K)):

- **idx**: the 0-based multi-index $\mathbf{i} \in \mathcal{I}\subset\mathbb{N}^N$ corresponding to the current tensor grid
- **knots**: matrix collecting the knots $\mathcal{T}_{\mathbf{i}}$, each knot being a row vector
- **weights**: vector of the quadrature weights $\omega_{m(\mathbf{i})}^{(\mathbf{j})}$ corresponding to the knots
- **size**: size of the tensor grid, i.e. the number of knots $M_{\mathbf{i}} = \prod_{n=0}^{N-1} m(i_n)$
- **knots_per_dim**: cell array with $N$ components, each component collecting in an array the set of one-dimensional knots $\mathcal{T}_{n, i_n}$ used to build the tensor grid
- **m**: vector collecting the number of knots used in each of the $N$ directions $m(\mathbf{i}) = [m(i_0), m(i_1), \ldots, m(i_{N-1})]$
- **coeff**: the coefficients $c_{\mathbf{i}}$ of the sparse grid in the combination technique formulas

Note that sparse grid knots (and in general knots $\mathbf{y} \in \Gamma$) are always stored in sgpykit as row vectors 
(and sets of knots such as $\mathbf{S}.\text{knots}$ are stored as matrices where knots are rows).

Reduced sparse grids contain the following fields:

- **knots**: matrix collecting the list of non-repeated knots, i.e., the set $\mathcal{T}_{\mathcal{I}}$
- **weights**: vector of quadrature weights corresponding to the knots above
- **size**: size of the sparse grid, i.e. the number of non-repeated knots
- **m**: 0-based index array that maps each knot of $\mathbf{Sr}.\text{knots}$ to their original position in $\mathbf{S}.\text{knots}$ (if they have been retained as unique representative of several repeated knots)
- **n**: 0-based index array that maps each knot of $\mathbf{S}.\text{knots}$ to $\mathbf{Sr}.\text{knots}$

In sgpykit these grids are stored as customizable Structure Arrays, where additional fields can be added by the user.

## Development Status

This implementation is currently in **Alpha** and the API is not yet finalized. Expect potential changes in future releases.

**Key Areas for Improvement:**
- **Testing:** More comprehensive test coverage is needed. Also looking for case studies.
- **Performance:** A performance analysis is planned to enhance efficiency. User reports are very welcome.
- **Streamlining Code:** The current implementation retains some legacy structures from its direct port from **SGMK (MATLAB)**. Future releases will focus on:
  - Reducing unnecessary overhead.
  - Improving Pythonic efficiency while maintaining familiarity for **SGMK** users.

**Code & Documentation Notes:**
- Many internal comments are just a copy from the original **SGMK** (MATLAB) codebase.
- Function docstrings are written from a Python perspective but were initially generated with LLMs.
  The type information can be incorrect.
  A thorough review and refinement of documentation is planned. 

The visualisation part is not in the focus, but the common plot functionality from SGMK is already supported
(using matplotlib as backend).

## Contributing

Contributions are welcome. If you wish to add new code, improve documentation, or fix bugs:

1. Fork the repository.
2. Create a feature branch (`git checkout -b my‑feature`).
3. Ensure that existing tests pass and add tests for new functionality. Run `pytest` in the sgpykit root directory, for example: 
   - `PYTHONPATH=$(pwd) pytest --benchmark-skip -v tests` for a complete run of all tests
   - `PYTHONPATH=$(pwd) pytest -v ./tests/sgpykit/main/test_interpolate_on_sparse_grid.py` for a single test run
4. Submit a pull request with a clear description of the changes.

The package is organized like SGMK with a hierarchy of sub‑packages that separate low‑level algorithmic building blocks:

- **main**: This subpackage contains the core functions for creating, manipulating, and evaluating sparse grids, including adaptive refinement, interpolation, and visualization.
- **src**: This subpackage provides utility functions and core algorithms for sparse grid operations, such as combination technique, grid comparison, and modal tensor computation.
- **tools**: This subpackage offers a variety of helper functions and utilities, including polynomial evaluations, knot generation, and type checking for sparse grid operations.

There are also helper utilities (`sgpykit.util`). 

## Acknowledgments

Special thanks to the developers of the original MATLAB toolbox Sparse Grids Matlab Kit for their great work and for all their support on sgpykit.
