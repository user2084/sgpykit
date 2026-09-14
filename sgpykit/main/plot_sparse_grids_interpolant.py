import logging
import numpy as np

from sgpykit.main.interpolate_on_sparse_grid import interpolate_on_sparse_grid
from sgpykit.util.misc import matlab_to_python_index, merge_all_args_to_kwargs
from sgpykit.util.plot import check_and_convert_to_fig3d

logger = logging.getLogger(__name__)


def plot_sparse_grids_interpolant(ax, S, Sr, domain, f_values, *args, **kwargs):
    """
    Plot the sparse grid interpolant of a function.

    Different plots are produced depending on the number of dimensions of the sparse grid:
    - If N==2, a surf plot will be generated.
    - If N==3, a number of contourfs (i.e., flat surfaces colored according to the value of the interpolant)
      will be stacked over the same axes.
    - If N>3, a number of bidimensional cuts will be considered, and for each of them a surf will be generated.
      In other words, all variables but two will be frozen to their average value and the resulting
      two-dimensional plot will be produced.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    S : object
        The sparse grid.
    Sr : object
        The reduced sparse grid.
    domain : numpy.ndarray
        A matrix 2 x N, describing the domain where the sparse grids should be plotted.
        The first row contains the lower bounds, the second row the upper bounds.
    f_values : numpy.ndarray
        The function values at the sparse grid points.
    *args : tuple
        Additional arguments to control the behavior of the plots.
    **kwargs : dict
        Additional keyword arguments to control the behavior of the plots.

    Returns
    -------
    matplotlib.collections.PolyCollection or matplotlib.collections.LineCollection
        The plot handle.

    Notes
    -----
    Additional inputs can be passed to control the behavior of the plots. Any combination of these optional
    inputs is allowed:
    - 'with_f_values': Adds dots with the values of the sparse grids interpolant to the plots above (case N=2 and N>3).
      For N==3, adds the sparse grid points in the 3D plot.
    - 'nb_plot_pts': Sets the number of points used in each direction for the surf/contourf plots (default 20).
    - 'nb_contourfs': Sets the number of contourfs in the vertical direction for the case N=3 (default 5).
    - 'nb_contourf_lines': Sets the number of contourf lines (default 10).
    - 'two_dim_cuts': Specifies the couples of variables to consider for the two-dimensional cuts when N>3.
      C is a vector with 2*k components denoting the directions of the cuts.
      For instance, the default value is C = [0, 1, 2, ...] and produces cut plots for (y1,y2)
      (y3,y4), (y5,y6).
    """
    N = domain.shape[2-1]
    h = None
    args_list = list(args)  # Convert args to a list for easier manipulation
    with_f_values = 'with_f_values' in args_list

    if with_f_values:
        args_list.remove('with_f_values') # remove it so we can do our args to kwargs merge
    # merge
    matlab_kwargs, _ = merge_all_args_to_kwargs(args_list, kwargs, to_lowercase=True)
    # value of 'plot_grid_size'
    NP = int(matlab_kwargs.get('nb_plot_pts', 20))
    # value of 'nb_contourfs'
    NC = int(matlab_kwargs.get('nb_contourfs', 5 if N==3 else -1)) # TODO: check -1

    if NC == -1:
        logger.warning('ignoring nb_contourfs input')

    # value of 'nb_contour_lines'
    NL = int(matlab_kwargs.get('nb_contourf_lines', 10 if N==3 else -1)) # TODO: check -1

    if NL == -1:
        logger.warning('ignoring nb_contourf_lines input')

    # value of 'two_dim_cuts'
    couples = matlab_kwargs.get('two_dim_cuts', np.arange(1,N+1) if N>3 else -1) # TODO: check -1, arange

    if isinstance(couples,int) and couples == -1:
        logger.warning('ignoring two_dim_cuts input')

    # extract info on lower and upper ends of each direction
    aa_vec = domain[0,:]
    bb_vec = domain[1,:]
    avg_vec = (aa_vec + bb_vec) / 2
    # wrap interpolate on sparse grid into a @-function for ease of plotting
    f_interp = lambda x: interpolate_on_sparse_grid(S, Sr, f_values, x)
    if 2 == N:
        # generate a mesh grid over the cut
        xp = np.linspace(aa_vec[0],bb_vec[0],NP)
        yp = np.linspace(aa_vec[1],bb_vec[1],NP)
        XP,YP = np.meshgrid(xp,yp, indexing='xy')
        nb_pts = xp.size * yp.size
        PTS = np.zeros((2,nb_pts))
        PTS[0,:] = XP.ravel()
        PTS[1,:] = YP.ravel()
        # interpolate on sparse grid
        f_interp_eval = f_interp(PTS)
        # reshape to use surf
        FIP = f_interp_eval.reshape(XP.shape)
        h = ax.plot_surface(XP,YP,FIP, cmap='viridis')
        ax.set_xlabel('y_1')
        ax.set_ylabel('y_2')
        if with_f_values:
            h = ax.plot(Sr.knots[0,:], Sr.knots[1,:], f_values, 'or')
    else:
        if 3 == N:
            ax = check_and_convert_to_fig3d(ax)
            # generate a mesh grid over the cut
            xp = np.linspace(aa_vec[0],bb_vec[0],NP)
            yp = np.linspace(aa_vec[1],bb_vec[1],NP)
            zp = np.linspace(aa_vec[2],bb_vec[2],NC)
            XP,YP = np.meshgrid(xp,yp,indexing='xy')
            XP_vect = XP.flatten()
            YP_vect = YP.flatten()
            PTS_XY = np.vstack((XP_vect, YP_vect))
            nb_pts = PTS_XY.shape[1]
            # Loop over each z level
            for z_lev in range(len(zp)):
                PTS_Z = zp[z_lev] * np.ones(nb_pts)
                PTS = np.vstack((PTS_XY, PTS_Z))
                f_interp_eval = f_interp(PTS)
                FIP = f_interp_eval.reshape(XP.shape)
                h = ax.contourf(XP, YP, FIP, NL, zdir='z', offset=zp[z_lev], cmap='viridis')

            #h=o2;
            #ax.view_init(elev=20, azim=-30)
            ax.set_xlabel('y_1')
            ax.set_ylabel('y_2')
            ax.set_zlabel('y_3')
            ax.set_zlim((0, np.max(zp)))
            if with_f_values:
                for pp in np.arange(Sr.knots.shape[1]):
                    h = ax.plot(Sr.knots[0,pp],Sr.knots[1,pp],Sr.knots[2,pp], 'ok', markersize=8, markerfacecolor='r', zorder=100)

            # ax.figure.colorbar(contour, ax=ax, shrink=0.5, aspect=5) # TODO: require global coloring over all z_lev planes
        else:
            CUTS = len(couples) // 2
            #h = zeros(1,CUTS);
            for ii in range(CUTS):
                axi = ax.flatten()[ii]
                couple_loc = [couples[2*ii], couples[2*ii+1]]
                v1 = matlab_to_python_index(couple_loc[0])
                v2 = matlab_to_python_index(couple_loc[1])
                # generate a mesh grid over the cut
                xp = np.linspace(aa_vec[v1],bb_vec[v1],NP)
                yp = np.linspace(aa_vec[v2],bb_vec[v2],NP)
                XP, YP = np.meshgrid(xp, yp)
                nb_pts = len(xp) * len(yp)
                # we need to generate the matrix of points where we want to evaluate our interpolant.
                # As usual, yt will be a fat matrix with points stored as columns. All directions
                # will be frozen to their average value, but the two of the local cut
                # (i.e., all rows but two will be constant).
                # We begin by making it all constant rows and then changing the rows we need
                # PTS = [ avg_dir1; avg_dir2; avg_dir3 ...]
                # Generate the matrix of points where we want to evaluate our interpolant
                PTS = np.empty((avg_vec.size, nb_pts), dtype=avg_vec.dtype)
                PTS[:] = avg_vec[:, None]  # broadcasts into the buffer
                # Replace lines of non-constant directions
                PTS[v1, :] = XP.ravel()
                PTS[v2, :] = YP.ravel()
                # Interpolate on sparse grid
                f_interp_eval = f_interp(PTS)
                # Reshape to use surf
                FIP = f_interp_eval.reshape(XP.shape)
                #h(ii)=figure;
                h = axi.plot_surface(XP, YP, FIP, cmap='viridis')
                if with_f_values:
                    h = axi.plot(Sr.knots[v1,:], Sr.knots[v2,:], f_values, 'ok', markersize=4, markerfacecolor='r',zorder=100)
                axi.set_title(f"cut {ii+1} of {CUTS} over directions {v1+1} and {v2+1}")
                axi.set_xlabel(f'y_{v1}')
                axi.set_ylabel(f'y_{v2}')

    return h

# def value_of(string = None,cell = None):
#     # if string is found in cell, return the value in the next cell
#
#     # logical array, 1 if string is found in cell
#     found = cellfun(lambda in_ = None: str(in_) == str(string),cell)
#     if np.any(found):
#         # find the location of 1 in found
#         pos = find(found)
#         # the next input is our guy
#         v = cell[pos + 1]
#     else:
#         v = []
#
#     return v
