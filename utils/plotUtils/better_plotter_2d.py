import numpy as np

from .binning_tools import get_bin_centers, get_bin_widths, get_bin_line
from .histogram2d import Histo2D
from .graph2d import Graph2D
from .formater import format_axes

from .better_plotter import get_figax, execute

import matplotlib as mpl

import scipy.interpolate as interp



def get_cmap_kwargs(zlim=None, cmap='YlOrRd', log=False, **kwargs):
    cmap_kwargs = dict(
        cmap = cmap
    )

    if zlim is None: return cmap_kwargs

    if len(zlim) == 2:
        cmap_kwargs['vmin'] = zlim[0]
        cmap_kwargs['vmax'] = zlim[1]

    elif len(zlim) > 2:
        if isinstance(cmap, str):
            cmap=mpl.cm.get_cmap(cmap)

        if log:
            norm = mpl.colors.LogNorm(vmin=np.min(zlim), vmax=np.max(zlim) )
        else:
            norm = mpl.colors.BoundaryNorm(boundaries=np.array(zlim), ncolors=cmap.N)

        cmap_kwargs['norm'] = norm

    return cmap_kwargs

_offset = 0.2
offsets = {
    2:[(-_offset,-_offset),(_offset, _offset)],
    3:[(-_offset,-_offset),(_offset,-_offset),(0      ,_offset)],
    4:[(-_offset,-_offset),(_offset,-_offset),(_offset,_offset),(-_offset,_offset)],
}
def scatter_histo2d(histo2d, figax=None, min_counts=10000, fraction=10000, alpha=0.25, size=5, discrete=False, discrete_offset=None, discrete_scale=0.5, **kwargs):
    fig, ax = get_figax(figax=figax)
    color = histo2d.kwargs.get('color', None)
    
    def _get_probs(weight):
        abs_weight = np.abs(weight)
        norm = np.sum(weight)/np.sum(abs_weight)
        prob = norm * abs_weight 
        return prob/prob.max()
    prob = _get_probs(histo2d.weights)
    def _sample_probs(prob):
        return prob > np.random.uniform(size=len(prob))
    index = np.arange(histo2d.counts)

    r = max(min_counts, int(histo2d.counts*fraction)) if fraction < 1 else int(fraction)
    r = min(histo2d.counts, r)
    randsample = np.array([]).astype(int)
    while len(randsample) < r:
        randsample = np.append(randsample, index[_sample_probs(prob)])

    randsample = randsample[:r]
    x, y, w = histo2d.x_array[randsample], histo2d.y_array[randsample], prob[randsample]
    bounds = (x > histo2d.x_bins[0]) & (x < histo2d.x_bins[-1]) & (y > histo2d.y_bins[0]) & (y < histo2d.y_bins[-1])
    x, y, w = [ v[bounds] for v in (x,y,w) ]
    w = alpha*np.ones_like(w)

    if discrete:
        x_dis = discrete_scale*get_bin_widths(histo2d.x_bins).mean()
        y_dis = discrete_scale*get_bin_widths(histo2d.y_bins).mean()

        x = x_dis*(x//x_dis)
        y = y_dis*(y//y_dis)

        if discrete_offset:
            i, of = discrete_offset 
            dx, dy = offsets[of][i]
            x += dx*x_dis
            y += dy*y_dis


    container = ax.scatter(x, y, s=size, c=color, alpha=w)
    ax.set(xlim=(histo2d.x_bins[0], histo2d.x_bins[-1]), ylim=(histo2d.y_bins[0], histo2d.y_bins[-1]))
    return container

def contour_histo2d(histo2d, figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)

    cmap = histo2d.kwargs.get('cmap','YlOrRd')
    alpha = histo2d.kwargs.get('alpha', None)

    x = get_bin_line(histo2d.x_bins)
    y = get_bin_line(histo2d.y_bins)
    Z = histo2d.histo2d

    cmap_kwargs = get_cmap_kwargs(cmap=cmap, **kwargs)
    container = ax.contourf( x, y, Z, alpha=alpha, **cmap_kwargs)
    return container

def interp_histo2d(histo2d, figax=None, kind='linear', scatter=False, **kwargs):
    fig, ax = get_figax(figax=figax)

    color = histo2d.kwargs.get('color', None)
    cmap = histo2d.kwargs.get('cmap','YlOrRd')
    alpha = histo2d.kwargs.get('alpha', None)

    if scatter:
        X, Y, Z = histo2d.x_array, histo2d.y_array, histo2d.weights
    else:
        X, Y, Z = histo2d.x_bins[:-1], histo2d.y_bins[:-1], histo2d.histo2d
        X, Y = np.meshgrid(X,Y)
        X, Y, Z = X[Z>0], Y[Z>0], Z[Z>0]

    interp_kind = dict(
        linear=interp.LinearNDInterpolator,
        clough=interp.CloughTocher2DInterpolator,
    ).get(kind, interp.LinearNDInterpolator)

    f = interp_kind(np.array([X,Y]).T, Z)
    nx = np.linspace(histo2d.x_bins[0], histo2d.x_bins[-1], 100)
    ny = np.linspace(histo2d.y_bins[0], histo2d.y_bins[-1], 100)
    nx, ny = np.meshgrid(nx, ny)
    nz = f(nx, ny)

    cmap_kwargs = get_cmap_kwargs(cmap=cmap, **kwargs)
    container = ax.pcolor(nx, ny, nz, alpha=alpha, **cmap_kwargs)
    return container

def bin_histo2d(histo2d, figax=None, show_counts=False, **kwargs):
    fig, ax = get_figax(figax=figax)

    cmap = histo2d.kwargs.get('cmap','YlOrRd')
    alpha = histo2d.kwargs.get('alpha', None)

    Z = np. where( histo2d.raw_counts>0, histo2d.histo2d, np.nan)
    # Z = np.where(Z > 0, Z, np.nan)
    cmap_kwargs = get_cmap_kwargs(cmap=cmap, **kwargs)
    container = ax.pcolor(histo2d.x_bins, histo2d.y_bins, Z, alpha=alpha, **cmap_kwargs)

    n = histo2d.histo2d 
    bx = histo2d.x_bins
    by = histo2d.y_bins

    if show_counts:
        fmt = lambda n : f'{n:0.2}'
        if callable(show_counts):
            fmt = show_counts
        for i,(bx_lo,bx_hi) in enumerate(zip(bx[:-1],bx[1:])):
            for j,(by_lo,by_hi) in enumerate(zip(by[:-1],by[1:])):
                if n[j,i] <= 0: continue
                # if vmin is not None and n[j,i] <= vmin: continue
                txt = fmt(n[j,i])
                if txt:
                    ax.text((bx_hi+bx_lo)/2,(by_hi+by_lo)/2,txt,ha="center", va="center", fontweight="bold")
    return container


def plot_histo2d(histo2d, figax=None, log=False, show_counts=False, zlim=None, contour=False, interp=False, scatter=False, exe=None, legend=False, colorbar=False, **kwargs):
    """Plot 2D histogram

    Args:
        histo2d (Histo2D): 2D Histogram to plot
        figax ((plt.fig,plt.ax), optional): Tuple of figure and axes to draw to. Defaults to None.
        cmap (str, optional): Color of histogram. Defaults to "YlOrRd".
        show_counts (bool, optional): Draw each bin count on plot. Defaults to False.
        log (bool, optional): Set z axis to log. Defaults to False.

    Returns:
        figax: Tuple of figure and axes
    """
    fig, ax = get_figax(figax=figax)

    if contour:
        container = contour_histo2d(histo2d, figax=(fig,ax), log=log, zlim=zlim, **(contour if isinstance(contour, dict) else {}))
    elif interp:
        container = interp_histo2d(histo2d, figax=(fig,ax), log=log, zlim=zlim, **(interp if isinstance(interp, dict) else {}))
    elif scatter:
        container = scatter_histo2d(histo2d, figax=(fig,ax), log=log, zlim=zlim, **(scatter if isinstance(scatter, dict) else {}))
    else:
        container = bin_histo2d(histo2d, figax=(fig,ax), log=log, show_counts=show_counts, zlim=zlim)
                
    if histo2d.kwargs.get('label',None) and legend:
        ax.text(0.05, 1.01, f"{histo2d.label} ({histo2d.stats.nevents:0.2e})", transform=ax.transAxes)
    
    cbar = None
    if colorbar and container:
        cbar = fig.colorbar(container, ax=ax)
        cbar.ax.minorticks_off()

    if exe: execute(**locals())
    if any(kwargs): format_axes(ax, is_2d=True, colorbar=cbar, **kwargs)
    return fig,ax


def plot_histo2ds(histo2ds, figax=None, show_counts=False, exe=None, log=False, legend=True, **kwargs):
    fig, ax = get_figax(figax=figax)
    for histo2d in histo2ds: plot_histo2d(histo2d, show_counts=show_counts, log=log, legend=legend, figax=(fig,ax))

    if exe: execute(**locals())
    if any(kwargs): format_axes(ax, is_2d=True, **kwargs)
    return fig,ax

def histo2d_array(x_array, y_array, x_bins=None, y_bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None, efficiency=False,
                figax=None, **kwargs):
    """Plot 2D histogram

    Args:
        x_arrays (array): Array for x axis histogram
        y_arrays (array): Array for y axis histogram
        x_bins (array, optional): Bins for x axis. Defaults to None.
        y_bins (array, optional): Bins for y axis. Defaults to None.
        weights (array, optional): Weights for both x and y arrays. Defaults to None.
        density (bool, optional): Normalize histogram to 1. Defaults to False.
        cumulative (bool, optional): _description_. Defaults to False.
        scale (scales, optional): Value to scale the weights by. Defaults to None.
        lumi (lumikey, optional): Luminosity to scale each histogram by, if reconginized lumikey will use corresponding luminosity (2018 -> 59.7 fb...). Defaults to None.
        figax ((plt.fig,plt.ax), optional): Tuple of the figure and axes to draw to. Defaults to None.

    Returns:
        figax: Tuple of figure and axes
    """
    fig, ax = get_figax(figax=figax)
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative,efficiency=efficiency,scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    histo2d = Histo2D(x_array,y_array, x_bins=x_bins, y_bins=y_bins,weights=weights,**hist_kwargs)
    plot_histo2d(histo2d, figax=(fig,ax), **kwargs)
    
    return fig,ax

def interp_graph2d(graph2d, figax=None, kind='linear', **kwargs):
    fig, ax = get_figax(figax=figax)

    cmap = graph2d.kwargs.get('cmap','YlOrRd')
    alpha = graph2d.kwargs.get('alpha', None)

    X, Y, Z = graph2d.x_array, graph2d.y_array, graph2d.z_array

    interp_kind = dict(
        linear=interp.LinearNDInterpolator,
        clough=interp.CloughTocher2DInterpolator,
    ).get(kind, interp.LinearNDInterpolator)

    f = interp_kind(np.array([X,Y]).T, Z)
    nx = np.linspace(np.min(graph2d.x_array), np.max(graph2d.x_array), 100)
    ny = np.linspace(np.min(graph2d.y_array), np.max(graph2d.y_array), 100)
    nx, ny = np.meshgrid(nx, ny)
    nz = f(nx, ny)

    cmap_kwargs = get_cmap_kwargs(cmap=cmap, **kwargs)
    container = ax.pcolor(nx, ny, nz, alpha=alpha, **cmap_kwargs)
    return container


def scatter_graph2d(graph2d, figax=None, alpha=0.25, size=5, discrete=False, discrete_offset=None, discrete_scale=0.5, **kwargs):
    fig, ax = get_figax(figax=figax)
    cmap = graph2d.kwargs.get('cmap', 'jet')
    
    x, y, z = graph2d.x_array, graph2d.y_array, graph2d.z_array

    # if discrete:
    #     x_dis = discrete_scale*get_bin_widths(graph2d.x_array).mean()
    #     y_dis = discrete_scale*get_bin_widths(graph2d.y_bins).mean()

    #     x = x_dis*(x//x_dis)
    #     y = y_dis*(y//y_dis)

    #     if discrete_offset:
    #         i, of = discrete_offset 
    #         dx, dy = offsets[of][i]
    #         x += dx*x_dis
    #         y += dy*y_dis


    cmap_kwargs = get_cmap_kwargs(cmap=cmap, **kwargs)
    container = ax.scatter(x, y, s=size, c=z, **cmap_kwargs)
    return container

def plot_graph2d(graph2d, figax=None, alpha=0.25, size=5, log=False, zlim=None, contour=False, interp=False, exe=None, legend=False, colorbar=False, **kwargs):
    """Plot 2D graph

    Args:
        graph2d (Graph2D): 2D Graph to plot
        figax ((plt.fig,plt.ax), optional): Tuple of figure and axes to draw to. Defaults to None.
        cmap (str, optional): Color of histogram. Defaults to "YlOrRd".
        log (bool, optional): Set z axis to log. Defaults to False.

    Returns:
        figax: Tuple of figure and axes
    """
    fig, ax = get_figax(figax=figax)

    if contour:
        container = contour_graph2d(graph2d, figax=(fig,ax), log=log, zlim=zlim, **(contour if isinstance(contour, dict) else {}))
    elif interp:
        container = interp_graph2d(graph2d, figax=(fig,ax), log=log, zlim=zlim, **(interp if isinstance(interp, dict) else {}))
    else:
        container = scatter_graph2d(graph2d, figax=(fig,ax), log=log, zlim=zlim, alpha=alpha, size=size)
                
    if graph2d.kwargs.get('label',None) and legend:
        ax.text(0.05, 1.01, f"{graph2d.label} ({graph2d.stats.nevents:0.2e})", transform=ax.transAxes)
    
    cbar = None
    if colorbar and container:
        cbar = fig.colorbar(container, ax=ax)
        cbar.ax.minorticks_off()

    if exe: execute(**locals())
    if any(kwargs): format_axes(ax, is_2d=True, colorbar=cbar, **kwargs)
    return fig,ax

def graph2d_array(x_array, y_array, z_array, figax=None, **kwargs):
    """Plot 2D graph

    Args:
        x_arrays (array): Array for x axis histogram
        y_arrays (array): Array for y axis histogram
        z_arrays (array): Array for z axis histogram
        figax ((plt.fig,plt.ax), optional): Tuple of the figure and axes to draw to. Defaults to None.

    Returns:
        figax: Tuple of figure and axes
    """
    fig, ax = get_figax(figax=figax)
    
    # --- Configure kwargs ---
    ext_kwargs = dict()
    graph_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('g_') }
    graph_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('g_') }
    kwargs.update(ext_kwargs)
    
    graph2d = Graph2D(x_array,y_array, z_array,**graph_kwargs)
    plot_graph2d(graph2d, figax=(fig,ax), **kwargs)
    
    return fig,ax