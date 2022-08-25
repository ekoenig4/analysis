import matplotlib.colors as clrs
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science','no-latex'])
plt.rcParams["figure.figsize"] = (6.5,6.5)
plt.rcParams['font.size'] =  15

from ..utils import get_bin_centers, get_bin_widths, get_bin_line
from .histogram import Histo, HistoList, Stack
from .histogram2d import Histo2D
from .graph import Graph, GraphList, Ratio
from .formater import format_axes

def execute(exe=None, **local):
    import awkward as ak
    import numpy as np

    local.update(locals())
    exes = exe if isinstance(exe, list) else [exe]
    def _execute(exe):
        if callable(exe): exe(**local)
        else: eval(exe, local)
    for exe in exes:
        _execute(exe)

def get_figax(figax=None):
    if figax is None: return plt.subplots()
    if figax == 'same':
        if not any(plt.get_fignums()): return plt.subplots()
        fig = plt.gcf()
        ax = fig.get_axes()[-1]
        return (fig,ax)
    return figax    

def plot_function(function, figax=None, exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    
    ax.plot(function.x_array, function.y_array, **function.kwargs)

    if exe: execute(**locals())
    if any(kwargs): format_axes(ax, **kwargs)
    return fig,ax

def plot_graph(graph, errors=True, fill_error=False, figax=None, exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    xerr, yerr = (graph.xerr, graph.yerr) if errors else (None,None)

    if not fill_error:
        fill_error = (graph.x_array.shape[0] >= 30) and (yerr is not None)

    if not fill_error:
        container = ax.errorbar(graph.x_array,graph.y_array, xerr=xerr, yerr=yerr, **graph.kwargs)
        graph.kwargs['color'] = container[0].get_color()
    elif yerr is not None:
        container = ax.errorbar(graph.x_array, graph.y_array, xerr=xerr, **graph.kwargs)
        graph.kwargs['color'] = container[0].get_color()
        for nstd in range(1, int(fill_error)+1 ):
            ax.fill_between(graph.x_array, graph.y_array-nstd*yerr, graph.y_array+nstd*yerr, color=graph.kwargs['color'], alpha=0.25/nstd)
    elif xerr is not None:
        container = ax.errorbar(graph.x_array, graph.y_array, yerr=yerr, **graph.kwargs)
        graph.kwargs['color'] = container[0].get_color()
        for nstd in range(1, int(fill_error)+1 ):
            ax.fill_betweenx(graph.y_array, graph.x_array-nstd*xerr, graph.x_array+nstd*xerr, color=graph.kwargs['color'], alpha=0.25/nstd)


    if getattr(graph, 'fit', None) is not None:
        plot_function(graph.fit, figax=(fig,ax))
    
    kwargs['ylabel'] = kwargs.get('ylabel', None)
    
    if exe: execute(**locals())
    if any(kwargs): format_axes(ax, **kwargs)
    return fig,ax

def plot_graphs(graphs, figax=None, errors=True, fill_error=False, exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    for graph in graphs: plot_graph(graph, errors=errors, fill_error=fill_error, figax=(fig,ax))
    kwargs['ylabel'] = kwargs.get('ylabel', None)
    
    if exe: execute(**locals())
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax
    
def graph_array(x, y, xerr=None, yerr=None, figax=None, **kwargs):
    (fig,ax) = get_figax(figax=figax)
    
    # --- Configure kwargs ---
    graph_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('g_') }
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('g_') }

    graph = Graph(x,y, xerr=xerr, yerr=yerr, **graph_kwargs)
    plot_graph(graph,figax=(fig,ax),**kwargs)
    
    return (fig,ax)    

def graph_arrays(x_arrays, y_arrays, xerr=None, yerr=None, figax=None, **kwargs):
    (fig,ax) = get_figax(figax=figax)
    
    # --- Configure kwargs ---
    graph_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('g_') }
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('g_') }

    graphlist = GraphList(x_arrays,y_arrays, xerr=xerr, yerr=yerr, **graph_kwargs)
    plot_graphs(graphlist,figax=(fig,ax),**kwargs)
    
    return (fig,ax)    
    
def graph_histo(histo, errors=True, figax=None, exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    
    bins = histo.bins 
    bin_centers = get_bin_centers(bins)
    yerr = histo.error if errors else None
    xerr = get_bin_widths(bins) if errors else None
    ax.errorbar(bin_centers, histo.histo, xerr=xerr, yerr=yerr, **histo.kwargs)
    
    if getattr(histo, 'fit', None) is not None:
        plot_function(histo.fit, figax=(fig,ax))
    
    if exe: execute(**locals())
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax
    
def graph_histos(histos, figax=None, exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    for histo in histos: graph_histo(histo, figax=(fig,ax))
    
    if exe: execute(**locals())
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax
    
def plot_histo(histo, errors=True, fill_error=False, fit_show=False, figax=None, exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    if histo.continous: 
        return graph_histo(histo, errors=errors, figax=(fig,ax), **kwargs)

    bin_centers = get_bin_centers(histo.bins)
    
    _,_,container = ax.hist(bin_centers, bins=histo.bins, weights=histo.histo, **histo.kwargs)
    histo.kwargs['color'] = container[0].get_ec() if container[0].get_ec() != (0.,0.,0.,0.) else container[0].get_fc()
    color = histo.kwargs['color'] if histo.kwargs.get('histtype',False) else 'black'
    
    if errors:
        if not fill_error:
            ax.errorbar(bin_centers, histo.histo, yerr=histo.error,fmt='none', color=color, capsize=1)
        else:
            for nstd in range(1, int(fill_error)+1 ):
                ax.fill_between(bin_centers, histo.histo-nstd*histo.error, histo.histo+nstd*histo.error, color=histo.kwargs['color'], alpha=0.25/nstd, step='mid')

    if getattr(histo, 'fit', None) is not None:
        if histo.fit.show:
            plot_function(histo.fit, figax=(fig,ax))
        
    if exe: execute(**locals())
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax

def plot_histos(histos, figax=None, errors=True, fill_error=False, exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    for histo in histos: plot_histo(histo, errors=errors, fill_error=fill_error, figax=(fig,ax))

    if exe: execute(**locals())
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax

def histo_array(array, bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None,
                figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative,scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    histo = Histo(array,bins=bins,weights=weights,**hist_kwargs)
    plot_histo(histo, figax=(fig,ax), **kwargs)
    
    return fig,ax

def histo_arrays(arrays, bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None,
                figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative,scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    histolist = HistoList(arrays,bins=bins,weights=weights,**hist_kwargs)
    plot_histos(histolist, figax=(fig,ax), **kwargs)
    
    return fig,ax

def plot_stack(stack, figax=None, fill_error=False, exe=None, **kwargs):
    fig, ax = get_figax(figax=(figax))
    
    bin_centers = get_bin_centers(stack.bins)
    bin_widths = 2*get_bin_widths(stack.bins)

    if not stack.stack_fill:
        histo_sum = np.zeros(stack[0].histo.shape)
        for i,histo in enumerate(stack):
            container = ax.bar(bin_centers, histo.histo, bin_widths,
                bottom=histo_sum, **histo.kwargs)
            histo.kwargs['color'] = container[0].get_fc()
            histo_sum = histo_sum + histo.histo

        error = np.sqrt((stack.error.npy**2).sum(axis=0))

        if not fill_error:
            ax.errorbar(bin_centers, histo_sum, yerr=error,
                        fmt='none', color='grey', capsize=1)
        else:
            for nstd in range(1, int(fill_error)+1 ):
                ax.fill_between(bin_centers, histo_sum-nstd*error, histo_sum+nstd*error, color='grey', alpha=0.25/nstd, step='mid')
    else:
        histo = stack.get_histo()
        plot_histo(histo, figax=(fig,ax), fill_error=fill_error)
    
    if exe: execute(**locals())
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax

def stack_arrays(arrays, bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None, stack_fill=False,
                figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative,scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    stack = Stack(arrays,bins=bins,weights=weights,stack_fill=stack_fill,**hist_kwargs)
    plot_stack(stack, figax=(fig,ax), **kwargs)

def histo_ratio(histos, figax=None, inv=False, ylabel='Ratio', exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    
    num = histos[0]
    dens = histos[1:]
    ratios = [ Ratio(num,den,inv) for den in dens ]
    plot_graphs(ratios, figax=(fig,ax), **kwargs)
    
    if exe: execute(**locals())
    if any(kwargs): format_axes(ax, ylabel=ylabel, **kwargs)
    return fig,ax

def array_ratio(arrays, bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None,
                figax=None, ylabel='Ratio', **kwargs):
    """Plots the ratio of the first array histogram to all the rest

    Args:
        arrays (List(arrays)): List of arrays to make histograms and take the ratio of
        bins (array, optional): Bin array to use for histograms. Defaults to None.
        weights (List(arrays), optional): Weights to weight each histogram. Defaults to None.
        density (bool, optional): Normalize each histogram to 1. Defaults to False.
        cumulative (bool, optional): Plot CDF of each histogram, 1 -> for cdf below, -1 -> for cdf above. Defaults to False.
        scale (List(scales), optional): List of scalings for each array. Defaults to None.
        lumi (lumikey, optional): Luminosity to scale each histogram by, if reconginized lumikey will use corresponding luminosity (2018 -> 59.7 fb...). Defaults to None.
        figax ((plt.fig,plt.ax), optional): Tuple of the figure and axes to draw to. Defaults to None.
        ylabel (str, optional): Set the ylabel. Defaults to 'Ratio'.
        kwargs: all kwargs staring with h_ are passed to histograms, remaining ones will be used to format axes

    Returns:
        figax: Tuple of figure and axes
    """
    fig, ax = get_figax(figax=figax)
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative,scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    histolist = HistoList(arrays, bins=bins,weights=weights,**hist_kwargs)
    histo_ratio(histolist,figax=(fig,ax), ylabel=ylabel, **kwargs)
    
    return fig,ax

def plot_histo2d(histo2d, figax=None, cmap="YlOrRd", show_counts=False, log=False, alpha=None, cmin=None, contour=False, exe=None, **kwargs):
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

    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogLocator

    if contour:
        # x = get_bin_centers(histo2d.x_bins)
        # y = get_bin_centers(histo2d.y_bins)

        x = get_bin_line(histo2d.x_bins)
        y = get_bin_line(histo2d.y_bins)

        Z = np.log(histo2d.histo2d) if log else histo2d.histo2d
        container = ax.contourf( x, y, Z, cmap=cmap, vim=cmin, alpha=alpha)
    else:
        container = ax.pcolor(histo2d.x_bins, histo2d.y_bins, histo2d.histo2d, cmap=cmap, vmin=cmin, alpha=alpha, norm=LogNorm() if log else None)

    # n, bx, by, im = ax.hist2d(histo2d.x_array, histo2d.y_array, (histo2d.x_bins, histo2d.y_bins), weights=histo2d.weights,
    #                                     norm=clrs.LogNorm() if log else clrs.Normalize(), cmap=cmap, alpha=alpha, cmin=cmin)
    
    n = histo2d.histo2d 
    bx = histo2d.x_bins
    by = histo2d.y_bins

    if show_counts:
        for i,(bx_lo,bx_hi) in enumerate(zip(bx[:-1],bx[1:])):
            for j,(by_lo,by_hi) in enumerate(zip(by[:-1],by[1:])):
                ax.text((bx_hi+bx_lo)/2,(by_hi+by_lo)/2,f'{n[i,j]:0.2}',ha="center", va="center", fontweight="bold")
                
    if histo2d.kwargs.get('label',None):
        ax.text(0.05, 1.01, f"{histo2d.label} ({histo2d.stats.nevents:0.2e})", transform=ax.transAxes)
                
    fig.colorbar(container, ax=ax)

    if exe: execute(**locals())
    if any(kwargs): format_axes(ax, is_2d=True, **kwargs)
    return fig,ax

def histo2d_arrays(x_array, y_array, x_bins=None, y_bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None, efficiency=False,
                figax=None, **kwargs):
    """Plot 2D histogram

    Args:
        x_array (array): Array for x axis histogram
        y_array (array): Array for y axis histogram
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

def plot_model(model, **kwargs):
    plotobjs = [model.h_sig, model.h_bkg]
    if model.h_data is not model.h_bkg:
        plotobjs.append(model.h_data)
    
    return plot_histos(plotobjs, **kwargs)
    
    