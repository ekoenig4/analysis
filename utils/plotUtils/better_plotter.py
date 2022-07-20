import matplotlib.colors as clrs
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science','no-latex'])
plt.rcParams["figure.figsize"] = (6.5,6.5)
plt.rcParams['font.size'] =  15

from ..utils import get_bin_centers, get_bin_widths
from .histogram import Histo, HistoList, Stack
from .graph import Graph, GraphList, Ratio
from .formater import format_axes

def get_figax(figax=None):
    if figax is None: return plt.subplots()
    if figax == 'same' and any(plt.get_fignums()):
        fig = plt.gcf()
        ax = fig.get_axes()[-1]
        return (fig,ax)
    if figax == 'same' and not any(plt.get_fignums()):
        return plt.subplots()
    return figax    

def plot_function(function, figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    
    ax.plot(function.x_array, function.y_array, **function.kwargs)

    if any(kwargs): format_axes(ax, **kwargs)
    return fig,ax

def plot_graph(graph, errors=True, fill_error=False, figax=None, **kwargs):
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


    if getattr(graph, 'fit', None) is not None:
        plot_function(graph.fit, figax=(fig,ax))
    
    kwargs['ylabel'] = kwargs.get('ylabel', None)
    if any(kwargs): format_axes(ax, **kwargs)
    return fig,ax

def plot_graphs(graphs, figax=None, errors=True, fill_error=False, **kwargs):
    fig, ax = get_figax(figax=figax)
    for graph in graphs: plot_graph(graph, errors=errors, fill_error=fill_error, figax=(fig,ax))
    kwargs['ylabel'] = kwargs.get('ylabel', None)
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
    
def graph_histo(histo, errors=True, figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    
    bins = histo.bins 
    bin_centers = get_bin_centers(bins)
    yerr = histo.error if errors else None
    xerr = get_bin_widths(bins) if errors else None
    ax.errorbar(bin_centers, histo.histo, xerr=xerr, yerr=yerr, **histo.kwargs)
    
    if getattr(histo, 'fit', None) is not None:
        plot_function(histo.fit, figax=(fig,ax))
    
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax
    
def graph_histos(histos, figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    for histo in histos: graph_histo(histo, figax=(fig,ax))
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax
    
    
def plot_histo(histo, errors=True, fill_error=False, figax=None, **kwargs):
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
        plot_function(histo.fit, figax=(fig,ax))
    
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax

def plot_histos(histos, figax=None, errors=True, fill_error=False, **kwargs):
    fig, ax = get_figax(figax=figax)
    for histo in histos: plot_histo(histo, errors=errors, fill_error=fill_error, figax=(fig,ax))
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

def plot_stack(stack, figax=None, fill_error=False, **kwargs):
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

def histo_ratio(histos, figax=None, inv=False, ylabel='Ratio', **kwargs):
    fig, ax = get_figax(figax=figax)
    
    num = histos[0]
    dens = histos[1:]
    ratios = [ Ratio(num,den,inv) for den in dens ]
    plot_graphs(ratios, figax=(fig,ax), **kwargs)
    
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

def plot_histo2d(x_histo, y_histo, figax=None, cmap="YlOrRd", show_counts=False, log=False, alpha=None, cmin=None, **kwargs):
    """Plot 2D histogram

    Args:
        x_histo (Histo): Histogram to use along x axis
        y_histo (Histo): Histogram to use along y axis
        figax ((plt.fig,plt.ax), optional): Tuple of figure and axes to draw to. Defaults to None.
        cmap (str, optional): Color of histogram. Defaults to "YlOrRd".
        show_counts (bool, optional): Draw each bin count on plot. Defaults to False.
        log (bool, optional): Set z axis to log. Defaults to False.

    Returns:
        figax: Tuple of figure and axes
    """
    fig, ax = get_figax(figax=figax)

    n, bx, by, im = ax.hist2d(x_histo.array, y_histo.array, (x_histo.bins, y_histo.bins), weights=x_histo.weights,
                                        norm=clrs.LogNorm() if log else clrs.Normalize(), cmap=cmap, alpha=alpha, cmin=cmin)

    if show_counts:
        for i,(bx_lo,bx_hi) in enumerate(zip(bx[:-1],bx[1:])):
            for j,(by_lo,by_hi) in enumerate(zip(by[:-1],by[1:])):
                ax.text((bx_hi+bx_lo)/2,(by_hi+by_lo)/2,f'{n[i,j]:0.2}',ha="center", va="center", fontweight="bold")
                
    if x_histo.kwargs.get('label',None):
        ax.text(0.05, 1.01, f"{x_histo.label} ({x_histo.stats.nevents:0.2e})", transform=ax.transAxes)
                
    fig.colorbar(im, ax=ax)
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
    
    histolist = HistoList([x_array,y_array],bins=[x_bins,y_bins],weights=weights,**hist_kwargs)
    plot_histo2d(histolist[0], histolist[1], figax=(fig,ax), **kwargs)
    
    return fig,ax
    
    
    
    
    