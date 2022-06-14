import matplotlib.colors as clrs
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
import numpy as np

from ..utils import get_bin_centers, get_bin_widths
from .histogram import HistoList, Stack
from .graph import GraphList, Ratio
from .formater import format_axes

def plot_graph(graph, figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    ax.errorbar(graph.x_array,graph.y_array,**graph.kwargs)
    
    if any(kwargs): format_axes(ax, **kwargs)
    return fig,ax

def plot_graphs(graphs, figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    for graph in graphs: plot_graph(graph, figax=figax)
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax
    
def graph_arrays(x_arrays, y_arrays, figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    # --- Configure kwargs ---
    graph_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('g_') }
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('g_') }

    graphlist = GraphList(x_arrays,y_arrays,**graph_kwargs)
    plot_graphs(graphlist,figax=figax,**kwargs)
    
    return fig,ax    
    
def graph_histo(histo, figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    bins = histo.bins 
    bin_centers, bin_widths = get_bin_centers(bins), get_bin_widths(bins)
    ax.errorbar(bin_centers, histo.histo, xerr=bin_widths, **histo.kwargs)
    
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax
    
def graph_histos(histos, figax=None, **kwargs):
    fig,ax = figax
    for histo in histos: graph_histo(histo, figax=figax)
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax
    
    
def plot_histo(histo, errors=True, figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax

    bin_centers = get_bin_centers(histo.bins)
    
    _,_,container = ax.hist(bin_centers, bins=histo.bins, weights=histo.histo, **histo.kwargs)
    histo.kwargs['color'] = container[0].get_ec()
    color = histo.kwargs['color'] if histo.kwargs.get('histtype',False) else 'black'
    if errors:
        ax.errorbar(bin_centers, histo.histo, yerr=histo.error,fmt='none', color=color, capsize=1)
    
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax

def plot_histos(histos, figax=None, errors=True, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    for histo in histos: plot_histo(histo, errors=errors, figax=figax)
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax
    
def histo_arrays(arrays, bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None,
                figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative,scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    histolist = HistoList(arrays,bins=bins,weights=weights,**hist_kwargs)
    plot_histos(histolist, figax=figax, **kwargs)
    
    return fig,ax

def plot_stack(stack, figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    bin_centers = get_bin_centers(stack.bins[0])
    bin_widths = 2*get_bin_widths(stack.bins[0])
    
    histo_sum = np.zeros(stack[0].histo.shape)
    for i,histo in enumerate(stack):
        container = ax.bar(bin_centers, histo.histo, bin_widths,
               bottom=histo_sum, **histo.kwargs)
        histo.kwargs['color'] = container[0].get_fc()
        histo_sum = histo_sum + histo.histo

    error = np.sqrt((stack.error.npy**2).sum(axis=0))
    ax.errorbar(bin_centers, histo_sum, yerr=error,
                fmt='none', color='grey', capsize=1)
    
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax

def stack_arrays(arrays, bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None,
                figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative,scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    stack = Stack(arrays,bins=bins,weights=weights,**hist_kwargs)
    plot_stack(stack, figax=figax, **kwargs)

def histo_ratio(histos, figax=None, inv=False, ylabel='Ratio', **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    num = histos[0]
    dens = histos[1:]
    ratios = [ Ratio(num,den,inv) for den in dens ]
    plot_graphs(ratios, figax=figax, **kwargs)
    
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
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative,scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    histolist = HistoList(arrays, bins=bins,weights=weights,**hist_kwargs)
    histo_ratio(histolist,figax=figax, ylabel=ylabel, **kwargs)
    
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
    if figax is None: figax = plt.subplots()
    fig,ax = figax

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
                cumulative=False, scale=None, lumi=None,
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
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative,scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    histolist = HistoList([x_array,y_array],bins=[x_bins,y_bins],weights=weights,**hist_kwargs)
    plot_histo2d(histolist[0], histolist[1], figax=figax, **kwargs)
    
    return fig,ax
    
    
    
    
    