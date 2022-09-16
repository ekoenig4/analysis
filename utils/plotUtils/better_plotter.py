import matplotlib.colors as clrs
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
import numpy as np

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
    
def plot_histo(histo, errors=True, fill_error=False, figax=None, exe=None, **kwargs):
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
        for i,histo in enumerate( sorted(stack, key=lambda h:h.ndata) ):
            container = ax.bar(bin_centers, histo.histo, bin_widths,
                bottom=histo_sum, **histo.kwargs)
            histo.kwargs['color'] = container[0].get_fc()
            histo_sum = histo_sum + histo.histo

        error = np.sqrt((stack.error.npy**2).sum(axis=0))

        if not fill_error:
            ax.errorbar(bin_centers, histo_sum, yerr=error,
                        fmt='none', color='black', capsize=1)
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

def plot_model(model, **kwargs):
    plotobjs = [model.h_sig, model.h_bkg]
    if model.h_data is not model.h_bkg:
        plotobjs.append(model.h_data)
    
    return plot_histos(plotobjs, **kwargs)
    
    