import matplotlib.colors as clrs
import matplotlib.pyplot as plt
import numpy as np

from .binning_tools import get_bin_centers, get_bin_widths, get_bin_line
from .histogram import Histo, HistoList, Stack
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
    if figax == 'none': return None,None
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

def plot_graph(graph, errors=True, bar=False, fill_error=False, fill_alpha=0.25, figax=None, exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    xerr, yerr = (graph.xerr, graph.yerr) if errors else (None,None)

    kwargs.update(
        xlabel=getattr(graph, 'xlabel', kwargs.get('xlabel', None))
    )

    if not fill_error:
        fill_error = (graph.x_array.shape[0] >= 30) and (yerr is not None)

    if bar:
        container = ax.errorbar(graph.x_array,graph.y_array, yerr=yerr, **dict(graph.kwargs, marker=None, linewidth=0))
        graph.kwargs['color'] = container[0].get_color()

        ax.bar(graph.x_array, graph.y_array, width=2*xerr,  fill=not bar == 'step', edgecolor=graph.kwargs['color'])

    elif not fill_error:
        container = ax.errorbar(graph.x_array,graph.y_array, xerr=xerr, yerr=yerr, **graph.kwargs)
        graph.kwargs['color'] = container[0].get_color()
    elif yerr is not None and xerr is not None:
        container = ax.errorbar(graph.x_array,graph.y_array, **graph.kwargs)
        graph.kwargs['color'] = container[0].get_color()

        xlo = graph.x_array-xerr
        xhi = graph.x_array+xerr
        ylo = graph.y_array-yerr
        yhi = graph.y_array+yerr

        for x, y1, y2 in zip( np.stack([xlo, xhi], axis=1), ylo, yhi ):
            if np.isnan(y1) or np.isnan(y2): continue
            ax.fill_between(x, y1, y2, color=graph.kwargs['color'], alpha=fill_alpha)
    elif yerr is not None:
        container = ax.errorbar(graph.x_array, graph.y_array, xerr=xerr, **graph.kwargs)
        graph.kwargs['color'] = container[0].get_color()
        for nstd in range(1, int(fill_error)+1 ):
            ax.fill_between(graph.x_array, graph.y_array-nstd*yerr, graph.y_array+nstd*yerr, color=graph.kwargs['color'], alpha=fill_alpha/nstd)
    elif xerr is not None:
        container = ax.errorbar(graph.x_array, graph.y_array, yerr=yerr, **graph.kwargs)
        graph.kwargs['color'] = container[0].get_color()
        for nstd in range(1, int(fill_error)+1 ):
            ax.fill_betweenx(graph.y_array, graph.x_array-nstd*xerr, graph.x_array+nstd*xerr, color=graph.kwargs['color'], alpha=fill_alpha/nstd)


    if getattr(graph, 'fit', None) is not None:
        plot_function(graph.fit, figax=(fig,ax))
    
    kwargs['ylabel'] = kwargs.get('ylabel', None)
    
    if exe: execute(**locals())
    if any(kwargs): format_axes(ax, **kwargs)
    return fig,ax

def plot_graphs(graphs, figax=None, errors=True, fill_error=False, fill_alpha=0.25, exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    for graph in graphs: plot_graph(graph, errors=errors, fill_error=fill_error, fill_alpha=fill_alpha, figax=(fig,ax))
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
    container = ax.errorbar(bin_centers, histo.histo, xerr=xerr, yerr=yerr, **histo.kwargs)
    histo.kwargs['color'] = container[0].get_c() if container[0].get_c() != (0.,0.,0.,0.) else container[0].get_mfc()
    
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

    bins = histo.bins 
    x = get_bin_centers(histo.bins)
    y = histo.histo * histo.plot_scale
    yerr = histo.error * histo.plot_scale
    
    _,_,container = ax.hist(x, bins=bins, weights=y, **histo.kwargs)
    histo.kwargs['color'] = container[0].get_ec() if container[0].get_ec() != (0.,0.,0.,0.) else container[0].get_fc()
    color = histo.kwargs['color'] if histo.kwargs.get('histtype',False) else 'black'
    
    if errors:
        yerr = np.where( (y-yerr)>0, yerr, y)

        if not fill_error:
            ax.errorbar(x, y, yerr=yerr,fmt='none', color=color, capsize=1)
        else:
            for nstd in range(1, int(fill_error)+1 ):
                ax.fill_between(x, y-nstd*yerr, y+nstd*yerr, color=histo.kwargs['color'], alpha=0.25/nstd, step='mid')

    if getattr(histo, 'fit', None) is not None:
        if histo.fit.show:
            plot_function(histo.fit, figax=(fig,ax))
        
    if exe: execute(**locals())
    if any(kwargs): format_axes(ax,**kwargs)
    return fig,ax

def plot_histo_error(histo, figax=None, ylim=(0, 2), g_linestyle='--', exe=None, grid=True, **kwargs):
    fig, ax = get_figax(figax=figax)
    centers = get_bin_centers(histo.bins)
    widths = get_bin_widths(histo.bins)
    errors = histo.error / histo.histo

    graph_array(
        centers,
        np.ones_like(centers),
        xerr=widths,
        yerr=errors,
        g_linestyle=g_linestyle,
        ylim=ylim,
        grid=grid,
        figax=(fig,ax),
        **kwargs,
    )

    if exe: execute(**locals())
    return fig,ax

def plot_histos(histos, figax=None, errors=True, fill_error=False, exe=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    for histo in histos: plot_histo(histo, errors=errors, fill_error=fill_error, figax=(fig,ax))

    if exe: execute(**locals())
    if any(kwargs): format_axes(ax,**kwargs)
    return fig, ax

def histo_array(array, bins=None, weights=None, 
                density = False, cumulative=False, efficiency=False,
                scale=None, lumi=None,
                figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative, efficiency=efficiency, scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    histo = Histo.from_array(array,bins=bins,weights=weights,**hist_kwargs)
    plot_histo(histo, figax=(fig,ax), **kwargs)
    
    return fig, ax, histo

def histo_arrays(arrays, bins=None, weights=None, 
                density = False, cumulative=False, efficiency=False,
                scale=None, lumi=None,
                figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)
    
    # --- Configure kwargs ---
    ext_kwargs = dict(density=density,cumulative=cumulative, efficiency=efficiency, scale=scale, lumi=lumi)
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(ext_kwargs)
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    kwargs.update(ext_kwargs)
    
    histolist = HistoList.from_arrays(arrays,bins=bins,weights=weights,**hist_kwargs)
    plot_histos(histolist, figax=(fig,ax), **kwargs)
    
    return fig, ax, histolist

def plot_stack(stack, figax=None, fill_error=False, exe=None, sort='largest', **kwargs):
    fig, ax = get_figax(figax=(figax))

    sorting = dict(
        smallest=lambda h:h.ndata,
        largest=lambda h:-h.ndata,
    ).get(sort)
    
    bin_centers = get_bin_centers(stack.bins)
    bin_widths = 2*get_bin_widths(stack.bins)

    if not stack.stack_fill:
        histo_sum = np.zeros(stack[0].histo.shape)
        for i,histo in enumerate( sorted(stack, key=sorting) ):
            container = ax.bar(bin_centers, histo.histo, bin_widths,
                bottom=histo_sum, **histo.kwargs)
            histo.kwargs['color'] = container[0].get_fc()
            histo_sum = histo_sum + histo.histo

        error = np.sqrt((stack.error.npy**2).sum(axis=0))
        error = np.where( (histo_sum-error)>0, error, histo_sum)
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

def plot_model(model, figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)

    plotobjs = [model.h_sig, model.h_bkg]
    if model.h_data is not model.h_bkg:
        graph_histo(model.h_data, figax=(fig,ax))
    
    return plot_histos(plotobjs, figax=(fig, ax), **kwargs)

def plot_model_brazil(models, label=None, xlabel='mass', ylabel=None, units='pb', figax=None, **kwargs):
    fig, ax = get_figax(figax=figax)

    exp_limits = np.array([model.h_sig.stats.exp_limits for model in models]).T
    
    unitMap = dict(
        pb=1,
        fb=1e3,
    )
    exp_limits *= unitMap[units]

    exp_p2 = exp_limits[2+2]
    exp_p1 = exp_limits[2+1]
    exp = exp_limits[2]
    exp_m1 = exp_limits[2-1]
    exp_m2 = exp_limits[2-2]
    exp_std2_mu = (exp_p2 + exp_m2)/2
    exp_std2_err = (exp_p2 - exp_m2)/2

    exp_std1_mu = (exp_p1 + exp_m1)/2
    exp_std1_err = (exp_p1 - exp_m1)/2

    def get_x(h, xlabel=xlabel):
        mx, my = h.label.split('_')[1::2]
        if xlabel == 'mx':
            return int(mx)
        if xlabel == 'my':
            return int(my)
        return f'({mx}, {my})'
    
    x = np.array( [get_x( model.h_sig ) for model in models] )

    if x.dtype == np.dtype('U1'):
        xlabel = x.tolist()
        x = np.arange(len(x))

    if isinstance(xlabel, str):
        xlabel = dict(
            mx='$M_{X}$ (GeV)',
            my='$M_{Y}$ (GeV)',
        ).get(xlabel, xlabel)

    g_exp = Graph(x, exp, color='black', label=label,
                  linestyle='--', marker='o')
    
    _x = np.concatenate([ x[:1]-25, x, x[-1:]+25 ])
    _exp_std1_mu = np.concatenate([ exp_std1_mu[:1], exp_std1_mu, exp_std1_mu[-1:] ])
    _exp_std1_err = np.concatenate([ exp_std1_err[:1], exp_std1_err, exp_std1_err[-1:] ])
    _exp_std2_mu = np.concatenate([ exp_std2_mu[:1], exp_std2_mu, exp_std2_mu[-1:] ])
    _exp_std2_err = np.concatenate([ exp_std2_err[:1], exp_std2_err, exp_std2_err[-1:] ])

    g_exp_std1 = Graph(_x, _exp_std1_mu, yerr=_exp_std1_err,
                       color='#00cc00', marker=None, linewidth=0)
    g_exp_std2 = Graph(_x, _exp_std2_mu, yerr=_exp_std2_err,
                       color='#ffcc00', marker=None, linewidth=0)
    
    plot_graphs([g_exp_std2, g_exp_std1], fill_error=True,
                fill_alpha=1, figax=(fig, ax))
    plot_graph(g_exp, figax=(fig, ax),
               xlabel=xlabel, ylabel=ylabel, **kwargs)
    return g_exp
    
    