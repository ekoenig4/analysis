from numpy import correlate
from ..classUtils.AttrArray import AttrArray
from .better_plotter import * 
from .histogram import * 
from .graph import * 
from .function import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import itertools 


def _configure_kwargs(**kwargs):
    # --- Configure kwargs ---

    def group_kwargs(prefix, **kwargs):
        grouped_kwargs = { key[len(prefix):]:value for key,value in kwargs.items() if key.startswith(prefix) }
        remaining_kwargs = { key:value for key,value in kwargs.items() if not key.startswith(prefix) }
        return grouped_kwargs, remaining_kwargs

    o_kwargs, kwargs = group_kwargs('o_', **kwargs)
    h_kwargs, kwargs = group_kwargs('h_', **kwargs)
    r_kwargs, kwargs = group_kwargs('r_', **kwargs)
    d_kwargs, kwargs = group_kwargs('d_', **kwargs)
    c_kwargs, kwargs = group_kwargs('c_', **kwargs)

    return dict(
        obj=o_kwargs,
        hist=h_kwargs,
        ratio=r_kwargs,
        difference=d_kwargs,
        correlation=c_kwargs,
        remaining=kwargs
    )

def _add_new_axis(ax, size='20%', sharex=True, pad=0.1, **kwargs):
    if not hasattr(ax, 'divider'): 
        ax.divider = make_axes_locatable(ax)
        sub_ax = ax
    else:
        prev = ax.sub_ax if hasattr(ax, 'sub_ax') else ax
        if sharex: prev.get_xaxis().set_visible(0)
        sub_ax = ax.divider.append_axes("bottom", size=size, pad=pad, sharex=prev if sharex else None, **kwargs)
        ax.sub_ax = sub_ax

    return ax, sub_ax

def _flatten_plotobjs(plotobjs):
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = []
    for plotobj in plotobjs:
        if isinstance(plotobj, Stack): 
            histobjs.append(plotobj)
        else:
            for hist in plotobj:
                histobjs.append(hist)
    return histobjs
                
def _group_objs(histobjs):
    # default group
    # -- First check for data
    x = next(filter(lambda hist : hist.is_data, histobjs), None)
    # -- If no data found, take the first histogram as the xerator
    if x is None: 
        x = next(filter(lambda hist: not isinstance(hist,Stack), histobjs))
    x = histobjs.index(x)
    ys = [ y for y in range(len(histobjs)) if y != x ]
    group = [ (x, ys) ]
    return group

def _plot_objects(figax, plotobjs, size='20%', sharex=True, pad=0.1, errors=True, **kwargs):
    fig, ax = figax
    ax, sub_ax = _add_new_axis(ax, size=size, sharex=sharex, pad=pad)

    for plotobj in plotobjs: 
        if isinstance(plotobj,DataList): graph_histos(plotobj,figax=(fig, sub_ax))
        elif isinstance(plotobj, Stack): plot_stack(plotobj,figax=(fig, sub_ax))
        elif isinstance(plotobj, HistoList): plot_histos(plotobj,errors=errors,figax=(fig, sub_ax))
        elif isinstance(plotobj, Histo): plot_histo(plotobj, errors=errors,figax=(fig,sub_ax))
        elif isinstance(plotobj, GraphList): plot_graphs(plotobj, errors=errors,figax=(fig,sub_ax))
        elif isinstance(plotobj, Graph): plot_graph(plotobj, errors=errors, figax=(fig, sub_ax))
        elif isinstance(plotobj, Function): plot_function(plotobj, figax=(fig, sub_ax))

    format_axes(sub_ax, **kwargs)


def _add_histo(figax, plotobjs, size='20%', sharex=True, pad=0.1, errors=True, xlabel=None, density=False, cumulative=False, **kwargs):
    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)
    kwargs['obj'].update(density=density, cumulative=cumulative)

    # print(plotobjs)
    histos = HistoListFromGraphs(plotobjs, **kwargs['obj'])

    _plot_objects(figax, histos, size=size, sharex=sharex, pad=pad, xlabel=xlabel, errors=errors, density=density, cumulative=cumulative, **kwargs['remaining'])

    

def _add_ratio(figax, plotobjs, show=True, ylim=(0.1, 1.9), ylabel=None, size='20%', sharex=True, grid=True, inv=False, group=None, method=None, label_stat=None, num_transform=None, den_transform=None,**kwargs):
    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)
    
    if ylabel is None:
        ylabel = {
            'g-test':'G-Test'
        }.get(method, 'Ratio')

    kwargs['remaining'].update(dict(ylim=ylim,ylabel=ylabel,grid=grid))
    
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = _flatten_plotobjs(plotobjs)
    if group is None: # default group
        group = _group_objs(histobjs)
        
    ratios = []
    def _plotobj_num(plotobj):
        if not isinstance(plotobj, Stack): return plotobj
    
    def _plotobj_ratio(plotobj, num):
        if isinstance(plotobj,Stack):
            den = plotobj[-1]
            den.histo = plotobj.histo.npy.sum(axis=0)
            den.error = np.sqrt((plotobj.error.npy**2).sum(axis=0))
            return Ratio(num, den,inv=inv,marker='o',color='black', label_stat=label_stat, method=method, num_transform=num_transform, den_transform=den_transform)
        elif isinstance(plotobj,HistoList):
            for den in plotobj:
                if num is den: continue
                return Ratio(num, den,inv=inv,marker='o', label_stat=label_stat, method=method, num_transform=num_transform, den_transform=den_transform)
        elif isinstance(plotobj, Histo):
            return Ratio(num, plotobj,inv=inv,marker='o', label_stat=label_stat, method=method, num_transform=num_transform, den_transform=den_transform)
    
    for num, dens in group:
        num = _plotobj_num( histobjs[num] )
        if isinstance(dens, int): dens = [dens]
        
        for den in map(histobjs.__getitem__, dens):
            ratio = _plotobj_ratio(den, num)
            if ratio is None: continue
            ratios.append(ratio)
        
    if (show):
        _plot_objects(figax, ratios, size=size, sharex=sharex, **kwargs['remaining'])
        # fig, ax = figax
        # ax, sub_ax = _add_new_axis(ax, size=size, sharex=sharex)
        # plot_graphs(ratios, figax=(fig,sub_ax), **kwargs)
    

def _add_difference(figax, plotobjs, show=True, ylim=None, ylabel=None, size='20%', sharex=True, grid=True, inv=False, method=None, label_stat=None, group=None, histo=False, **kwargs):    
    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)
    
    # print(kwargs['hist'])
    if ylabel is None:
        ylabel = {
            'normalize':'$\%\Delta$',
            'standardize':'$\Delta/\sigma$',
            'chi2':'$\chi^2$',
            'r2':'$R^2$'
        }.get(method, '$\Delta$')
        
    if label_stat is None:
        label_stat = {
            'normalize':'y_std',
            'standardize':'y_mean',
            'chi2':'y_sum',
            'r2':'y_sum'
        }.get(method, 'area')


    kwargs['remaining'].update(dict(ylim=ylim,ylabel=ylabel,grid=grid))
    
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = _flatten_plotobjs(plotobjs)
    if group is None: # default group
        group = _group_objs(histobjs)
        
    differences = []
    def _plotobj_num(plotobj):
        if not isinstance(plotobj, Stack): return plotobj
    
    def _plotobj_difference(plotobj, num):
        if isinstance(plotobj,Stack):
            den = plotobj[-1]
            den.histo = plotobj.histo.npy.sum(axis=0)
            den.error = np.sqrt((plotobj.error.npy**2).sum(axis=0))
            return Difference(num, den,inv=inv,marker='o',color='black', method=method, label_stat=label_stat)
        elif isinstance(plotobj,HistoList):
            for den in plotobj:
                if num is den: continue
                return Difference(num, den,inv=inv,marker='o', method=method, label_stat=label_stat)
        elif isinstance(plotobj, Histo):
            return Difference(num, plotobj,inv=inv,marker='o', method=method, label_stat=label_stat)
    
    for num, dens in group:
        num = _plotobj_num( histobjs[num] )
        if isinstance(dens, int): dens = [dens]
        
        for den in map(histobjs.__getitem__, dens):
            difference = _plotobj_difference(den, num)
            if difference is None: continue
            differences.append(difference)


    if (show): 
        _plot_objects(figax, differences, size=size, sharex=sharex, **kwargs['remaining'])
        # fig, ax = figax
        # ax, sub_ax = _add_new_axis(ax, size=size, sharex=sharex)
        # plot_graphs(differences, figax=(fig,sub_ax), **kwargs)

    if (histo): 
        _add_histo(figax, differences, size='50%', sharex=False, pad=0.6, xlabel=ylabel, errors=False, **kwargs['hist'])
        # histos = HistoListFromGraphs(differences, **kwargs['hist'])
        # legend = kwargs['hist'].get('label_stat', None) != None
        # _plot_objects(figax, histos, size='50%', sharex=False, pad=0.6, ylabel='counts', xlabel=ylabel, errors=False, legend=legend)
    
def _add_correlation(figax,plotobjs,show=True,size='75%', ylabel = "", grid=True, inv=False, label_stat='area', group=None, legend=True, **kwargs):

    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)
    kwargs['remaining'].update(dict(grid=grid, ylabel=ylabel))
    
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = _flatten_plotobjs(plotobjs)
    if group is None: # default group
        group = _group_objs(histobjs)
    
    correlations = []
    def _plotobj_x(plotobj):
        if not isinstance(plotobj, Stack): return plotobj
    
    def _plotobj_correlation(plotobj, x):
        if isinstance(plotobj,Stack):
            y = plotobj[-1]
            y.histo = plotobj.histo.npy.sum(axis=0)
            y.error = np.sqrt((plotobj.error.npy**2).sum(axis=0))
            return Graph(x.histo,y.histo,marker='o',color='black', inv=inv, label_stat=label_stat)
        elif isinstance(plotobj,HistoList):
            for y in plotobj:
                if x is y: continue
                return Graph(x.histo,y.histo,marker='o', inv=inv, label_stat=label_stat)
        elif isinstance(plotobj, Histo):
            return Graph(x.histo,plotobj.histo,marker='o', inv=inv, label_stat=label_stat)
    
    for x, ys in group:
        x = _plotobj_x( histobjs[x] )
        if isinstance(ys, int): ys = [ys]
        
        for y in map(histobjs.__getitem__, ys):
            correlation = _plotobj_correlation(y, x)
            if correlation is None: continue
            correlations.append(correlation)
            
    if (show):
        _plot_objects(figax, correlations, size=size, sharex=False, pad=0.5, legend=legend, **kwargs['remaining'])
        # fig, ax = figax
        # ax, sub_ax = _add_new_axis(ax, size=size, sharex=False, pad=0.5)
        # plot_graphs(correlations, figax=(fig,sub_ax), legend=legend, **kwargs)

def hist_multi(arrays, bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None,
                is_data=False, is_signal=False,stacked=False, 
                histo=True, ratio=False, correlation=False, difference=False,
                figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)

    kwargs['hist'].update(dict(scale=scale,lumi=lumi))
    kwargs['ratio'].update(dict(xlabel=kwargs['ratio'].get('xlabel', kwargs['remaining'].get('xlabel', None))))
    kwargs['difference'].update(dict(xlabel=kwargs['difference'].get('xlabel', kwargs['remaining'].get('xlabel', None))))
    kwargs['remaining'].update(dict(density=density, cumulative=cumulative, scale=scale, lumi=lumi))
    
    
    attrs = AttrArray(arrays=arrays,weights=weights, is_data=is_data,is_signal=is_signal, **kwargs['hist']) 
    plotobjs = []
    
    datas,attrs = attrs.split(lambda h : h.is_data)
    if len(datas) > 0: 
        data_kwargs = datas.unzip(datas.fields[1:])
        data_kwargs.update(dict(color='black', marker='o', linestyle='None'))
        datas = DataList(datas.arrays, bins=bins, density=density, cumulative=cumulative, **data_kwargs)
        bins = datas[0].bins
        plotobjs.append(datas)

    if stacked: 
        bkgs,attrs = attrs.split(lambda h : not h.is_signal)
        if len(bkgs) > 0:
            bkg_kwargs = bkgs.unzip(bkgs.fields[1:])
            stack = Stack(bkgs.arrays, bins=bins, density=density, cumulative=cumulative, **bkg_kwargs)
            bins = stack[0].bins
            plotobjs.append(stack)
        
    if len(attrs) > 0:
        histo_kwargs = attrs.unzip(attrs.fields[1:])
        histo_kwargs.update(dict(histtype='step',linewidth=2))
        histos = HistoList(attrs.arrays, bins=bins, density=density, cumulative=cumulative, **histo_kwargs)
        bins = histos[0].bins
        plotobjs.append(histos)
        
    # if (histo): _add_histo((fig,ax),plotobjs, **kwargs)
    if (histo): _plot_objects((fig,ax),plotobjs, **kwargs['remaining'])
    if (ratio): _add_ratio((fig,ax), plotobjs, **kwargs['ratio'])
    if (difference): _add_difference((fig,ax),plotobjs, **kwargs['difference'])
    if (correlation): _add_correlation((fig,ax), plotobjs, **kwargs['correlation'])
        
    return fig,ax

def hist2d_simple(x_array, y_array, 
                is_data=False, is_signal=False,
                stacked=False, ratio=False,
                figax=None, **kwargs):
    return histo2d_arrays(x_array,y_array, figax=figax, **kwargs)