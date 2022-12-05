from ..classUtils.AttrArray import AttrArray
from .better_plotter import * 
from .histogram import * 
from .graph import * 
from .model import *
from .function import *
from .extension import obj_store
from mpl_toolkits.axes_grid1 import make_axes_locatable

def group_kwargs(prefix, **kwargs):
    grouped_kwargs = { key[len(prefix):]:value for key,value in kwargs.items() if key.startswith(prefix) }
    remaining_kwargs = { key:value for key,value in kwargs.items() if not key.startswith(prefix) }
    return grouped_kwargs, remaining_kwargs

def _configure_kwargs(**kwargs):
    # --- Configure kwargs ---


    o_kwargs, kwargs = group_kwargs('o_', **kwargs)
    h_kwargs, kwargs = group_kwargs('h_', **kwargs)
    r_kwargs, kwargs = group_kwargs('r_', **kwargs)
    d_kwargs, kwargs = group_kwargs('d_', **kwargs)
    c_kwargs, kwargs = group_kwargs('c_', **kwargs)
    e_kwargs, kwargs = group_kwargs('e_', **kwargs)
    l_kwargs, kwargs = group_kwargs('l_', **kwargs)

    return dict(
        obj=o_kwargs,
        hist=h_kwargs,
        ratio=r_kwargs,
        difference=d_kwargs,
        correlation=c_kwargs,
        empirical=e_kwargs,
        limits=l_kwargs,
        remaining=kwargs
    )

def _add_new_axis(ax, position='bottom', size='20%', sharex=True, sharey=False, pad=0.1, **kwargs):
    if not hasattr(ax, 'divider'): 
        ax.divider = make_axes_locatable(ax)
        sub_ax = ax
    else:
        prev = ax.sub_ax if hasattr(ax, 'sub_ax') else ax
        if sharex: prev.get_xaxis().set_visible(0)
        sub_ax = ax.divider.append_axes(position, size=size, pad=pad, sharex=prev if sharex else None, sharey=prev if sharey else None, **kwargs)
        ax.sub_ax = sub_ax

    return ax, sub_ax

def _flatten_plotobjs(plotobjs):
    # -- Flatten PlotObjs to primitive Histo/Stack type
    for plotobj in plotobjs:
        if isinstance(plotobj, Stack):
            yield plotobj 
        elif isinstance(plotobj, HistoList) or isinstance(plotobj, GraphList):
            for obj in plotobj:
                yield obj
        else:
            yield plotobj
                
def _group_objs(histobjs):
    # default group
    # -- First check for data
    x = next(filter(lambda hist : getattr(hist,'is_data',False), histobjs), None)
    # -- If no data found, take the first histogram as the xerator
    if x is None: 
        x = next(filter(lambda hist: not isinstance(hist,Stack), histobjs))
    x = histobjs.index(x)
    ys = [ y for y in range(len(histobjs)) if y != x ]
    group = [ (x, ys) ]
    return group

def _plot_objects(figax, plotobjs, position='bottom', size='20%', sharex=True, sharey=False, pad=0.1, errors=True, fill_error=False, exe=None, as_new_plot=False, **kwargs):
    if as_new_plot:
        position, size, pad, sharex, sharey = 'right', '100%', 1, False, True

    fig, ax = figax
    ax, sub_ax = _add_new_axis(ax, position=position, size=size, sharex=sharex, sharey=sharey, pad=pad)

    for plotobj in plotobjs: 
        if isinstance(plotobj,DataList): 
            graph_histos(plotobj,figax=(fig, sub_ax), exe=exe)
        elif isinstance(plotobj, Stack): 
            plot_stack(plotobj,figax=(fig, sub_ax), exe=exe)
        elif isinstance(plotobj, HistoList): 
            plot_histos(plotobj,errors=errors, fill_error=fill_error, figax=(fig, sub_ax), exe=exe)
        elif isinstance(plotobj, Histo): 
            plot_histo(plotobj, errors=errors, fill_error=fill_error, figax=(fig,sub_ax), exe=exe)
        elif isinstance(plotobj, GraphList): 
            plot_graphs(plotobj, errors=errors, fill_error=fill_error, figax=(fig,sub_ax), exe=exe)
        elif isinstance(plotobj, Graph): 
            plot_graph(plotobj, errors=errors, fill_error=fill_error, figax=(fig, sub_ax), exe=exe)
        elif isinstance(plotobj, Function): 
            plot_function(plotobj, figax=(fig, sub_ax), exe=exe)

    format_axes(sub_ax, **kwargs)

def _store_objects(store, plotobjs):
    if store is None: return

    store.append(
        [ obj for obj in _flatten_plotobjs(plotobjs) ]
    )

def _add_histo(figax, plotobjs, store=None, size='50%', sharex=False, pad=0.6, errors=True, xlabel=None, density=False, cumulative=False, efficiency=False, **kwargs):
    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)
    kwargs['obj'].update(density=density, cumulative=cumulative, efficiency=efficiency)

    def _to_histo_(plotobj):
        if isinstance(plotobj, Graph): return HistoFromGraph(plotobj, **kwargs['obj'])
        if isinstance(plotobj, GraphList): return HistoListFromGraphs(plotobj, **kwargs['obj'])
        return plotobj

    histos = [ _to_histo_(plotobj) for plotobj in plotobjs ]
    _store_objects(store, histos)

    _plot_objects(figax, histos,  size=size, sharex=sharex, pad=pad, xlabel=xlabel, errors=errors, density=density, cumulative=cumulative, efficiency=efficiency, **kwargs['remaining'])

def _add_limits(figax, plotobjs, xy=(0.05,0.95), xycoords='axes fraction', poi=np.linspace(0,2,21), saveas=None, **kwargs):
    # --- Configure kwargs ---d
    kwargs = _configure_kwargs(**kwargs)

    h_sigs, h_bkgs, h_data = [], [], None
    for plotobj in _flatten_plotobjs(plotobjs):
        plotobj.set_label(None)
        if isinstance(plotobj, Stack): h_bkgs = plotobj
        elif plotobj.is_signal: h_sigs.append(plotobj)
        elif plotobj.is_bkg:    h_bkgs.append(plotobj)
        elif plotobj.is_data:   h_data = plotobj

    models = [ Model(h_sig, h_bkgs, h_data) for h_sig in h_sigs ]

    for model in models:
        model.upperlimit(poi=poi)
        model.h_sig.set_label('exp_lim')
        if saveas:
            model.export_to_root(saveas=saveas)
        

    
def _add_ratio(figax, plotobjs, store=None, show=True, ylim=(0.1, 1.9), ylabel=None, size='20%', sharex=True, grid=True, inv=False, group=None, method=None, label_stat=None, num_transform=None, den_transform=None, histo=False, **kwargs):
    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)
    
    if ylabel is None:
        ylabel = {
            'g-test':'G-Test'
        }.get(method, 'Ratio')

    kwargs['remaining'].update(dict(ylim=ylim,ylabel=ylabel,grid=grid))
    kwargs['obj'].update(inv=inv, label_stat=label_stat, method=method, num_transform=num_transform, den_transform=den_transform)
    # kwargs['obj']['marker'] = kwargs['obj'].get('marker','o')
    
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = list(_flatten_plotobjs(plotobjs))
    if group is None: # default group
        group = _group_objs(histobjs)
    
    ratios = []
    def _plotobj_num(plotobj):
        if isinstance(plotobj, Stack): return plotobj.get_histo()
        return plotobj
    
    def _plotobj_ratio(plotobj, num):
        if isinstance(plotobj,Stack):
            den = plotobj[-1]
            den.histo = plotobj.histo.npy.sum(axis=0)
            den.error = np.sqrt((plotobj.error.npy**2).sum(axis=0))
            return Ratio(num, den, color='black', **kwargs['obj'])
        elif isinstance(plotobj,HistoList):
            for den in plotobj:
                if num is den: continue
                return Ratio(num, den, **kwargs['obj'])
        return Ratio(num, plotobj, **kwargs['obj'])
    
    for num, dens in group:
        num = _plotobj_num( histobjs[num] )
        if isinstance(dens, int): dens = [dens]
        
        for den in map(histobjs.__getitem__, dens):
            ratio = _plotobj_ratio(den, num)
            if ratio is None: continue
            ratios.append(ratio)

    _store_objects(store, ratios)

    if (show):
        _plot_objects(figax, ratios, size=size, sharex=sharex, **kwargs['remaining'])

    _multi_driver(plotobjs, kwargs, figax=figax, histo=histo)

def _add_difference(figax, plotobjs, store=None, show=True, ylim=None, ylabel=None, size='20%', sharex=True, grid=True, inv=False, method=None, label_stat=None, group=None, histo=False, **kwargs):    
    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)
    
    if ylabel is None:
        ylabel = {
            'normalize':'$\%\Delta$',
            'standardize':'$\Delta/\sigma(\Delta)$',
            'stderr':'$\Delta/\epsilon$',
            'chi2':'$\chi^2$',
            'r2':'$R^2$'
        }.get(method, '$\Delta$')
        
    if label_stat is None:
        label_stat = {
            'normalize':'y_std',
            'standardize':'y_mean',
            'chi2':'y_sum',
            'r2':'y_sum',
            'ks':'ks'
        }.get(method, 'area')


    kwargs['remaining'].update(dict(ylim=ylim,ylabel=ylabel,grid=grid))
    kwargs['obj'].update(inv=inv,method=method, label_stat=label_stat)
    # kwargs['obj']['marker'] = kwargs['obj'].get('marker','o')
    
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = list(_flatten_plotobjs(plotobjs))
    if group is None: # default group
        group = _group_objs(histobjs)
    
    differences = []
    def _plotobj_num(plotobj):
        if isinstance(plotobj, Stack): return plotobj.get_histo()
        return plotobj
    
    def _plotobj_difference(plotobj, num):
        if isinstance(plotobj,Stack):
            den = plotobj[-1]
            den.histo = plotobj.histo.npy.sum(axis=0)
            den.error = np.sqrt((plotobj.error.npy**2).sum(axis=0))
            return Difference(num, den, color='black', **kwargs['obj'])
        elif isinstance(plotobj,HistoList):
            for den in plotobj:
                if num is den: continue
                return Difference(num, den, **kwargs['obj'])
        return Difference(num, plotobj, **kwargs['obj'])
    
    for num, dens in group:
        num = _plotobj_num( histobjs[num] )
        if isinstance(dens, int): dens = [dens]
        
        for den in map(histobjs.__getitem__, dens):
            difference = _plotobj_difference(den, num)
            if difference is None: continue
            differences.append(difference)

    _store_objects(store, differences)

    if (show): 
        _plot_objects(figax, differences, size=size, sharex=sharex, **kwargs['remaining'])

    _multi_driver(plotobjs, kwargs, figax=figax, histo=histo)

def _add_empirical(figax, plotobjs, store=None, show=True, ylim=(0,1.05), ylabel=None, size='20%', sharex=True, grid=True, inv=False, sf=False, label_stat=None, difference=False, correlation=False, **kwargs):    
    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)
    
    if ylabel is None:
        ylabel = 'ECDF'

    kwargs['remaining'].update(dict(ylim=ylim,ylabel=ylabel,grid=grid))
    kwargs['obj'].update(inv=inv, label_stat=label_stat)

    kwargs['correlation'].update(
            method=kwargs['correlation'].get('method','ad'),
            legend=kwargs['correlation'].get('legend', True)
        )
    
    kwargs['difference'].update(
            method=kwargs['difference'].get('method','ks'),
            legend=kwargs['difference'].get('legend', True),
            xlabel=kwargs['difference'].get('xlabel', kwargs['remaining'].get('xlabel', None))
        )

    if difference or correlation: size = "50%"
    
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = list(_flatten_plotobjs(plotobjs))

    def _get_ecdf(plotobj):
        if isinstance(plotobj, Stack): plotobj = plotobj.get_histo()
        return plotobj.ecdf(sf=sf)
        
    empiricals = [ _get_ecdf(hist) for hist in histobjs ]
    _store_objects(store, empiricals)

    if (show): 
        _plot_objects(figax, empiricals, size=size, sharex=sharex, **kwargs['remaining'])
    _multi_driver(empiricals, kwargs, figax=figax, difference=difference, correlation=correlation)
    
def _add_correlation(figax,plotobjs, store=None,show=True,size='75%', grid=True, inv=False, group=None, legend=True, method=None, label_stat=None, **kwargs):

    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)
    kwargs['remaining'].update(dict(
        grid=grid,
        ylabel=kwargs['remaining'].get('ylabel', {'ad':r'$\frac{dA^2}{dH_n}$'}.get(method, '')),
        xlabel=kwargs['remaining'].get('xlabel', {'ad':'Hn'}.get(method, None))
        ))

    if label_stat is None:
        label_stat = {
            'ad':'area',
        }.get(method, None)
    
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = list(_flatten_plotobjs(plotobjs))
    if group is None: # default group
        group = _group_objs(histobjs)
    
    correlations = []
    def _plotobj_x(plotobj):
        if isinstance(plotobj, Stack): return plotobj.get_histo()
        return plotobj
    
    def _plotobj_correlation(plotobj, x):
        if isinstance(plotobj,Stack):
            y = plotobj.get_histo()
            return Correlation(x, y,color='black', inv=inv, label_stat=label_stat, method=method)
        elif isinstance(plotobj,HistoList):
            for y in plotobj:
                if x is y: continue
                return Correlation(x, y, inv=inv, label_stat=label_stat, method=method)
        return Correlation(x,plotobj, inv=inv, label_stat=label_stat, method=method)
    
    for x, ys in group:
        x = _plotobj_x( histobjs[x] )
        if isinstance(ys, int): ys = [ys]
        
        for y in map(histobjs.__getitem__, ys):
            correlation = _plotobj_correlation(y, x)
            if correlation is None: continue
            correlations.append(correlation)
    _store_objects(store, correlations)
            
    if (show):
        _plot_objects(figax, correlations, size=size, sharex=False, pad=0.5, legend=legend, **kwargs['remaining'])

    _multi_driver(plotobjs, kwargs, figax=figax)

def _multi_driver(plotobjs, kwargs, histo=False, ratio=False, difference=False, empirical=False, correlation=False, figax=None):
    fig, ax = get_figax(figax)

    if (histo): _add_histo((fig,ax),plotobjs, **kwargs['hist'])
    if (ratio): _add_ratio((fig,ax), plotobjs, **kwargs['ratio'])
    if (difference): _add_difference((fig,ax),plotobjs, **kwargs['difference'])
    if (empirical): _add_empirical((fig,ax), plotobjs, **kwargs['empirical'])
    if (correlation): _add_correlation((fig,ax), plotobjs, **kwargs['correlation'])

def hist_multi(arrays, bins=None, weights=None, density = False, efficiency=False,
                cumulative=False, scale=None, lumi=None, plot_scale=1, store=None,
                is_data=False, is_signal=False, is_model=False, stacked=False, stack_fill=False,
                histo=True, ratio=False, correlation=False, difference=False, empirical=False, limits=False, 
                as_new_plot=False, figax=None, **kwargs):
    fig, ax = get_figax(figax)
    
    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)

    kwargs['hist'].update(dict(scale=scale,lumi=lumi))
    kwargs['ratio'].update(dict(xlabel=kwargs['ratio'].get('xlabel', kwargs['remaining'].get('xlabel', None))))
    kwargs['difference'].update(dict(xlabel=kwargs['difference'].get('xlabel', kwargs['remaining'].get('xlabel', None))))
    kwargs['empirical'].update(dict(xlabel=kwargs['empirical'].get('xlabel', kwargs['remaining'].get('xlabel', None))))
    kwargs['correlation'].update(dict(label_stat=kwargs['correlation'].get('label_stat','area')))
    kwargs['remaining'].update(dict(density=density, cumulative=cumulative, efficiency=efficiency, scale=scale, lumi=lumi))
    
    attrs = AttrArray(arrays=arrays,weights=weights, is_data=is_data,is_signal=is_signal, is_model=is_model, **kwargs['hist']) 
    plotobjs = []
    
    datas,attrs = attrs.split(lambda h : h.is_data and not h.is_model)
    if len(datas) > 0: 
        data_kwargs = datas.unzip(datas.fields[1:])
        data_kwargs.update(dict(color='black', marker='o', linestyle='None'))
        datas = DataList.from_arrays(datas.arrays, bins=bins, density=density, cumulative=cumulative, efficiency=efficiency, **data_kwargs)
        bins = datas[0].bins
        plotobjs.append(datas)

    if stacked: 
        bkgs,attrs = attrs.split(lambda h : not h.is_signal)
        if len(bkgs) > 0:
            bkg_kwargs = bkgs.unzip(bkgs.fields[1:])
            stack = Stack.from_arrays(bkgs.arrays, bins=bins, density=density, cumulative=cumulative, efficiency=efficiency, stack_fill=stack_fill, **bkg_kwargs)
            bins = stack[0].bins
            plotobjs.append(stack)
        
    if len(attrs) > 0:
        histo_kwargs = attrs.unzip(attrs.fields[1:])
        histo_kwargs.update(dict(histtype='step',linewidth=2))
        histos = HistoList.from_arrays(attrs.arrays, bins=bins, density=density, cumulative=cumulative, efficiency=efficiency, plot_scale=plot_scale, **histo_kwargs)
        bins = histos[0].bins
        plotobjs.append(histos)
        
        
    if (limits): 
        _add_limits((fig,ax), plotobjs, **kwargs['limits'])
    _store_objects(store, plotobjs)

    if (histo): _plot_objects((fig,ax), plotobjs, as_new_plot=as_new_plot, **kwargs['remaining'])
    _multi_driver(plotobjs, kwargs, histo=False, ratio=ratio, difference=difference, empirical=empirical, correlation=correlation, figax=(fig,ax))

    return fig, ax, plotobjs


def count_multi(counts, bins=None, error=None, density = False, efficiency=False,
                cumulative=False, scale=None, lumi=None, plot_scale=1, store=None,
                is_data=False, is_signal=False, is_model=False, stacked=False, stack_fill=False,
                histo=True, ratio=False, correlation=False, difference=False, empirical=False, limits=False, 
                as_new_plot=False, figax=None, **kwargs):
    fig, ax = get_figax(figax)
    
    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)

    kwargs['hist'].update(dict(scale=scale,lumi=lumi))
    kwargs['ratio'].update(dict(xlabel=kwargs['ratio'].get('xlabel', kwargs['remaining'].get('xlabel', None))))
    kwargs['difference'].update(dict(xlabel=kwargs['difference'].get('xlabel', kwargs['remaining'].get('xlabel', None))))
    kwargs['empirical'].update(dict(xlabel=kwargs['empirical'].get('xlabel', kwargs['remaining'].get('xlabel', None))))
    kwargs['correlation'].update(dict(label_stat=kwargs['correlation'].get('label_stat','area')))
    kwargs['remaining'].update(dict(density=density, cumulative=cumulative, efficiency=efficiency, scale=scale, lumi=lumi))
    
    attrs = AttrArray(counts=counts,error=error, is_data=is_data,is_signal=is_signal, is_model=is_model, **kwargs['hist']) 
    plotobjs = []
    
    datas,attrs = attrs.split(lambda h : h.is_data and not h.is_model)
    if len(datas) > 0: 
        data_kwargs = datas.unzip(datas.fields[1:])
        data_kwargs.update(dict(color='black', marker='o', linestyle='None'))
        datas = DataList.from_counts(datas.counts, bins=bins, density=density, cumulative=cumulative, efficiency=efficiency, **data_kwargs)
        bins = datas[0].bins
        plotobjs.append(datas)

    if stacked: 
        bkgs,attrs = attrs.split(lambda h : not h.is_signal)
        if len(bkgs) > 0:
            bkg_kwargs = bkgs.unzip(bkgs.fields[1:])
            stack = Stack.from_counts(bkgs.counts, bins=bins, density=density, cumulative=cumulative, efficiency=efficiency, stack_fill=stack_fill, **bkg_kwargs)
            bins = stack[0].bins
            plotobjs.append(stack)
        
    if len(attrs) > 0:
        histo_kwargs = attrs.unzip(attrs.fields[1:])
        histo_kwargs.update(dict(histtype='step',linewidth=2))
        histos = HistoList.from_counts(attrs.counts, bins=bins, density=density, cumulative=cumulative, efficiency=efficiency, plot_scale=plot_scale, **histo_kwargs)
        bins = histos[0].bins
        plotobjs.append(histos)
        
        
    if (limits): 
        _add_limits((fig,ax), plotobjs, **kwargs['limits'])
    _store_objects(store, plotobjs)

    if (histo): _plot_objects((fig,ax), plotobjs, as_new_plot=as_new_plot, **kwargs['remaining'])
    _multi_driver(plotobjs, kwargs, histo=False, ratio=ratio, difference=difference, empirical=empirical, correlation=correlation, figax=(fig,ax))

    return fig, ax, plotobjs