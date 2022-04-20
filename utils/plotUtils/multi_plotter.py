from ..classUtils.AttrArray import AttrArray
from .better_plotter import * 
from .histogram import * 
from .graph import * 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import itertools 

def _add_histo(figax, plotobjs, size='20%', **kwargs):
    fig, ax = figax

    if not hasattr(ax, '__first__'): 
        ax.__first__ = True
        sub_ax = ax 
    else:
        divider = make_axes_locatable(ax)
        ax.get_xaxis().set_visible(0)
        sub_ax = divider.append_axes("bottom", size=size, pad=0.1, sharex=ax)

    for plotobj in plotobjs: 
        if isinstance(plotobj,DataList): graph_histos(plotobj,figax=(fig, sub_ax))
        elif isinstance(plotobj, Stack): plot_stack(plotobj,figax=(fig, sub_ax))
        elif isinstance(plotobj, HistoList): plot_histos(plotobj,figax=(fig, sub_ax))
        
    format_axes(sub_ax, **kwargs)
    

def _add_ratio(figax, plotobjs, ylim=(0.1, 1.9), ylabel='Ratio',size='20%', grid=True, inv=False, group=None, label_stat=None, num_transform=None, den_transform=None,**kwargs):
    fig, ax = figax
    
    if not hasattr(ax, '__first__'): 
        ax.__first__ = True
        sub_ax = ax 
    else:
        divider = make_axes_locatable(ax)
        ax.get_xaxis().set_visible(0)
        sub_ax = divider.append_axes("bottom", size=size, pad=0.1, sharex=ax)
    
    # --- Configure kwargs ---
    kwargs.update(dict(ylim=ylim,ylabel=ylabel,grid=grid))
    
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = []
    for plotobj in plotobjs:
        if isinstance(plotobj, Stack): 
            histobjs.append(plotobj)
        else:
            for hist in plotobj:
                histobjs.append(hist)
    
    if group is None: # default group
        # -- First check for data
        num = next(filter(lambda hist : hist.is_data, histobjs), None)
        # -- If no data found, take the first histogram as the numerator
        if num is None: 
            num = next(filter(lambda hist: not isinstance(hist,Stack), histobjs))
        num = histobjs.index(num)
        dens = [ den for den in range(len(histobjs)) if den != num ]
        group = [ (num, dens) ]
        
    ratios = []
    def _plotobj_num(plotobj):
        if not isinstance(plotobj, Stack): return plotobj
    
    def _plotobj_ratio(plotobj, num):
        if isinstance(plotobj,Stack):
            den = plotobj[-1]
            den.histo = plotobj.histo.npy.sum(axis=0)
            den.error = np.sqrt((plotobj.error.npy**2).sum(axis=0))
            return Ratio(num, den,inv=inv,marker='o',color='black', label_stat=label_stat, num_transform=num_transform, den_transform=den_transform)
        elif isinstance(plotobj,HistoList):
            for den in plotobj:
                if num is den: continue
                return Ratio(num, den,inv=inv,marker='o', label_stat=label_stat, num_transform=num_transform, den_transform=den_transform)
        elif isinstance(plotobj, Histo):
            return Ratio(num, plotobj,inv=inv,marker='o', label_stat=label_stat, num_transform=num_transform, den_transform=den_transform)
    
    for num, dens in group:
        num = _plotobj_num( histobjs[num] )
        if isinstance(dens, int): dens = [dens]
        
        for den in map(histobjs.__getitem__, dens):
            ratio = _plotobj_ratio(den, num)
            if ratio is None: continue
            ratios.append(ratio)
        
    plot_graphs(ratios, figax=(fig,sub_ax), **kwargs)
    

def _add_difference(figax, plotobjs, ylim=(-2, 2), ylabel='$\Delta/\sigma$',size='20%', grid=True, inv=False, standardize=True, label_stat='chi2', group=None, **kwargs):
    fig, ax = figax
    
    if not hasattr(ax, '__first__'): 
        ax.__first__ = True
        sub_ax = ax 
    else:
        divider = make_axes_locatable(ax)
        ax.get_xaxis().set_visible(0)
        sub_ax = divider.append_axes("bottom", size=size, pad=0.1, sharex=ax)
    
    # --- Configure kwargs ---
    kwargs.update(dict(ylim=ylim,ylabel=ylabel,grid=grid))
    
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = []
    for plotobj in plotobjs:
        if isinstance(plotobj, Stack): 
            histobjs.append(plotobj)
        else:
            for hist in plotobj:
                histobjs.append(hist)
    
    if group is None: # default group
        # -- First check for data
        num = next(filter(lambda hist : hist.is_data, histobjs), None)
        # -- If no data found, take the first histogram as the numerator
        if num is None: 
            num = next(filter(lambda hist: not isinstance(hist,Stack), histobjs))
        num = histobjs.index(num)
        dens = [ den for den in range(len(histobjs)) if den != num ]
        group = [ (num, dens) ]
        
    differences = []
    def _plotobj_num(plotobj):
        if not isinstance(plotobj, Stack): return plotobj
    
    def _plotobj_difference(plotobj, num):
        if isinstance(plotobj,Stack):
            den = plotobj[-1]
            den.histo = plotobj.histo.npy.sum(axis=0)
            den.error = np.sqrt((plotobj.error.npy**2).sum(axis=0))
            return Difference(num, den,inv=inv,marker='o',color='black', standardize=standardize, label_stat=label_stat)
        elif isinstance(plotobj,HistoList):
            for den in plotobj:
                if num is den: continue
                return Difference(num, den,inv=inv,marker='o', standardize=standardize, label_stat=label_stat)
        elif isinstance(plotobj, Histo):
            return Difference(num, plotobj,inv=inv,marker='o', standardize=standardize, label_stat=label_stat)
    
    for num, dens in group:
        num = _plotobj_num( histobjs[num] )
        if isinstance(dens, int): dens = [dens]
        
        for den in map(histobjs.__getitem__, dens):
            difference = _plotobj_difference(den, num)
            if difference is None: continue
            differences.append(difference)
        
    plot_graphs(differences, figax=(fig,sub_ax), **kwargs)
    
def _add_correlation(figax,plotobjs,size='75%', ylabel = "", grid=True, inv=False, label_stat='area', group=None, legend=True, **kwargs):
    fig, ax = figax
    
    if not hasattr(ax, '__first__'): 
        ax.__first__ = True
        sub_ax = ax 
    else:
        divider = make_axes_locatable(ax)
        sub_ax = divider.append_axes("bottom", size=size, pad=0.5)
    
    # --- Configure kwargs ---
    kwargs.update(dict(grid=grid, ylabel=ylabel))
    
    # -- Flatten PlotObjs to primitive Histo/Stack type
    histobjs = []
    for plotobj in plotobjs:
        if isinstance(plotobj, Stack): 
            histobjs.append(plotobj)
        else:
            for hist in plotobj:
                histobjs.append(hist)
                
    if group is None: # default group
        # -- First check for data
        x = next(filter(lambda hist : hist.is_data, histobjs), None)
        # -- If no data found, take the first histogram as the xerator
        if x is None: 
            x = next(filter(lambda hist: not isinstance(hist,Stack), histobjs))
        x = histobjs.index(x)
        ys = [ y for y in range(len(histobjs)) if y != x ]
        group = [ (x, ys) ]
        
    
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
            
    plot_graphs(correlations, figax=(fig,sub_ax), legend=legend, **kwargs)

def hist_multi(arrays, bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None,
                is_data=False, is_signal=False,stacked=False, 
                histo=True, ratio=False,correlation=False, difference=False,
                figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    # --- Configure kwargs ---
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(dict(scale=scale,lumi=lumi))
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }

    ratio_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('r_') }
    ratio_kwargs.update(dict(xlabel=ratio_kwargs.get('xlabel', kwargs.get('xlabel', None))))
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('r_') }

    correlation_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('c_') }
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('c_') }

    difference_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('d_') }
    difference_kwargs.update(dict(xlabel=difference_kwargs.get('xlabel', kwargs.get('xlabel', None))))
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('d_') }

    kwargs.update(dict(density=density, cumulative=cumulative, scale=scale, lumi=lumi))
    
    
    attrs = AttrArray(arrays=arrays,weights=weights, is_data=is_data,is_signal=is_signal, **hist_kwargs) 
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
        
    if (histo): _add_histo((fig,ax),plotobjs, **kwargs)
    if (ratio): _add_ratio((fig,ax), plotobjs, **ratio_kwargs)
    if (difference): _add_difference((fig,ax),plotobjs, **difference_kwargs)
    if (correlation): _add_correlation((fig,ax), plotobjs, **correlation_kwargs)
        
    return fig,ax

def hist2d_simple(x_array, y_array, 
                is_data=False, is_signal=False,
                stacked=False, ratio=False,
                figax=None, **kwargs):
    return histo2d_arrays(x_array,y_array, figax=figax, **kwargs)