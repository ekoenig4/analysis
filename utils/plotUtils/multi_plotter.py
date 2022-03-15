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

def _add_ratio(figax, plotobjs, ylim=(0.1, 1.9), ylabel='Ratio',size='20%', grid=True, inv=False, **kwargs):
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
    
    # -- First check for data
    num = next(filter(lambda plotobj : isinstance(plotobj,DataList), plotobjs),None)
    # -- If no data found, take the first histogram as the numerator
    if num is None: 
        num = next(filter(lambda plotobj: not isinstance(plotobj,Stack),plotobjs))
        
    # -- Only use the first histogram in the list
    num = num[0]
    
    ratios = []
    
    for plotobj in plotobjs:
        if isinstance(plotobj,Stack):
            den = plotobj[-1]
            den.histo = plotobj.histo.npy.sum(axis=0)
            den.error = np.sqrt((plotobj.error.npy**2).sum(axis=0))
            ratios.append( Ratio(num,den,inv=inv,marker='o',color='black'))
        elif isinstance(plotobj,HistoList):
            for den in plotobj:
                if num is den: continue
                ratios.append( Ratio(num,den,inv=inv,marker='o'))
    plot_graphs(ratios, figax=(fig,sub_ax), **kwargs)
    
def _add_correlation(figax,plotobjs,size='20%', ylabel = "", grid=True, **kwargs):
    fig, ax = figax
    
    if not hasattr(ax, '__first__'): 
        ax.__first__ = True
        sub_ax = ax 
    else:
        divider = make_axes_locatable(ax)
        sub_ax = divider.append_axes("bottom", size=size, pad=0.5)
    
    # --- Configure kwargs ---
    kwargs.update(dict(grid=grid, ylabel=ylabel))
    
    # -- First check for data
    num = next(filter(lambda plotobj : isinstance(plotobj,DataList), plotobjs),None)
    # -- If no data found, take the first histogram as the numerator
    if num is None: 
        num = next(filter(lambda plotobj: not isinstance(plotobj,Stack),plotobjs))
        
    # -- Only use the first histogram in the list
    num = num[0]
    
    correlations = []
    
    for plotobj in plotobjs:
        if isinstance(plotobj,Stack):
            den = plotobj[-1]
            den.histo = plotobj.histo.npy.sum(axis=0)
            den.error = np.sqrt((plotobj.error.npy**2).sum(axis=0))
            correlations.append( Graph(num.histo,den.histo,marker='o',color='black'))
        elif isinstance(plotobj,HistoList):
            for den in plotobj:
                if num is den: continue
                correlations.append( Graph(num.histo,den.histo,marker='o'))
    plot_graphs(correlations, figax=(fig,sub_ax), **kwargs)

def hist_multi(arrays, bins=None, weights=None, density = False, 
                cumulative=False, scale=None, lumi=None,
                is_data=False, is_signal=False,stacked=False, 
                histo=True, ratio=False,correlation=False,
                figax=None, **kwargs):
    if figax is None: figax = plt.subplots()
    fig,ax = figax
    
    # --- Configure kwargs ---
    hist_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('h_') }
    hist_kwargs.update(dict(scale=scale,lumi=lumi))
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('h_') }
    ratio_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('r_') }
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('r_') }
    correlation_kwargs = { key[2:]:value for key,value in kwargs.items() if key.startswith('c_') }
    kwargs = { key:value for key,value in kwargs.items() if not key.startswith('c_') }
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
        
    if (histo): _add_histo((fig,ax),plotobjs,**kwargs)
    if (ratio): _add_ratio((fig,ax), plotobjs, **ratio_kwargs)
    if (correlation): _add_correlation((fig,ax), plotobjs, **correlation_kwargs)
        
    return fig,ax

def hist2d_simple(x_array, y_array, 
                is_data=False, is_signal=False,
                stacked=False, ratio=False,
                figax=None, **kwargs):
    return histo2d_arrays(x_array,y_array, figax=figax, **kwargs)