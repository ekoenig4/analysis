from numpy import correlate
from ..classUtils.AttrArray import AttrArray
from .better_plotter_2d import * 
from .histogram2d import * 
from .graph import * 
from .model import *
from .function import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import itertools 

from .multi_plotter import _configure_kwargs, _add_new_axis, obj_store

def _flatten_plotobjs(plotobjs):
    # -- Flatten PlotObjs to primitive Histo/Stack type
    for plotobj in plotobjs:
        if isinstance(plotobj, Stack2D):
            yield plotobj 
        elif isinstance(plotobj, Histo2DList):
            for obj in plotobj:
                yield obj
        else:
            yield plotobj

def _plot_objects(figax, plotobjs, position='right', size="100%", pad=1, contour=False, interp=False, scatter=False, exe=None, log=False, **kwargs):
    fig, ax = figax
    ax, sub_ax = _add_new_axis(ax, position=position, size=size, sharex=False, sharey=True, pad=pad)

    for plotobj in plotobjs: 
        if isinstance(plotobj, Histo2DList): 
            plot_histo2ds(plotobj,figax=(fig, sub_ax), exe=exe, contour=contour, interp=interp, scatter=scatter, log=log)
        elif isinstance(plotobj, Histo2D): 
            plot_histo2d(plotobj, figax=(fig, sub_ax), exe=exe, contour=contour, interp=interp, scatter=scatter, log=log)

    format_axes(sub_ax, is_2d=True, **kwargs)

def hist2d_multi(x_arrays, y_arrays, x_bins=None, y_bins=None, weights=None, 
                is_data=False, is_signal=False, is_model=False, stacked=False, 
                density=False, cumulative=False, efficiency=False, lumi=None,
                scale=None, overlay=False, figax=None, **kwargs):
    fig, ax = get_figax(figax)

    # --- Configure kwargs ---
    kwargs = _configure_kwargs(**kwargs)
    
    kwargs['hist'].update(dict(scale=scale,lumi=lumi))
    kwargs['remaining'].update(dict(density=density, cumulative=cumulative, efficiency=efficiency, scale=scale, lumi=lumi))

    attrs = AttrArray(x_arrays=x_arrays, y_arrays=y_arrays, weights=weights, is_data=is_data,is_signal=is_signal, is_model=is_model, **kwargs['hist']) 
    plotobjs = []
    
    datas,attrs = attrs.split(lambda h : h.is_data and not h.is_model)
    if len(datas) > 0: 
        data_kwargs = datas.unzip(datas.fields[1:])
        # data_kwargs.update(dict(color='black', marker='o', linestyle='None'))
        datas = Data2DList(datas.x_arrays, x_bins=x_bins, y_bins=y_bins, density=density, cumulative=cumulative, efficiency=efficiency, **data_kwargs)
        x_bins = datas[0].x_bins
        y_bins = datas[0].y_bins
        plotobjs.append(datas)

    if stacked: 
        bkgs,attrs = attrs.split(lambda h : not h.is_signal)
        if len(bkgs) > 0:
            bkg_kwargs = bkgs.unzip(bkgs.fields[1:])
            bkg_kwargs.update(label='MC-Bkg', color='grey')
            stack = Stack2D(bkgs.x_arrays, x_bins=x_bins, y_bins=y_bins, density=density, cumulative=cumulative, efficiency=efficiency, **bkg_kwargs)
            x_bins = stack.x_bins
            y_bins = stack.y_bins
            plotobjs.append(stack)
        
    if len(attrs) > 0:
        histo_kwargs = attrs.unzip(attrs.fields[1:])
        # histo_kwargs.update(dict(histtype='step',linewidth=2))
        histos = Histo2DList(attrs.x_arrays, x_bins=x_bins, y_bins=y_bins, cumulative=cumulative, efficiency=efficiency, **histo_kwargs)
        x_bins = histos[0].x_bins
        y_bins = histos[0].y_bins
        plotobjs.append(histos)

    # store = obj_store(plotobjs)

    histo2ds = list(_flatten_plotobjs(plotobjs))
    total = len(histo2ds)
    # sizes = [1, 2, 3]

    if overlay:
        _plot_objects((fig,ax), histo2ds, **kwargs['remaining'])
    else:
        for i, plotobj in enumerate(histo2ds):
            _plot_objects((fig,ax), [plotobj], **kwargs['remaining'])
    
    # ax.store = store
    return fig, ax

def hist2d_simple(x_array, y_array, 
                is_data=False, is_signal=False, is_model=False,
                stacked=False, ratio=False,
                figax=None, **kwargs):
    return histo2d_array(x_array,y_array, figax=figax, **kwargs)