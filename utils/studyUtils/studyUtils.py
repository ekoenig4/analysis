#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict

from .studyUtils import *
# from ..plotUtils.plotUtils import graph_multi, boxplot_multi,  plot_barrel_display, plot_endcap_display
from ..plotUtils.multi_plotter import hist_multi, count_multi
from ..plotUtils.multi_plotter_2d import hist2d_simple, hist2d_multi
from ..selectUtils import *
# from ..classUtils.Study import Study, save_fig, format_var
from ..classUtils.Study import save_fig, format_var
from .study_args import _study_args as Study
from ..varConfig import varinfo
from ..utils import loop_iter

from ..plotUtils import obj_store, plot_graph, plot_graphs, Graph

import vector
import matplotlib.pyplot as plt
from tqdm import tqdm
import functools

from .default_args import *


def autodim(nvar, dim=None, flip=False):
    if dim == -1: dim = (-1, nvar)
    if dim == (-1,-1): dim = None
    if nvar % 2 == 1 and nvar != 1:
        nvar += 1
    if dim is not None:
        nrows, ncols = dim
        nrows = nrows if nrows > 0 else nvar//ncols
        ncols = ncols if ncols > 0 else nvar//nrows
    elif nvar == 1:
        nrows, ncols = 1, 1
    elif flip:
        ncols = nvar//2
        nrows = nvar//ncols
    else:
        nrows = int(np.sqrt(nvar))
        ncols = nvar//nrows
    return nrows, ncols

def autosize(size=(-1,-1),dim=(-1, 1)):
    nrows, ncols = dim
    xsize, ysize = size
    if xsize == -1:
        xsize = 16/3 if ncols > 2 else 18/3
    if ysize == -1:
        ysize = 5
    return xsize,ysize

def get_figax(nvar=1, dim=(-1,-1), flip=False, size=(-1,-1)):
    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size,(nrows,ncols))
    return plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(int(xsize*ncols), ysize*nrows),
                        dpi=80)

def cutflow(*args, size=(16, 8), log=1, h_label_stat=None,scale=True,density=False,lumi=2018, **kwargs):
    study = Study(*args, log=log,
                  h_label_stat=h_label_stat, scale=scale, lumi=lumi, **kwargs)
    def get_scaled_cutflow(tree): 
        from ..plotUtils import Histo
        cutflows = [ Histo(cutflow.histo, cutflow.bins, cutflow.error, scale=fn.scale)
                     for cutflow, fn in zip(tree.cutflow, tree.filelist)  ]
        return functools.reduce(Histo.add, cutflows)

    is_mc = [ not tree.is_data for tree in study.selections ]
    
    scaled_cutflows = [get_scaled_cutflow(tree) for tree in study.selections]
    cutflow_labels = max(
        (selection.cutflow_labels for selection in study.selections), key=lambda a: len(a))
    ncutflow = len(cutflow_labels)+1
    
    bins = np.arange(ncutflow)
    
    figax = plt.subplots(figsize=size) if size else None
    
    counts = [ cutflow.histo for cutflow in scaled_cutflows ]
    error = [ cutflow.error for cutflow in scaled_cutflows ]

    study.attrs['xlabel'] = study.attrs.get('xlabel', cutflow_labels)

    fig, ax, _ = count_multi(counts, bins=bins, error=error, h_histtype='step', **study.attrs, figax=figax)

    # if scale is False:
    #     if density: 
    #         flatten_cutflows = [ cutflow/cutflow[0] for cutflow in flatten_cutflows ]
    #     fig,ax = graph_arrays(cutflow_labels,flatten_cutflows,xlabel=cutflow_labels,**study.attrs,figax=figax)
    # else:
    #     fig, ax, _ = hist_multi(cutflow_bins, bins=bins, weights=flatten_cutflows, xlabel=cutflow_labels, h_histtype=[
    #                     "step"]*len(study.selections), **study.attrs, figax=figax)
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)
    
    if study.return_figax:
        return fig,ax


def boxplot(*args, varlist=[], binlist=None, xlabels=None, dim=None, flip=False, **kwargs):
    study = Study(*args, **kwargs)
    nvar = len(varlist)
    binlist = init_attr(binlist, None, nvar)
    xlabels = init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(int((16/3)*ncols), 5*nrows))

    for i, (var, bins, xlabel) in enumerate(varlist):
        if var is None: continue

        bins, xlabel = format_var(var, bins, xlabel)
        hists = study.get_array(var)
        weights = study.get_scale(hists)

        if not isinstance(axs, np.ndarray):
            ax = axs
        else:
            ax = axs.flat[i]

        raise NotImplemented("boxplot multi not implemeted yet bud")
        # boxplot_multi(hists, bins=bins, xlabel=xlabel,
        #               weights=weights, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)
    
    if study.return_figax:
        return fig,axs

def brazil(*args, varlist=[], binlist=None, xlabels=None, dim=(-1,-1), size=(-1,-1), flip=False, figax=None, use_norm=False, **kwargs):
    study = Study(*args, histo=False, limits=True, **kwargs)

    nvar = 1
    binlist = init_attr(binlist, None, nvar)
    xlabels = init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size,(nrows,ncols))
    
    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(int(xsize*ncols), ysize*nrows),
                                dpi=80)
    fig, axs = figax

    var, bins, xlabel = next(varlist)

    bins, xlabel = format_var(var, bins, xlabel)
    hists = study.get_array(var)
    weights = study.get_scale(hists)

    _, _, histos = hist_multi(hists, bins=bins, xlabel=xlabel, weights=weights, **study.attrs, figax=(fig, axs))
    h_signal = histos[-1]

    if use_norm:
        ylabel = '95% CL upper limit on r'
        exp_limits = h_signal.stats.norm_exp_limits.npy.T
    else:
        ylabel = '95% CL upper limit on $\sigma(X\\rightarrow YY\\rightarrow 4H)$ pb'
        exp_limits = h_signal.stats.exp_limits.npy.T

    exp_p2 = exp_limits[2+2]
    exp_p1 = exp_limits[2+1]
    exp = exp_limits[2]
    exp_m1 = exp_limits[2-1]
    exp_m2 = exp_limits[2-2]

    exp_std2_mu = (exp_p2 + exp_m2)/2
    exp_std2_err = (exp_p2 - exp_m2)/2

    exp_std1_mu = (exp_p1 + exp_m1)/2
    exp_std1_err = (exp_p1 - exp_m1)/2

    x = np.arange(len(h_signal))
    def get_mass(h):
        mx, my = h.label.split('_')[1::2]
        return f'({mx}, {my})'
    label = h_signal.apply(get_mass).list
    
    g_exp = Graph(x, exp, color='black', label='Expected', linestyle='--', marker=None)
    g_exp_std1 = Graph(x, exp_std1_mu, yerr=exp_std1_err, color='#00cc00', marker=None, linewidth=0)
    g_exp_std2 = Graph(x, exp_std2_mu, yerr=exp_std2_err, color='#ffcc00', marker=None, linewidth=0)
    plot_graphs([g_exp_std2, g_exp_std1], fill_error=True, fill_alpha=1, figax=(fig, axs))
    plot_graph(g_exp, figax=(fig, axs), legend=True, xlabel=label, ylabel=ylabel, ylim=(0, 0.35))


    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)


def h_quick(*args, varlist=[], binlist=None, xlabels=None, dim=(-1,-1), size=(-1,-1), flip=False, figax=None, **kwargs):
    study = Study(*args, **kwargs)

    nvar = len(varlist)
    binlist = init_attr(binlist, None, nvar)
    xlabels = init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size,(nrows,ncols))
    
    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(int(xsize*ncols), ysize*nrows),
                                dpi=80)
    fig, axs = figax

    it = tqdm(enumerate(varlist),total=nvar) if study.report else enumerate(varlist)
    for i, (var, bins, xlabel) in it:
        if not isinstance(axs, np.ndarray):
            ax = axs
        else:
            ax = axs.flat[i]
            
        if var is None: 
            ax.set_visible(False)
            continue
        
        bins, xlabel = format_var(var, bins, xlabel)
        counts, bins, error = study.get_histogram(var)
        count_multi(counts, bins=bins, xlabel=xlabel,
                    error=error, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    fig.canvas.draw()

    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)
        
    if study.return_figax:
        return fig,axs

def quick(*args, varlist=[], binlist=None, xlabels=None, dim=(-1,-1), size=(-1,-1), flip=False, figax=None, **kwargs):
    study = Study(*args, **kwargs)

    nvar = len(varlist)
    binlist = init_attr(binlist, None, nvar)
    xlabels = init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size,(nrows,ncols))
    
    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(int(xsize*ncols), ysize*nrows),
                                dpi=80)
    fig, axs = figax

    it = tqdm(enumerate(varlist),total=nvar) if study.report else enumerate(varlist)
    for i, (var, bins, xlabel) in it:
        if not isinstance(axs, np.ndarray):
            ax = axs
        else:
            ax = axs.flat[i]
            
        if var is None: 
            ax.set_visible(False)
            continue
        
        bins, xlabel = format_var(var, bins, xlabel)
        hists = study.get_array(var)
        weights = study.get_scale(hists)
        hist_multi(hists, bins=bins, xlabel=xlabel,
                   weights=weights, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    fig.canvas.draw()

    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)
        
    if study.return_figax:
        return fig,axs

def overlay(tree, varlist=[], binlist=None, dim=(-1,-1), size=(-1,-1), xlabels=None, flip=None, figax=None, **kwargs):
    if type(varlist[0]) != list:
        varlist = [varlist]
    study = Study(tree, **kwargs)

    nvar = len(varlist)
    binlist = init_attr(binlist, None, nvar)
    xlabels = init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size,(nrows,ncols))
    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(int(xsize*ncols), ysize*nrows),
                                dpi=80)
    fig, axs = figax

    it = tqdm(enumerate(varlist),total=nvar) if study.report else enumerate(varlist)
    for i, (group, bins, xlabel) in it:
        hists = [study.get_array(var)[0] for var in group]
        weights = [study.get_scale(hists)[0] for var in group]
        # if labels is None:
        #     study.attrs['labels'] = group

        if not isinstance(axs, np.ndarray):
            ax = axs
        else:
            ax = axs.flat[i]
        hist_multi(hists, bins=bins, weights=weights,
                   xlabel=xlabel, **study.attrs, figax=(fig, ax))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)
    
    if study.return_figax:
        return fig,axs

def compare_masks(treelist, bkg=None, varlist=[], masks=[], label=[], h_linestyle=["-","-.","--",":"], figax=None, saveas=None, **kwargs):
    n = len(treelist) + (1 if bkg is not None else 0)
    if figax is None:
      figax = get_figax(n*len(varlist), dim=(-1,n))
    fig, axs = figax

    nmasks = len(masks)
    h_linestyle = init_attr(h_linestyle, None, nmasks)
    _kwargs = defaultdict(list)
    for i,mask in enumerate(masks):
        _kwargs['label'].append(
                    label[i%len(masks)] if any(label)
                    else getattr(mask, '__name__', str(mask))
                    )
        _kwargs['h_linestyle'].append(h_linestyle[i%len(masks)])
    kwargs.update(**_kwargs)

    def get_ax(i, axs=axs):
        if not isinstance(axs, np.ndarray): return axs
        if axs.ndim == 2 and n > 1: return axs[:,i]
        return axs[i]

    for i, sample in enumerate(treelist):
      if bkg is not None: i+=1
      quick(
          [sample]*nmasks,
          masks=masks,
          varlist=varlist,
          text=(0.0, 1.0, sample.sample),
          text_style=dict(ha='left', va='bottom'),
          figax=(fig, get_ax(i)),
          **kwargs,
      )

    if bkg is not None:
        _masks = []
        for mask in masks:
            _masks += [mask]*len(bkg)
        masks = _masks

        kwargs['h_color'] = kwargs.get('h_color', ['grey']*len(bkg)*nmasks)
        quick_region(
            *([bkg]*nmasks),
            masks=masks,    
            varlist=varlist,
            text=(0.0,1.0,'MC-Bkg'),
            text_style=dict(ha='left',va='bottom'),
            figax=(fig, get_ax(0)),
            **kwargs,
        )
    
    if saveas:
        save_fig(fig, saveas)


def compare_masks_by_sample(treelist, bkg=None, varlist=[], masks=[], label=[], figax=None, saveas=None, **kwargs):
    n = len(masks)
    if figax is None:
      figax = get_figax(n*len(varlist), dim=(-1,n))
    fig, axs = figax

    nmasks = len(masks)
    get_ax = lambda i : axs[:,i] if axs.ndim == 2 else axs[i]

    if bkg is not None:
        treelist = treelist + bkg

    for i, mask in enumerate(masks):
        mask_label = label[i % len(masks)] if any(label) \
                     else getattr(mask, '__name__', str(mask))
        quick(
            treelist,
            masks=mask,
            varlist=varlist,
            text=(0.0, 1.0, mask_label),
            text_style=dict(ha='left', va='bottom'),
            figax=(fig, get_ax(i)),
            **kwargs,
        )
    
    if saveas:
        save_fig(fig, saveas)

def compare_masks_v2(treelist, masks=[], label=[], varlist=[], h_linestyle=["-","-.","--",":"], binlist=None, xlabels=None, dim=(-1,-1), size=(-1,-1), flip=False, figax=None, **kwargs):

    _treelist = []
    _kwargs = defaultdict(list)
    for i, mask in enumerate(masks):
        for tree in treelist:
            _treelist.append(tree)
            _kwargs['masks'].append(mask)
            _kwargs['h_linestyle'].append(h_linestyle[i%len(h_linestyle)])
            _kwargs['label'].append(
                label[i%len(masks)] if any(label)
                else getattr(mask, '__name__', str(mask))
                )
    kwargs.update(**_kwargs)

    study = Study(_treelist, **kwargs)

    ntrees = len(treelist)
    samples = [tree.sample for tree in treelist]
    nmasks = len(masks)
    nvar = len(varlist)
    binlist = init_attr(binlist, None, nvar)
    xlabels = init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, (nvar, 1), flip)
    ncols = min(nvar,4)
    xsize, ysize = autosize(size,(nrows, ncols))
    
    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=1,
                                figsize=(int(ntrees*xsize*ncols), ysize*nrows),
                                dpi=80)
    fig, axs = figax

    for i, (var, bins, xlabel) in tqdm(enumerate(varlist),total=nvar):
        if not isinstance(axs, np.ndarray):
            ax = axs
        else:
            ax = axs.flat[i]
            
        if var is None: 
            ax.set_visible(False)
            continue
        
        bins, xlabel = format_var(var, bins, xlabel)

        hists = study.get_array(var)
        weights = study.get_scale(hists)
        attrs = study.attrs

        for j, sample in enumerate(samples):
            _hists = hists[j::ntrees]
            _weights = weights[j::ntrees]
            get_attr = lambda value : value[j::ntrees] if isinstance(value, list) else value
            _attrs = { key:get_attr(value) for key, value in attrs.items() }
            _attrs.update(
                    text=(0.0,1.0, sample),
                    text_style=dict(ha='left',va='bottom'),
            )

            hist_multi(_hists, bins=bins, xlabel=xlabel,
                    weights=_weights, **_attrs, figax=(fig, ax), as_new_plot=True)
    fig.suptitle(study.title)
    fig.tight_layout()
    fig.canvas.draw()

    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)
        
    if study.return_figax:
        return fig,axs


def quick2d(*args, varlist=None, binlist=None, xvarlist=[], yvarlist=[], xbinlist=[], ybinlist=[], dim=None, size=(-1,-1),  flip=False, figax=None, overlay=False, **kwargs):
    study = Study(*args, overlay=overlay, **kwargs)

    if varlist is not None:
        xvarlist=varlist[::2]
        yvarlist=varlist[1::2]
    if binlist is not None:
        xbinlist=binlist[::2]
        ybinlist=binlist[1::2]

    nvar = len(study.selections)
    nplots = len(xvarlist)

    xbinlist = init_attr(xbinlist, None, nplots)
    ybinlist = init_attr(ybinlist, None, nplots)

    # nrows, ncols = autodim(nplots, dim, flip)
    # xsize, ysize = autosize(size,(nrows,ncols))

    # if figax is None:
    #     figax = plt.subplots(nrows=nrows, ncols=ncols,
    #                             figsize=(int(xsize*ncols), ysize*nrows),
    #                             dpi=80)
    # fig, axs = figax
    
    if dim is None:
        dim = (nplots, 1)
    nrows, ncols = autodim(nplots, dim, flip)
    ncols = min(nvar,4) if not overlay else ncols
    xsize, ysize = autosize(size,(nrows, ncols))
    
    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols if overlay else 1,
                                figsize=(int(xsize*ncols), ysize*nrows),
                                dpi=80)
    fig, axs = figax

    # labels = study.attrs.pop("h_label")

    it1 = enumerate(zip(xvarlist, yvarlist, xbinlist, ybinlist))
    for i, (xvar, yvar, xbins, ybins) in tqdm(it1, total=nplots, position=0):
        xbins, xlabel = format_var(xvar, bins=xbins, xlabel=None)
        ybins, ylabel = format_var(yvar, bins=ybins, xlabel=None)
        info = dict(x_bins=xbins, xlabel=xlabel, y_bins=ybins, ylabel=ylabel)

        xhists = study.get_array(xvar)
        yhists = study.get_array(yvar)
        weights = study.get_scale(xhists)

        if not isinstance(axs, np.ndarray):
            ax = axs
        else:
            ax = axs.flat[i]

        if xvar == yvar:
            keys2d = ('contour','interp','scatter','overlay')
            attrs = { key:value for key,value in study.attrs.items() if not key in keys2d }
            attrs['efficiency'] = True
            attrs['exe'] = None
            hist_multi(xhists, bins=xbins, xlabel=xlabel, weights=weights, **attrs, figax=(fig,ax))
        else:
            attrs = dict(study.attrs)
            if overlay and 'legend' in attrs: 
                del attrs['legend']
            hist2d_multi(xhists, yhists, weights=weights, **info, **attrs, figax=(fig,ax))

        # it2 = enumerate(zip(xhists, yhists, weights, labels))
        # for j, (xhist, yhist, weight, label) in tqdm(it2, total=nvar, position=1, leave=False):

        #     if nvar == ncols:
        #         k = i*nvar + j
        #     else:
        #         k = j*nplots + i

        #     study.attrs["h_label"] = label

        #     if not isinstance(axs, np.ndarray):
        #         ax = axs
        #     else:
        #         ax = axs.flat[k]

        #     hist2d_simple(xhist, yhist, weights=weight, **
        #                 info, **study.attrs, figax=(fig, ax))

        fig.suptitle(study.title)
        fig.tight_layout()
    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)
    
    if study.return_figax:
        return fig,axs

def overlay2d(*args, varlist=None, binlist=None, xvarlist=[], yvarlist=[], xbinlist=[], ybinlist=[], dim=(-1,-1), size=(-1,-1),  flip=False, alpha=0.8, cmin=100, **kwargs):
    study = Study(*args, h_label_stat=None, alpha=alpha,cmin=cmin, **kwargs)

    if varlist is not None:
        xvarlist=varlist[::2]
        yvarlist=varlist[1::2]
    if binlist is not None:
        xbinlist=binlist[::2]
        ybinlist=binlist[1::2]

    nvar = len(study.selections)
    nplots = len(xvarlist)

    xbinlist = init_attr(xbinlist, None, nplots)
    ybinlist = init_attr(ybinlist, None, nplots)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size,(nrows,ncols))
    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(int(xsize*ncols), ysize*nrows),
                                dpi=80)
    fig, axs = figax
    labels = study.attrs.pop("h_label")

    for i, (xvar, yvar, xbins, ybins) in enumerate(zip(xvarlist, yvarlist, xbinlist, ybinlist)):
        xbins, xlabel = format_var(xvar, bins=xbins, xlabel=xvar)
        ybins, ylabel = format_var(yvar, bins=ybins, xlabel=yvar)
        info = dict(x_bins=xbins, xlabel=xlabel, y_bins=ybins, ylabel=ylabel)

        xhists = study.get_array(xvar)
        yhists = study.get_array(yvar)

        weights = study.get_scale(xhists)

        
        cmaps = ['Reds','Blues','Greens','Oranges','Greys','Purples']
        cmapiter = loop_iter(cmaps)
    
        for j, (xhist, yhist, weight, label) in enumerate(zip(xhists, yhists, weights, labels)):
            if nvar == ncols:
                k = i*nvar + j
            else:
                k = j*nplots + i

            study.attrs["h_label"] = label

            if not isinstance(axs, np.ndarray):
                ax = axs
            else:
                ax = axs.flat[k]

            hist2d_simple(xhist, yhist, weights=weight, **info, **study.attrs, cmap=next(cmapiter), figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)
    
    if study.return_figax:
        return fig,axs

def pairplot(*args, varlist=[], binlist=None, scatter=True, dim=None, overlay=True, **kwargs):
    nvar = len(varlist)
    binlist = init_attr(binlist, None, nvar)

    IJ = np.stack(np.meshgrid(np.arange(nvar),np.arange(nvar)),axis=2).flatten()
    varlist = [ varlist[i] for i in IJ ]
    binlist = [ binlist[i] for i in IJ ]

    quick2d( 
        *args,
        varlist=varlist,
        binlist=binlist,
        overlay=True, 
        scatter=scatter,
        dim=(-1, nvar),
        **kwargs
    )


def quick_region(*rtrees, varlist=[], binlist=None, xlabels=None, dim=(-1,-1), size=(-1,-1), flip=False, figax=None, **kwargs):
    ftrees = rtrees[0]
    for rt in rtrees[1:]:
        ftrees = ftrees + rt
    study = Study(ftrees, stacked=False, **kwargs)

    nr = [0] + [ len(rt) for rt in rtrees ]
    nr = np.cumsum(nr)

    nvar = len(varlist)
    binlist = init_attr(binlist, None, nvar)
    xlabels = init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size,(nrows,ncols))
    
    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(int(xsize*ncols), ysize*nrows),
                                dpi=80)
    fig, axs = figax

    for i, (var, bins, xlabel) in tqdm(enumerate(varlist), total=nvar):
        if not isinstance(axs, np.ndarray):
            ax = axs
        else:
            ax = axs.flat[i]
            
        if var is None: 
            ax.set_visible(False)
            continue
        
        bins, xlabel = format_var(var, bins, xlabel)
        hists = study.get_array(var)
        weights = study.get_scale(hists)

        hists = [ ak.concatenate(hists[lo:hi]) for lo,hi in zip(nr[:-1],nr[1:]) if hi <= len(hists) ]
        weights = [ ak.concatenate(weights[lo:hi]) for lo,hi in zip(nr[:-1],nr[1:]) if hi <= len(weights) ]

        hist_multi(hists, bins=bins, xlabel=xlabel,
                   weights=weights, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
        
    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)
        
    if study.return_figax:
        return fig,axs


def quick2d_region(*rtrees, varlist=None, binlist=None, xvarlist=[], yvarlist=[], xbinlist=[], ybinlist=[],  dim=(-1,-1), size=(-1,-1),  flip=False, figax=None, **kwargs):
    ftrees = rtrees[0]
    for rt in rtrees[1:]:
        ftrees = ftrees + rt
    study = Study(ftrees, **kwargs)

    nr = [0] + [ len(rt) for rt in rtrees ]
    nr = np.cumsum(nr)

    if varlist is not None:
        xvarlist=varlist[::2]
        yvarlist=varlist[1::2]
    if binlist is not None:
        xbinlist=binlist[::2]
        ybinlist=binlist[1::2]

    nvar = len(rtrees)
    nplots = len(xvarlist)

    xbinlist = init_attr(xbinlist, None, nplots)
    ybinlist = init_attr(ybinlist, None, nplots)

    nrows, ncols = autodim(nvar*nplots, dim, flip)
    xsize, ysize = autosize(size,(nrows,ncols))
    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(int(xsize*ncols), ysize*nrows),
                                dpi=80)
    fig, axs = figax
    labels = study.attrs.pop("h_label")

    for i, (xvar, yvar, xbins, ybins) in enumerate(zip(xvarlist, yvarlist, xbinlist, ybinlist)):
        xbins, xlabel = format_var(xvar, bins=xbins, xlabel=xvar)
        ybins, ylabel = format_var(yvar, bins=ybins, xlabel=yvar)
        info = dict(x_bins=xbins, xlabel=xlabel, y_bins=ybins, ylabel=ylabel)

        xhists = study.get_array(xvar)
        yhists = study.get_array(yvar)

        weights = study.get_scale(xhists)
        
        xhists = [ ak.concatenate(xhists[lo:hi]) for lo,hi in zip(nr[:-1],nr[1:]) if hi <= len(xhists)  ]
        yhists = [ ak.concatenate(yhists[lo:hi]) for lo,hi in zip(nr[:-1],nr[1:]) if hi <= len(yhists)  ]
        weights = [ ak.concatenate(weights[lo:hi]) for lo,hi in zip(nr[:-1],nr[1:]) if hi <= len(weights)  ]
        
        for j, (xhist, yhist, weight, label) in enumerate(zip(xhists, yhists, weights, labels)):
            if nvar == ncols:
                k = i*nvar + j
            else:
                k = j*nplots + i

            study.attrs["h_label"] = label

            if not isinstance(axs, np.ndarray):
                ax = axs
            else:
                ax = axs.flat[k]
            
            hist2d_simple(xhist, yhist, weights=weight, **
                        info, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)
    
    if study.return_figax:
        return fig,axs


def table(*args, varlist=[], binlist=None, xlabels=None, dim=(-1,-1), size=(-1,-1), flip=False, figax=None,tablef=None, **kwargs):
    study = Study(*args, table=True, tablef=tablef, **kwargs)

    nvar = len(varlist)
    binlist = init_attr(binlist, None, nvar)
    xlabels = init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    xsize, ysize = autosize(size,(1,1))

    for i, (var, bins, xlabel) in tqdm(enumerate(varlist),total=nvar):
        fig, ax = plt.subplots(figsize=(int(xsize), int(ysize)))
        
        bins, xlabel = format_var(var, bins, xlabel)
        hists = study.get_array(var)
        weights = study.get_scale(hists)
        hist_multi(hists, bins=bins, xlabel=xlabel,
                   weights=weights, **study.attrs, figax=(fig, ax))
        fig.canvas.draw()
        study.table(var, xlabel, figax=(fig,ax), **study.attrs)
    plt.close()

def njets(*args, **kwargs):
    study = Study(*args, **kwargs)

    nrows, ncols = 1, 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5))

    weights = study.get_array("scale")

    varlist = ["n_jet", "nloose_btag", "nmedium_btag", "ntight_btag"]

    for i, var in enumerate(varlist):
        tree_vars = study.get_array(var)
        maxjet = int(max(ak.max(var) for var in tree_vars))
        hist_multi(tree_vars, weights=weights, bins=range(maxjet+3),
                   xlabel=var, figax=(fig, axs[i]), **study.attrs)

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, "njets", study.saveas)


def jets(*args, **kwargs):
    study = Study(*args, **kwargs)

    varlist = ["jet_pt", "jet_btag", "jet_eta", "jet_phi", "jet_qgl"]

    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 10))

    for i, varname in enumerate(varlist):
        hists = study.get_array(varname)
        weights = study.get_scale(varname)
        info = varinfo[varname]
        hist_multi(hists, weights=weights, **info,
                   figax=(fig, axs[i//ncols, i % ncols]), **study.attrs)

    n_jet_list = study.get_array("n_jet")
    hist_multi(n_jet_list, bins=range(12), weights=weights,
               xlabel="N Jet", figax=(fig, axs[1, 2]), **study.attrs)

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, "jets", study.saveas)


def ijets(*args, njets=6, **kwargs):
    study = Study(*args, **kwargs)

    varlist = ["jet_pt", "jet_btag", "jet_eta", "jet_phi"]
    weights = study.get_array("scale")

    for ijet in range(njets):
        nrows, ncols = 1, 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5))

        for i, varname in enumerate(varlist):
            hists = [var[:, ijet] for var in study.get_array(varname)]
            info = varinfo[varname]
            hist_multi(hists, weights=weights, **info,
                       figax=(fig, axs[i]), **study.attrs)

        fig.suptitle(f"{ordinal(ijet+1)} Jet Distributions")
        fig.tight_layout()
        plt.show()
        if study.saveas:
            save_fig(fig, "ijets", f"jet{ijet}_{study.saveas}")


def higgs(*args, **kwargs):
    study = Study(*args, **kwargs)

    varlist = ["higgs_pt", "higgs_m", "higgs_eta", "higgs_phi"]

    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 10))

    for i, varname in enumerate(varlist):
        hists = study.get_array(varname)
        weights = study.get_scale(varname)
        info = varinfo[varname]
        hist_multi(hists, weights=weights, **info,
                   figax=(fig, axs[i//ncols, i % ncols]), **study.attrs)

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, "higgs", study.saveas)


def ihiggs(*args, nhiggs=3, **kwargs):
    study = Study(*args, **kwargs)

    varlist = ["higgs_pt", "higgs_m", "higgs_eta", "higgs_phi"]

    weights = [selection["scale"] for selection in study.selections]
    for ihigg in range(nhiggs):

        nrows, ncols = 1, 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5))
        for i, varname in enumerate(varlist):
            hists = [var[:, ihigg] for var in study.get_array(varname)]
            info = varinfo[varname]
            hist_multi(hists, weights=weights, **info,
                       figax=(fig, axs[i]), **study.attrs)

        fig.suptitle(f"{ordinal(ihigg+1)} Higgs Distributions")
        fig.tight_layout()
        plt.show()
        if study.saveas:
            save_fig(fig, "ihiggs", f"higgs{ihigg}_{study.saveas}")


def njet_var_sum(*args, variable="jet_btag", start=3, **kwargs):
    study = Study(*args, **kwargs)

    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 10))
    info = varinfo[variable]
    binmax = info['bins'][-1]

    weights = [selection["scale"] for selection in study.selections]
    selection_vars = [ak.fill_none(ak.pad_none(
        selection[variable], 6, axis=-1, clip=1), 0) for selection in study.selections]
    for i in range(4):
        ijet = i+start
        ijet_var_sum = [ak.sum(var[:, :ijet], axis=-1)
                        for var in selection_vars]

        varstd = max([ak.std(var, axis=None) for var in ijet_var_sum])
        varavg = max([ak.mean(var, axis=None) for var in ijet_var_sum])

        bins = np.linspace(varavg-varstd, varavg+varstd, 50)
        if variable == "jet_btag":
            bins = np.linspace(0, binmax*ijet, 50)

        hist_multi(ijet_var_sum, weights=weights, bins=bins, **study.attrs,
                   xlabel=f"{ijet} {info['xlabel']} Sum", figax=(fig, axs[i//ncols, i % ncols]))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, f"n{variable}_sum", study.saveas)


def jet_display(*args, ie=0, printout=[], boosted=False, **kwargs):
    study = Study(*args, title="", **kwargs)
    tree = study.selections[0]

    for out in printout:
        print(f"{out}: {tree[out][ie]}")

    njet = tree["n_jet"][ie]
    jet_pt = tree["jet_pt"][ie][np.newaxis]
    jet_eta = tree["jet_eta"][ie][np.newaxis]
    jet_phi = tree["jet_phi"][ie][np.newaxis]
    jet_m = tree["jet_m"][ie][np.newaxis]

    if boosted:
        boost = com_boost_vector(jet_pt, jet_eta, jet_phi, jet_m, njet=njet)
        boosted_jets = vector.obj(
            pt=jet_pt, eta=jet_eta, phi=jet_phi, m=jet_m).boost_p4(boost)
        jet_pt, jet_eta, jet_phi, jet_m = boosted_jets.pt, boosted_jets.eta, boosted_jets.phi, boosted_jets.m

    fig = plt.figure(figsize=(10, 5))
    plot_barrel_display(jet_eta, jet_phi, jet_pt,
                        figax=(fig, fig.add_subplot(1, 2, 1)))
    plot_endcap_display(jet_eta, jet_phi, jet_pt, figax=(
        fig, fig.add_subplot(1, 2, 2, projection='polar')))

    r, l, e, id = [tree[info][ie]
                   for info in ("Run", "Event", "LumiSec", "sample_id")]
    sample = tree.samples[id]

    title = f"{sample} | Run: {r} | Lumi: {l} | Event: {e}"
    if boosted:
        title = f"Boosted COM: {title}"
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    if study.saveas:
        save_fig(fig, "jet_display", study.saveas)


def jet_sphericity(*args, **kwargs):
    study = Study(*args, **kwargs)

    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 10))
    shapes = ["M_eig_w1", "M_eig_w2", "M_eig_w3",
              "event_S", "event_St", "event_A"]
    weights = [selection["scale"] for selection in study.selections]
    for i, shape in enumerate(shapes):
        shape_var = [selection[shape] for selection in study.selections]
        info = varinfo[shape]
        hist_multi(shape_var, weights=weights, **info, **
                   study.attrs, figax=(fig, axs[i//ncols, i % ncols]))
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, "sphericity", study.saveas)


def jet_thrust(*args, **kwargs):
    study = Study(*args, **kwargs)

    nrows, ncols = 1, 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5))
    shapes = ["thrust_phi", "event_Tt", "event_Tm"]
    weights = [selection["scale"] for selection in study.selections]

    for i, shape in enumerate(shapes):
        shape_var = [selection[shape] for selection in study.selections]
        info = varinfo[shape]
        hist_multi(shape_var, weights=weights, **info,
                   **study.attrs, figax=(fig, axs[i]))
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, "thrust", study.saveas)
