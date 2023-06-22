#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict

from ..plotUtils import *
from .study_args import _study_args as Study
from ..varConfig import varinfo
from ..utils import loop_iter, GIT_WD, ordinal
from ..classUtils import AttrArray

from ..plotUtils import obj_store, plot_graph, plot_graphs, Graph

import vector, os
import matplotlib.pyplot as plt
from tqdm import tqdm
import functools
from datetime import date

from .default_args import *

date_tag = date.today().strftime("%Y%m%d")
default_base = f"{GIT_WD}/plots/{date_tag}_plots"

def get_path(path, base=default_base):
    outfn = os.path.join(base, path)
    return outfn

def make_path(path, base=default_base):
    outfn = get_path(path, base)
    directory = '/'.join(outfn.split('/')[:-1])
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return outfn

def save_fig(fig, saveas=None, fmt=['jpg']):
    outfn = make_path(saveas)
    print('saving to', outfn)

    if len(outfn.split('.')) == 2:
        outfn, ext = outfn.split('.')
        fmt = [ext]

    if 'pdf' in fmt:
        fig.savefig(f"{outfn}.pdf", format="pdf")
    if 'png' in fmt:
        fig.savefig(f"{outfn}.png", format="png", dpi=400)
    if 'jpg' in fmt:
        fig.savefig(f"{outfn}.jpg", format="jpg", dpi=400)
    # fig.savefig(f"{outfn}.png", format="png")
    plt.clf()
    plt.close()


def format_var(var, bins=None, xlabel=None):
    info = varinfo.find(var)
    if info and bins is None:
        bins = info.get('bins', None)
    if info and xlabel is None:
        xlabel = info.get('xlabel', var)
    if xlabel is None:
        xlabel = var

    if isinstance(xlabel, str) and any(re.findall(r'\[.*\]', xlabel)):
        slice = re.findall(r'\[.*\]', xlabel)[0]
        place = int(next(idx for idx in slice[1:-1].split(',') if idx != ':'))
        xlabel = f'{ordinal(place+1)} {xlabel.replace(slice,"")}'

    return bins, xlabel


def autodim(nvar, dim=None, flip=False):
    if dim == -1:
        dim = (1, nvar)
    elif dim == (-1, -1) or dim is None:
        nrows = int(np.sqrt(nvar))
        ncols = nvar // nrows
        dim = (nrows, ncols)
    else:
        nrows, ncols = dim
        nrows = nrows if nrows > 0 else nvar // ncols
        ncols = ncols if ncols > 0 else nvar // nrows
        dim = (nrows, ncols)

    nrows, ncols = dim
    off = nvar - nrows*ncols
    if off > 0:
        ncols += 1

    if flip:
        nrows, ncols = ncols, nrows

    return nrows, ncols


def autosize(size=(-1, -1), dim=(-1, 1)):
    nrows, ncols = dim
    xsize, ysize = size
    if xsize == -1:
        xsize = 16/3 if ncols > 2 else 18/3
    if ysize == -1:
        ysize = 5
    return xsize, ysize


def get_figax(nvar=1, dim=(-1, -1), flip=False, size=(-1, -1), **kwargs):
    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size, (nrows, ncols))
    return plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(int(xsize*ncols), ysize*nrows),
                        dpi=80, **kwargs)


def cutflow(*args, size=(16, 8), log=1, h_label_stat=None, scale=True, density=False, lumi=2018, **kwargs):
    study = Study(*args, log=log,
                  h_label_stat=h_label_stat, scale=scale, lumi=lumi, **kwargs)

    def get_scaled_cutflow(tree):
        from ..plotUtils import Histo
        cutflows = [Histo(cutflow.histo, cutflow.bins, cutflow.error, scale=fn.scale)
                    for cutflow, fn in zip(tree.cutflow, tree.filelist)]
        return functools.reduce(Histo.add, cutflows)

    is_mc = [not tree.is_data for tree in study.selections]

    scaled_cutflows = [get_scaled_cutflow(tree) for tree in study.selections]
    cutflow_labels = max(
        (selection.cutflow_labels for selection in study.selections), key=lambda a: len(a))
    ncutflow = len(cutflow_labels)+1

    bins = np.arange(ncutflow)

    figax = plt.subplots(figsize=size) if size else None

    counts = [cutflow.histo for cutflow in scaled_cutflows]
    error = [cutflow.error for cutflow in scaled_cutflows]

    study.attrs['xlabel'] = study.attrs.get('xlabel', cutflow_labels)

    fig, ax, _ = count_multi(
        counts, bins=bins, error=error, h_histtype='step', **study.attrs, figax=figax)

    # if scale is False:
    #     if density:
    #         flatten_cutflows = [ cutflow/cutflow[0] for cutflow in flatten_cutflows ]
    #     fig,ax = graph_arrays(cutflow_labels,flatten_cutflows,xlabel=cutflow_labels,**study.attrs,figax=figax)
    # else:
    #     fig, ax, _ = hist_multi(cutflow_bins, bins=bins, weights=flatten_cutflows, xlabel=cutflow_labels, h_histtype=[
    #                     "step"]*len(study.selections), **study.attrs, figax=figax)
    fig.tight_layout()
    if study.saveas:
        save_fig(fig, study.saveas)

    if study.return_figax:
        return fig, ax


def boxplot(*args, varlist=[], binlist=None, xlabels=None, dim=None, flip=False, **kwargs):
    study = Study(*args, **kwargs)
    nvar = len(varlist)
    binlist = AttrArray.init_attr(binlist, None, nvar)
    xlabels = AttrArray.init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(int((16/3)*ncols), 5*nrows))

    for i, (var, bins, xlabel) in enumerate(varlist):
        if var is None:
            continue

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
        return fig, axs

def limits(*args, varlist=[], binlist=[], report=True, parallel=8, poi=np.linspace(0,2,11), **kwargs):
    study = Study(*args, histo=False, **kwargs)

    binlist = AttrArray.init_attr(binlist, None, 1)
    varlist = zip(varlist, binlist)
    var, bins = next(varlist)

    bins, _ = format_var(var, bins)
    hists = study.get_array(var)
    weights = study.get_scale(hists)

    _, _, histos = hist_multi(hists, bins=bins, weights=weights, **study.attrs, figax='none')

    h_bkg, h_sigs = histos
    sig_models = h_sigs.apply( lambda h : Model(h, h_bkg) )

    if parallel:
        if isinstance(parallel, bool): parallel = 4
        parallel = min(len(sig_models), parallel)

        import multiprocessing as mp

        upperlimit = Model.f_upperlimit(poi=poi)
        with mp.Pool(parallel) as pool:
            sig_models.parallel_apply(upperlimit, pool=pool, report=report)
    else:
        sig_models.apply(Model.upperlimit, report=report)

    return sig_models

def brazil_limits(sig_models, xlabel='my', units='fb', ylim=(0,250), grid=True, figax=None, saveas=None):
    from ..plotUtils.better_plotter import plot_model_brazil
    unique_mx = np.unique([ s.mx for s in sig_models])
    unique_my = np.unique([ s.my for s in sig_models])
    

    if xlabel == 'my':
        unique_x = unique_my
        unique_y = unique_mx
        check = lambda s, mx : s.mx == mx
        label = lambda mx : f'Expected for $M_{{X}} = {mx}$ GeV'
        supxlabel = '$M_{Y}$ [GeV]'
    elif xlabel == 'mx':
        unique_x = unique_mx
        unique_y = unique_my
        check = lambda s, my : s.my == my
        label = lambda my : f'Expected for $M_{{Y}} = {my}$ GeV'
        supxlabel = '$M_{X}$ [GeV]'

    if figax is None:
        figax = get_figax(nvar=len(unique_y), dim=(len(unique_y),1), size=(5,1.5), sharex=True, sharey=True)
    fig, axs = figax

    xlim = (min(unique_x)-50, max(unique_x)+50)

    # fig, axs = study.get_figax(nvar=1,)
    for i, y in enumerate( tqdm(unique_y) ):
        y_signal = ObjIter([ s for s in sig_models if check(s, y) ])

        plot_model_brazil(
            y_signal,
            label=label(y),
            xlabel=xlabel,
            units=units,
            ylim=ylim,
            xlim=xlim,
            grid=grid,
            legend=True,
            figax=(fig,axs.flat[i])
        )
        axs.flat[i].set(xlabel=None)

    fig.supxlabel(supxlabel)
    fig.supylabel(f'95% CL upper limit on $\sigma(X\\rightarrow YY\\rightarrow 4H)$ [{units}]')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    if saveas:
        save_fig(fig, saveas)

def brazil2d_limits(sig_models,  units='fb', figax=None, saveas=None, **kwargs):
    if figax is None:
        figax = get_figax()
    fig, ax = figax
    
    mx = np.array([ s.mx for s in sig_models])
    my = np.array([ s.my for s in sig_models])
    exp_limits = np.array([ s.h_sig.stats.exp_limits[2] for s in sig_models])

    unitMap = dict(
        pb=1,
        fb=1e3,
    )
    exp_limits *= unitMap[units]

    xlim = (min(mx), max(mx))
    ylim = (min(my), max(my))

    zlabel = (f'95% CL upper limit on $\sigma(X\\rightarrow YY\\rightarrow 4H)$ [{units}]')

    graph2d_array(mx, my, exp_limits, figax=figax, interp=interp, colorbar=True, zlabel=zlabel, **kwargs)
    graph_array(mx, my, figax=figax, g_color='grey', g_ls='none', g_marker='o', g_markersize=10, xlim=xlim, ylim=ylim, xlabel='$M_{X}$ [GeV]', ylabel='$M_{Y}$ [GeV]')

    if saveas:
        save_fig(fig, saveas)

def brazil(*args, varlist=[], label='Expected', xlabel='mass', ylabel='95% CL upper limit on cross section', binlist=None, dim=(-1, -1), size=(-1, -1), flip=False, figax=None, use_norm=False, units='pb', xlim=None, ylim=None, **kwargs):
    study = Study(*args, histo=False, limits=True, l_report=True, l_parallel=8, **kwargs)

    nvar = 1
    binlist = AttrArray.init_attr(binlist, None, nvar)
    varlist = zip(varlist, binlist)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size, (nrows, ncols))

    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(int(xsize*ncols), ysize*nrows),
                             dpi=80)
    fig, axs = figax

    var, bins = next(varlist)

    bins, _ = format_var(var, bins)
    hists = study.get_array(var)
    weights = study.get_scale(hists)

    _, _, histos = hist_multi(
        hists, bins=bins, weights=weights, **study.attrs, figax=(fig, axs))
    
    h_signal = histos[-1]

    exp_limits = h_signal.stats.exp_limits.npy.T
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
    
    x = h_signal.apply(get_x).npy

    if x.dtype == np.dtype('U1'):
        xlabel = x.tolist()
        x = np.arange(len(x))

    if isinstance(xlabel, str):
        xlabel = dict(
            mx='$M_{X}$ (GeV)',
            my='$M_{Y}$ (GeV)',
        ).get(xlabel, xlabel)

    g_exp = Graph(x, exp, color='black', label=label,
                  linestyle='--', marker=None)
    g_exp_std1 = Graph(x, exp_std1_mu, yerr=exp_std1_err,
                       color='#00cc00', marker=None, linewidth=0)
    g_exp_std2 = Graph(x, exp_std2_mu, yerr=exp_std2_err,
                       color='#ffcc00', marker=None, linewidth=0)
    plot_graphs([g_exp_std2, g_exp_std1], fill_error=True,
                fill_alpha=1, figax=(fig, axs))
    plot_graph(g_exp, figax=(fig, axs), legend=True, grid=True,
               xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    
    fig.tight_layout()

    if study.saveas:
        save_fig(fig, study.saveas)

def brazil2d(*args, varlist=[], binlist=None, xlabels=None, dim=(-1, -1), size=(-1, -1), flip=False, figax=None, use_norm=False, units='pb', **kwargs):
    study = Study(*args, histo=False, limits=True, l_report=True, l_parallel=8, **kwargs)

    nvar = 1
    binlist = AttrArray.init_attr(binlist, None, nvar)
    xlabels = AttrArray.init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size, (nrows, ncols))

    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(int(xsize*ncols), ysize*nrows),
                             dpi=80)
    fig, axs = figax

    var, bins, xlabel = next(varlist)

    bins, xlabel = format_var(var, bins, xlabel)
    hists = study.get_array(var)
    weights = study.get_scale(hists)

    _, _, histos = hist_multi(
        hists, bins=bins, xlabel=xlabel, weights=weights, **study.attrs, figax=(fig, axs))
    h_signal = histos[-1]

    if use_norm:
        ylabel = '95% CL upper limit on r'
        exp_limits = h_signal.stats.norm_exp_limits.npy.T
    else:
        ylabel = f'95% CL upper limit on $\sigma(X\\rightarrow YY\\rightarrow 4H)$ {units}'
        exp_limits = h_signal.stats.exp_limits.npy.T

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

    def get_mxmy(h):
        mx, my = h.label.split('_')[1::2]
        return int(mx), int(my)
    mx, my = h_signal.apply(get_mxmy).npy.T
    xlim = (min(mx), max(mx))
    ylim = (min(my), max(my))

    graph2d_array(mx, my, exp, figax=figax, interp=interp, colorbar=True, zlabel=ylabel)
    graph_array(mx, my, figax=figax, g_color='grey', g_ls='none', g_marker='o', g_markersize=10, xlabel='$M_{X}$ (GeV)', ylabel='$M_{Y}$ (GeV)', xlim=xlim, ylim=ylim)
    
    fig.tight_layout()
    if study.saveas:
        save_fig(fig, study.saveas)
        
def mxmy_phase(signal, f_var, figax=None, interp=True, colorbar=True, g_markersize=10, xlim=None, ylim=None, xlabel=None, ylabel=None, saveas=None, **kwargs):
    if figax is None: figax = get_figax(size=(10,8))
    fig, ax = figax

    mx = signal.mx.npy 
    my = signal.my.npy 

    var = f_var
    if callable(f_var):
        var = signal.apply(f_var).npy
    

    graph2d_array(mx, my, var, figax=figax, interp=interp, colorbar=True, **kwargs)
    graph_array(mx, my, figax=figax, g_color='grey', g_ls='none', g_marker='o', g_markersize=g_markersize, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    
    if saveas:
        save_fig(fig, saveas)

def mxmy_reduction(tree, f_var, figax=None, interp=True, colorbar=True, xlim=None, ylim=None, xlabel=None, ylabel=None, saveas=None, **kwargs):
    if figax is None: figax = get_figax(size=(10,8))
    fig, ax = figax

    mx = tree.mx
    my = tree.my

    var = f_var
    if callable(f_var):
        var = f_var(tree)
    else:
        var = tree[f_var]

    graph2d_array(mx, my, var, figax=figax, interp=interp, colorbar=True, **kwargs)
    graph_array(mx, my, figax=figax, g_color='grey', g_ls='none', g_marker='o', g_markersize=10, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    
    if saveas:
        save_fig(fig, saveas)


def h_quick(*args, varlist=[], binlist=None, xlabels=None, dim=(-1, -1), size=(-1, -1), flip=False, figax=None, **kwargs):
    study = Study(*args, **kwargs)

    nvar = len(varlist)
    binlist = AttrArray.init_attr(binlist, None, nvar)
    xlabels = AttrArray.init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size, (nrows, ncols))

    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(int(xsize*ncols), ysize*nrows),
                             dpi=80)
    fig, axs = figax

    it = tqdm(enumerate(varlist),
              total=nvar) if study.report else enumerate(varlist)
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
        return fig, axs


def group_kwargs(prefix, **kwargs):
    grouped_kwargs = {key[len(prefix):]: value for key,
                      value in kwargs.items() if key.startswith(prefix)}
    remaining_kwargs = {key: value for key,
                        value in kwargs.items() if not key.startswith(prefix)}
    return grouped_kwargs, remaining_kwargs


def statsplot(*args, varlist=[], binlist=None, stat='{stats.mean}', stat_err=None, xlabels=None, dim=(-1, -1), size=(-1, -1), flip=False, figax=None, **kwargs):
    study = Study(*args, **kwargs)

    nvar = len(varlist)
    binlist = AttrArray.init_attr(binlist, None, nvar)
    xlabels = AttrArray.init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size, (nrows, ncols))

    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(int(xsize*ncols), ysize*nrows),
                             dpi=80)
    fig, axs = figax

    g_kwargs, study.attrs = group_kwargs('g_', **study.attrs)

    def get_stat(histos, stat):
        if stat is None:
            return None
        if callable(stat):
            return np.array([stat(h) for h in histos])
        elif any(re.findall(r'{(.*?)}', stat)):
            return np.array([float(stat.format(**vars(h))) for h in histos])
        else:
            return np.array([getattr(h, stat) for h in histos])

    it = tqdm(enumerate(varlist),
              total=nvar) if study.report else enumerate(varlist)
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
        fig, ax, histos = hist_multi(
            hists, bins=bins, weights=weights, **study.attrs, histo=False, figax=(fig, ax))
        histos = histos[-1]

        labels = np.array([h.label for h in histos])
        h_stat = get_stat(histos, stat)
        h_stat_err = get_stat(histos, stat_err)

        graph_array(
            labels,
            h_stat,
            yerr=h_stat_err,
            ylabel=xlabel,
            **g_kwargs,
            figax=(fig, ax)
        )

    fig.suptitle(study.title)
    fig.tight_layout()
    fig.canvas.draw()

    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)

    if study.return_figax:
        return fig, axs


def quick(*args, varlist=[], binlist=None, xlabels=None, dim=(-1, -1), size=(-1, -1), flip=False, figax=None, **kwargs):
    study = Study(*args, **kwargs)

    nvar = len(varlist)
    nrows, ncols = autodim(nvar, dim, flip)

    varlist = varlist + [None]*(nrows*ncols-nvar)
    binlist = AttrArray.init_attr(binlist, None, len(varlist))
    xlabels = AttrArray.init_attr(xlabels, None, len(varlist))
    varlist = zip(varlist, binlist, xlabels)

    xsize, ysize = autosize(size, (nrows, ncols))

    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(int(xsize*ncols), ysize*nrows),
                             dpi=80)
    fig, axs = figax

    it = tqdm(enumerate(varlist),
              total=nvar) if study.report else enumerate(varlist)
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
        return fig, axs


def overlay(tree, varlist=[], binlist=None, dim=(-1, -1), size=(-1, -1), xlabels=None, flip=None, figax=None, **kwargs):
    if type(varlist[0]) != list:
        varlist = [varlist]
    study = Study(tree, **kwargs)

    nvar = len(varlist)
    binlist = AttrArray.init_attr(binlist, None, nvar)
    xlabels = AttrArray.init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size, (nrows, ncols))
    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(int(xsize*ncols), ysize*nrows),
                             dpi=80)
    fig, axs = figax

    it = tqdm(enumerate(varlist),
              total=nvar) if study.report else enumerate(varlist)
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
        return fig, axs


def compare_masks(treelist, bkg=None, varlist=[], masks=[], label=[], h_linestyle=["-", "-.", "--", ":"], overlay=False, figax=None, saveas=None, **kwargs):
    n = len(treelist) + (1 if bkg is not None else 0)
    if figax is None:
        figax = get_figax(n*len(varlist), dim=(-1, n))
    fig, axs = figax

    nmasks = len(masks)
    h_linestyle = AttrArray.init_attr(h_linestyle, None, nmasks)
    _kwargs = defaultdict(list)
    for i, mask in enumerate(masks):
        _kwargs['label'].append(
            label[i % len(masks)] if any(label)
            else getattr(mask, '__name__', str(mask))
        )
        _kwargs['h_linestyle'].append(h_linestyle[i % len(masks)])
    kwargs.update(**_kwargs)

    def get_ax(i, axs=axs):
        if not isinstance(axs, np.ndarray):
            return axs
        if axs.ndim == 2 and n > 1:
            return axs[:, i]
        elif n == 1:
            return axs
        return axs[i]

    for i, sample in enumerate(treelist):
        if bkg is not None:
            i += 1
        if overlay:
            i = 0
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
            text=(0.0, 1.0, 'MC-Bkg'),
            text_style=dict(ha='left', va='bottom'),
            figax=(fig, get_ax(0)),
            **kwargs,
        )

    if saveas:
        save_fig(fig, saveas)


def compare_masks_by_sample(treelist, bkg=None, varlist=[], masks=[], label=[], figax=None, saveas=None, **kwargs):
    n = len(masks)
    if figax is None:
        figax = get_figax(n*len(varlist), dim=(-1, n))
    fig, axs = figax

    nmasks = len(masks)
    def get_ax(i): return axs[:, i] if axs.ndim == 2 else axs[i]

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


def compare_masks_v2(treelist, masks=[], label=[], varlist=[], h_linestyle=["-", "-.", "--", ":"], binlist=None, xlabels=None, dim=(-1, -1), size=(-1, -1), flip=False, figax=None, **kwargs):

    _treelist = []
    _kwargs = defaultdict(list)
    for i, mask in enumerate(masks):
        for tree in treelist:
            _treelist.append(tree)
            _kwargs['masks'].append(mask)
            _kwargs['h_linestyle'].append(h_linestyle[i % len(h_linestyle)])
            _kwargs['label'].append(
                label[i % len(masks)] if any(label)
                else getattr(mask, '__name__', str(mask))
            )
    kwargs.update(**_kwargs)

    study = Study(_treelist, **kwargs)

    ntrees = len(treelist)
    samples = [tree.sample for tree in treelist]
    nmasks = len(masks)
    nvar = len(varlist)
    binlist = AttrArray.init_attr(binlist, None, nvar)
    xlabels = AttrArray.init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, (nvar, 1), flip)
    ncols = min(nvar, 4)
    xsize, ysize = autosize(size, (nrows, ncols))

    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=1,
                             figsize=(int(ntrees*xsize*ncols), ysize*nrows),
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
        attrs = study.attrs

        for j, sample in enumerate(samples):
            _hists = hists[j::ntrees]
            _weights = weights[j::ntrees]
            def get_attr(value): return value[j::ntrees] if isinstance(
                value, list) else value
            _attrs = {key: get_attr(value) for key, value in attrs.items()}
            _attrs.update(
                text=(0.0, 1.0, sample),
                text_style=dict(ha='left', va='bottom'),
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
        return fig, axs


def quick2d(*args, varlist=None, binlist=None, xvarlist=[], yvarlist=[], xbinlist=[], ybinlist=[], dim=None, size=(-1, -1),  flip=False, figax=None, overlay=False, **kwargs):
    study = Study(*args, overlay=overlay, **kwargs)

    if varlist is not None:
        xvarlist = varlist[::2]
        yvarlist = varlist[1::2]
    if binlist is not None:
        xbinlist = binlist[::2]
        ybinlist = binlist[1::2]

    nvar = len(study.selections)
    nplots = len(xvarlist)

    xbinlist = AttrArray.init_attr(xbinlist, None, nplots)
    ybinlist = AttrArray.init_attr(ybinlist, None, nplots)

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
    ncols = min(nvar, 4) if not overlay else ncols
    xsize, ysize = autosize(size, (nrows, ncols))

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
            keys2d = ('contour', 'interp', 'scatter', 'overlay')
            attrs = {key: value for key, value in study.attrs.items()
                     if not key in keys2d}
            attrs['efficiency'] = True
            attrs['exe'] = None
            hist_multi(xhists, bins=xbins, xlabel=xlabel,
                       weights=weights, **attrs, figax=(fig, ax))
        else:
            attrs = dict(study.attrs)
            if overlay and 'legend' in attrs:
                del attrs['legend']
            hist2d_multi(xhists, yhists, weights=weights,
                         **info, **attrs, figax=(fig, ax))

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
        return fig, axs


def overlay2d(*args, varlist=None, binlist=None, xvarlist=[], yvarlist=[], xbinlist=[], ybinlist=[], dim=(-1, -1), size=(-1, -1),  flip=False, alpha=0.8, **kwargs):
    study = Study(*args, h_label_stat=None, alpha=alpha, **kwargs)

    if varlist is not None:
        xvarlist = varlist[::2]
        yvarlist = varlist[1::2]
    if binlist is not None:
        xbinlist = binlist[::2]
        ybinlist = binlist[1::2]

    nvar = len(study.selections)
    nplots = len(xvarlist)

    xbinlist = AttrArray.init_attr(xbinlist, None, nplots)
    ybinlist = AttrArray.init_attr(ybinlist, None, nplots)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size, (nrows, ncols))
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

        cmaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Greys', 'Purples']
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

            hist2d_simple(xhist, yhist, weights=weight, **info, **
                          study.attrs, cmap=next(cmapiter), figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)

    if study.return_figax:
        return fig, axs


def pairplot(*args, varlist=[], binlist=None, scatter=True, dim=None, overlay=True, **kwargs):
    nvar = len(varlist)
    binlist = AttrArray.init_attr(binlist, None, nvar)

    IJ = np.stack(np.meshgrid(np.arange(nvar),
                    np.arange(nvar)), axis=2)
    uptri = (IJ[:,:,0] >= IJ[:,:,1]).flatten()
    IJ = IJ.flatten()

    varlist = [varlist[i] for i in IJ]
    binlist = [binlist[i] for i in IJ]

    quick2d(
        *args,
        varlist=varlist,
        binlist=binlist,
        overlay=True,
        scatter=scatter,
        dim=(-1, nvar),
        **kwargs
    )


def quick_region(*rtrees, varlist=[], binlist=None, xlabels=None, dim=(-1, -1), size=(-1, -1), flip=False, figax=None, **kwargs):
    ftrees = rtrees[0]
    for rt in rtrees[1:]:
        ftrees = ftrees + rt
    study = Study(ftrees, stacked=False, **kwargs)

    nr = [0] + [len(rt) for rt in rtrees]
    nr = np.cumsum(nr)

    nvar = len(varlist)
    binlist = AttrArray.init_attr(binlist, None, nvar)
    xlabels = AttrArray.init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    nrows, ncols = autodim(nvar, dim, flip)
    xsize, ysize = autosize(size, (nrows, ncols))

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

        hists = [ak.concatenate(hists[lo:hi]) for lo, hi in zip(
            nr[:-1], nr[1:]) if hi <= len(hists)]
        weights = [ak.concatenate(weights[lo:hi]) for lo, hi in zip(
            nr[:-1], nr[1:]) if hi <= len(weights)]

        hist_multi(hists, bins=bins, xlabel=xlabel,
                   weights=weights, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()

    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)

    if study.return_figax:
        return fig, axs


def quick2d_region(*rtrees, varlist=None, binlist=None, xvarlist=[], yvarlist=[], xbinlist=[], ybinlist=[],  dim=(-1, -1), size=(-1, -1),  flip=False, figax=None, **kwargs):
    ftrees = rtrees[0]
    for rt in rtrees[1:]:
        ftrees = ftrees + rt
    study = Study(ftrees, **kwargs)

    nr = [0] + [len(rt) for rt in rtrees]
    nr = np.cumsum(nr)

    if varlist is not None:
        xvarlist = varlist[::2]
        yvarlist = varlist[1::2]
    if binlist is not None:
        xbinlist = binlist[::2]
        ybinlist = binlist[1::2]

    nvar = len(rtrees)
    nplots = len(xvarlist)

    xbinlist = AttrArray.init_attr(xbinlist, None, nplots)
    ybinlist = AttrArray.init_attr(ybinlist, None, nplots)

    nrows, ncols = autodim(nvar*nplots, dim, flip)
    xsize, ysize = autosize(size, (nrows, ncols))
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

        xhists = [ak.concatenate(xhists[lo:hi]) for lo, hi in zip(
            nr[:-1], nr[1:]) if hi <= len(xhists)]
        yhists = [ak.concatenate(yhists[lo:hi]) for lo, hi in zip(
            nr[:-1], nr[1:]) if hi <= len(yhists)]
        weights = [ak.concatenate(weights[lo:hi]) for lo, hi in zip(
            nr[:-1], nr[1:]) if hi <= len(weights)]

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

            # hist2d_simple(xhist, yhist, weights=weight, **
            #               info, **study.attrs, figax=(fig, ax))
            
            hist2d_multi([xhist], [yhist], weights=[weight], **
                          info, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    # plt.show()
    if study.saveas:
        save_fig(fig, study.saveas)

    if study.return_figax:
        return fig, axs


def table(*args, varlist=[], binlist=None, xlabels=None, dim=(-1, -1), size=(-1, -1), flip=False, figax=None, tablef=None, **kwargs):
    study = Study(*args, table=True, tablef=tablef, **kwargs)

    nvar = len(varlist)
    binlist = AttrArray.init_attr(binlist, None, nvar)
    xlabels = AttrArray.init_attr(xlabels, None, nvar)
    varlist = zip(varlist, binlist, xlabels)

    xsize, ysize = autosize(size, (1, 1))

    for i, (var, bins, xlabel) in tqdm(enumerate(varlist), total=nvar):
        fig, ax = plt.subplots(figsize=(int(xsize), int(ysize)))

        bins, xlabel = format_var(var, bins, xlabel)
        hists = study.get_array(var)
        weights = study.get_scale(hists)
        hist_multi(hists, bins=bins, xlabel=xlabel,
                   weights=weights, **study.attrs, figax=(fig, ax))
        fig.canvas.draw()
        study.table(var, xlabel, figax=(fig, ax), **study.attrs)
    plt.close()

def plot_timing(f_method, size=(25,10), saveas=None):
    from collections import defaultdict
    fig, ax = get_figax(size=size)

    # create a set of workers used 
    def draw_process(timing, color='r'):
        X = (timing.end + timing.start)/2
        widths = (timing.end - timing.start)/2
        workers = timing.worker
        threads = timing.thread

        worker_map = defaultdict(list)
        worker_map['MainProcess']
        for worker, thread in zip(workers, threads):
            worker_map[worker].append(thread)

        for x, width, worker, thread in zip(X, widths, workers, threads):
            worker_offset = list(worker_map.keys()).index(worker)
            worker_threads = list(set((worker_map[worker])))
            thread_offset = worker_threads.index(thread)/len(worker_threads)

            height = 0.5/len(worker_threads) if len(worker_threads) > 1 else 0.8
            offset = worker_offset + thread_offset + 0.1*height

            rect = plt.Rectangle((x-width, offset), width*2, height, color=color, alpha=0.2)
            ax.add_patch(rect)

        return len(worker_map)

    nworkers = 0
    nworkers = max(nworkers, draw_process(f_method.start_timing, color='r'))
    nworkers = max(nworkers, draw_process(f_method.run_timing, color='g'))
    nworkers = max(nworkers, draw_process(f_method.end_timing, color='b'))

    end = ak.max(f_method.end_timing.end)
    ax.set(xlabel='Time (s)', ylabel='Worker')
    ax.set_xlim(0, end)
    ax.set_ylim(-0.1, nworkers+0.1)

    ax.set_yticks(np.arange(nworkers))
    ax.grid()

    if saveas:
        save_fig(fig, f'debug/{saveas}')
        return

    plt.show()
