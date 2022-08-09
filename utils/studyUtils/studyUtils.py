#!/usr/bin/env python
# coding: utf-8

from .studyUtils import *
from ..plotUtils.plotUtils import graph_multi, boxplot_multi,  plot_barrel_display, plot_endcap_display
from ..plotUtils.multi_plotter import hist_multi,hist2d_simple
from ..selectUtils import *
from ..classUtils.Study import Study, save_fig, format_var
from ..varConfig import varinfo
from ..utils import loop_iter

import vector
import matplotlib.pyplot as plt
from tqdm import tqdm


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
        nrows = nvar//2
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
    study = Study(*args, sumw2=False, log=log,
                  h_label_stat=h_label_stat,scale=scale,lumi=lumi, **kwargs)
    def get_scaled_cutflow(tree): return ak.Array(
        [cutflow*(fn.scale if scale else 1) for cutflow, fn in zip(tree.cutflow, tree.filelist)])
    is_mc = [ not tree.is_data for tree in study.selections ]
    
    scaled_cutflows = [get_scaled_cutflow(tree) for tree in study.selections]
    cutflow_labels = max(
        (selection.cutflow_labels for selection in study.selections), key=lambda a: len(a))
    ncutflow = len(cutflow_labels)+1
    flatten_cutflows = [ ak.sum(ak.fill_none(ak.pad_none(cutflow,len(cutflow_labels),axis=-1),0),axis=0) for cutflow in scaled_cutflows ]
    
    bins = np.arange(ncutflow)-0.5
    cutflow_bins = [ak.local_index(cutflow, axis=-1)
                    for cutflow in flatten_cutflows]
    
    
    figax = plt.subplots(figsize=size) if size else None
    
    if scale is False:
        if density: 
            flatten_cutflows = [ cutflow/cutflow[0] for cutflow in flatten_cutflows ]
        fig,ax = graph_multi(cutflow_labels,flatten_cutflows,xlabel=cutflow_labels,**study.attrs,figax=figax)
    else:
        fig, ax = hist_multi(cutflow_bins, bins=bins, weights=flatten_cutflows, xlabel=cutflow_labels, h_histtype=[
                        "step"]*len(study.selections), **study.attrs, figax=figax)
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, "cutflow", study.saveas)
    
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

        bins, xlabel = study.format_var(var, bins, xlabel)
        hists = study.get(var)
        weights = study.get_scale(hists)

        if ncols == 1 and nrows == 1:
            ax = axs
        elif bool(ncols > 1) != bool(nrows > 1):
            ax = axs[i]
        else:
            ax = axs[i//ncols, i % ncols]

        boxplot_multi(hists, bins=bins, xlabel=xlabel,
                      weights=weights, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, "", study.saveas)
    
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

    for i, (var, bins, xlabel) in tqdm(enumerate(varlist),total=nvar):
        if ncols == 1 and nrows == 1:
            ax = axs
        elif bool(ncols > 1) != bool(nrows > 1):
            ax = axs[i]
        else:
            ax = axs[i//ncols, i % ncols]
            
        if var is None: 
            ax.set_visible(False)
            continue
        
        bins, xlabel = study.format_var(var, bins, xlabel)
        hists = study.get(var)
        weights = study.get_scale(hists)
        hist_multi(hists, bins=bins, xlabel=xlabel,
                   weights=weights, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
        
    # plt.show()
    if study.saveas:
        save_fig(fig, "", study.saveas)
        
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

    for i, (group, bins, xlabel) in enumerate(varlist):
        hists = [study.get(var)[0] for var in group]
        weights = [study.get_scale(hists)[0] for var in group]
        # if labels is None:
        #     study.attrs['labels'] = group

        if ncols == 1 and nrows == 1:
            ax = axs
        elif bool(ncols > 1) != bool(nrows > 1):
            ax = axs[i]
        else:
            ax = axs[i//ncols, i % ncols]
        hist_multi(hists, bins=bins, weights=weights,
                   xlabel=xlabel, **study.attrs, figax=(fig, ax))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas:
        save_fig(fig, "", study.saveas)
    
    if study.return_figax:
        return fig,axs


def quick2d(*args, varlist=None, binlist=None, xvarlist=[], yvarlist=[], xbinlist=[], ybinlist=[], dim=(-1,-1), size=(-1,-1),  flip=False, figax=None, **kwargs):
    study = Study(*args, **kwargs)

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

    nrows, ncols = autodim(nvar*nplots, dim, flip)
    xsize, ysize = autosize(size,(nrows,ncols))

    if figax is None:
        figax = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(int(xsize*ncols), ysize*nrows),
                                dpi=80)
    fig, axs = figax
    labels = study.attrs.pop("h_label")

    it1 = enumerate(zip(xvarlist, yvarlist, xbinlist, ybinlist))
    for i, (xvar, yvar, xbins, ybins) in tqdm(it1, total=nplots, position=0):
        xbins, xlabel = study.format_var(xvar, bins=xbins, xlabel=xvar)
        ybins, ylabel = study.format_var(yvar, bins=ybins, xlabel=yvar)
        info = dict(x_bins=xbins, xlabel=xlabel, y_bins=ybins, ylabel=ylabel)

        xhists = study.get(xvar)
        yhists = study.get(yvar)

        weights = study.get_scale(xhists)

        it2 = enumerate(zip(xhists, yhists, weights, labels))
        for j, (xhist, yhist, weight, label) in tqdm(it2, total=nvar, position=1, leave=False):

            if nvar == ncols:
                k = i*nvar + j
            else:
                k = j*nplots + i

            study.attrs["h_label"] = label

            if ncols == 1 and nrows == 1:
                ax = axs
            elif bool(ncols > 1) != bool(nrows > 1):
                ax = axs[k]
            else:
                ax = axs[k//ncols, k % ncols]

            hist2d_simple(xhist, yhist, weights=weight, **
                        info, **study.attrs, figax=(fig, ax))
        fig.suptitle(study.title)
        fig.tight_layout()
    # plt.show()
    if study.saveas:
        save_fig(fig, "", study.saveas)
    
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
        xbins, xlabel = study.format_var(xvar, bins=xbins, xlabel=xvar)
        ybins, ylabel = study.format_var(yvar, bins=ybins, xlabel=yvar)
        info = dict(x_bins=xbins, xlabel=xlabel, y_bins=ybins, ylabel=ylabel)

        xhists = study.get(xvar)
        yhists = study.get(yvar)

        weights = study.get_scale(xhists)

        
        cmaps = ['Reds','Blues','Greens','Oranges','Greys','Purples']
        cmapiter = loop_iter(cmaps)
    
        for j, (xhist, yhist, weight, label) in enumerate(zip(xhists, yhists, weights, labels)):
            if nvar == ncols:
                k = i*nvar + j
            else:
                k = j*nplots + i

            study.attrs["h_label"] = label

            if ncols == 1 and nrows == 1:
                ax = axs
            elif bool(ncols > 1) != bool(nrows > 1):
                ax = axs[k]
            else:
                ax = axs[k//ncols, k % ncols]

            hist2d_simple(xhist, yhist, weights=weight, **info, **study.attrs, cmap=next(cmapiter), figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    # plt.show()
    if study.saveas:
        save_fig(fig, "", study.saveas)
    
    if study.return_figax:
        return fig,axs

def quick_region(*rtrees, varlist=[], binlist=None, xlabels=None, dim=(-1,-1), size=(-1,-1), flip=False, figax=None, **kwargs):
    ftrees = rtrees[0]
    for rt in rtrees[1:]:
        ftrees = ftrees + rt
    study = Study(ftrees, **kwargs)

    nr = [0] + [ len(rt) for rt in rtrees ]

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
        if ncols == 1 and nrows == 1:
            ax = axs
        elif bool(ncols > 1) != bool(nrows > 1):
            ax = axs[i]
        else:
            ax = axs[i//ncols, i % ncols]
            
        if var is None: 
            ax.set_visible(False)
            continue
        
        bins, xlabel = study.format_var(var, bins, xlabel)
        hists = study.get(var)
        weights = study.get_scale(hists)

        hists = [ ak.concatenate(hists[lo:hi]) for lo,hi in zip(nr[:-1],nr[1:]) ]
        weights = [ ak.concatenate(weights[lo:hi]) for lo,hi in zip(nr[:-1],nr[1:]) ]

        hist_multi(hists, bins=bins, xlabel=xlabel,
                   weights=weights, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
        
    # plt.show()
    if study.saveas:
        save_fig(fig, "", study.saveas)
        
    if study.return_figax:
        return fig,axs


def quick2d_region(r1trees, *args, varlist=None, binlist=None, xvarlist=[], yvarlist=[], xbinlist=[], ybinlist=[],  dim=(-1,-1), size=(-1,-1),  flip=False, figax=None, **kwargs):
    study = Study(r1trees, *args, **kwargs)

    if varlist is not None:
        xvarlist=varlist[::2]
        yvarlist=varlist[1::2]
    if binlist is not None:
        xbinlist=binlist[::2]
        ybinlist=binlist[1::2]

    nvar = 1
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
        xbins, xlabel = study.format_var(xvar, bins=xbins, xlabel=xvar)
        ybins, ylabel = study.format_var(yvar, bins=ybins, xlabel=yvar)
        info = dict(x_bins=xbins, xlabel=xlabel, y_bins=ybins, ylabel=ylabel)

        xhists = study.get(xvar)
        yhists = study.get(yvar)

        weights = study.get_scale(xhists)
        weights = [ak.concatenate(weights)]
        xhists = [ak.concatenate(xhists)]
        yhists = [ak.concatenate(yhists)]
        
        for j, (xhist, yhist, weight, label) in enumerate(zip(xhists, yhists, weights, labels)):
            if nvar == ncols:
                k = i*nvar + j
            else:
                k = j*nplots + i

            study.attrs["h_label"] = label

            if ncols == 1 and nrows == 1:
                ax = axs
            elif bool(ncols > 1) != bool(nrows > 1):
                ax = axs[k]
            else:
                ax = axs[k//ncols, k % ncols]
            
            hist2d_simple(xhist, yhist, weights=weight, **
                        info, **study.attrs, figax=(fig, ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    # plt.show()
    if study.saveas:
        save_fig(fig, "", study.saveas)
    
    if study.return_figax:
        return fig,axs


def njets(*args, **kwargs):
    study = Study(*args, **kwargs)

    nrows, ncols = 1, 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5))

    weights = study.get("scale")

    varlist = ["n_jet", "nloose_btag", "nmedium_btag", "ntight_btag"]

    for i, var in enumerate(varlist):
        tree_vars = study.get(var)
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
        hists = study.get(varname)
        weights = study.get_scale(varname)
        info = varinfo[varname]
        hist_multi(hists, weights=weights, **info,
                   figax=(fig, axs[i//ncols, i % ncols]), **study.attrs)

    n_jet_list = study.get("n_jet")
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
    weights = study.get("scale")

    for ijet in range(njets):
        nrows, ncols = 1, 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5))

        for i, varname in enumerate(varlist):
            hists = [var[:, ijet] for var in study.get(varname)]
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
        hists = study.get(varname)
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
            hists = [var[:, ihigg] for var in study.get(varname)]
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
