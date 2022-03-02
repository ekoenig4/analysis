#!/usr/bin/env python
# coding: utf-8

from ..utils import *
from ..xsecUtils import lumiMap
from ..classUtils.Sample import Samplelist
from ..classUtils.Stack import Stack

import matplotlib.colors as clrs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.style.use('science')
plt.rcParams["figure.figsize"] = (16/3,5)
plt.rcParams['font.size'] =  15


def format_axis(ax, title=None, xlabel=None, xlim=None, ylabel=None, ylim=None, grid=False, **kwargs):
    ax.set_ylabel(ylabel)

    if grid:
        ax.grid()
    if type(xlabel) == str:
        ax.set_xlabel(xlabel)
    elif xlabel is not None:
        ax.set_xticks(range(len(xlabel)))

        rotation = 0
        if type(xlabel[0]) == str:
            rotation = -45
        ax.set_xticklabels(xlabel, rotation=rotation)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)


def graph_simple(xdata, ydata, xlabel=None, ylabel=None, title=None, label=None, marker='o', ylim=None, xticklabels=None, figax=None):
    if figax is None:
        figax = plt.subplots()
    (fig, ax) = figax

    ax.plot(xdata, ydata, label=label, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if xticklabels is not None:
        ax.set_xticks(xdata)
        ax.set_xticklabels(xticklabels)

    if ylim:
        ax.set_ylim(ylim)
    if label:
        ax.legend()
    return (fig, ax)


def graph_multi(xdata, ydatalist, yerrs=None, title=None, labels=None, markers=None, colors=None, log=False, figax=None, **kwargs):
    if figax is None:
        figax = plt.subplots()
    (fig, ax) = figax

    ndata = len(ydatalist)
    labels = init_attr(labels, None, ndata)
    markers = init_attr(markers, "o", ndata)
    colors = init_attr(colors, None, ndata)
    yerrs = init_attr(yerrs, None, ndata)

    for i, (ydata, yerr, label, marker, color) in enumerate(zip(ydatalist, yerrs, labels, markers, colors)):
        ax.errorbar(xdata, ydata, yerr=yerr, label=label,
                    marker=marker, color=color, capsize=1)

    if log:
        ax.set_yscale('log')
    if any(label for label in labels):
        ax.legend()
    format_axis(ax, **kwargs)

    return (fig, ax)


def graph_avgstd(ydata, ylabel=None, xlabels=None, set={}, figax=None):
    ndata = len(ydata)
    ydata = np.array([get_avg_std(array) for array in ydata])
    fig, ax = graph_multi(range(ndata), ydatalist=[ydata[:, 0]], yerrs=[
                          ydata[:, 1]], ylabel=ylabel, figax=figax)
    ax.set_xticks(list(range(ndata)))
    if xlabels:
        ax.set_xticklabels(xlabels, rotation=45)
    ax.set(**set)
    return fig, ax


def plot_simple(data, bins=None, xlabel=None, title=None, label=None, figax=None):
    if figax is None:
        figax = plt.subplots()
    (fig, ax) = figax

    ax.hist(data, bins=bins, label=label)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if label:
        ax.legend()
    return (fig, ax)


def plot_branch(variable, tree, mask=None, selected=None, bins=None, xlabel=None, title=None, label=None, figax=None):
    if figax is None:
        figax = plt.subplots()
    if mask is None:
        mask = np.ones(ak.size(tree['Run']), dtype=bool)
    (fig, ax) = figax

    data = tree[variable][mask]
    if selected is not None:
        data = tree[variable][mask][selected]
    data = ak.flatten(data, axis=-1)

    ax.hist(data, bins=bins, label=label)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    return (fig, ax)


def ratio_plot(num, dens, denerrs, bins, xlabel, figax, ylim=(0.1, 1.9), ylabel='Ratio',size='20%', grid=True, inv=False,**kwargs):

    fig, ax = figax
    ax.get_xaxis().set_visible(0)
    divider = make_axes_locatable(ax)
    ax_ratio = divider.append_axes("bottom", size=size, pad=0.1, sharex=ax)

    def _calc_ratio(data_hist, hist, error):
        ratio = safe_divide(data_hist, hist,0)
        ratio_error = ratio * safe_divide(error, hist,0)
        return ratio, ratio_error
    
    def _calc_ratio_inv(data_hist,hist,error): return _calc_ratio(hist,data_hist,error);
    
    calc_ratio = _calc_ratio if not inv else _calc_ratio_inv

    xdata = get_bin_centers(bins)
    ratio_info = np.array([calc_ratio(num, den, denerr)
                          for den, denerr in zip(dens, denerrs)])
    ratio_data, ratio_error = ratio_info[:, 0], ratio_info[:, 1]    
    padding = np.zeros((1,len(bins)-1))
    ratio_data = np.concatenate([padding-99,ratio_data])
    ratio_error = np.concatenate([padding,ratio_error])
    
    graph_multi(xdata, ratio_data, yerrs=ratio_error, figax=(
        fig, ax_ratio), xlabel=xlabel, ylabel=ylabel, ylim=ylim, grid=grid, **kwargs)
    
def build_ratio(num,denlist,bins=None,xlabel=None,figax=None,**kwargs):
    if num is None:
        num = denlist.pop(0).histo

    denerrs = [sample.error for sample in denlist]
    colors = [sample.color for sample in denlist]
    denlist = [sample.histo for sample in denlist]
    ratio_plot(num, denlist, denerrs, bins, xlabel,
                figax, colors=colors, **kwargs)

def hist_error(ax, data, error=None, **attrs):
    histo, bins, container = ax.hist(data, **attrs)

    if error is None:
        return histo, error

    color = container[0].get_ec()
    histtype = attrs.get('histtype', None)
    if histtype != 'step':
        color = 'black'

    bin_centers, bin_widths = get_bin_centers(bins), get_bin_widths(bins)
    ax.errorbar(bin_centers, histo, yerr=error,
                fmt='none', color=color, capsize=1)

    return histo, error


def stack_error(ax, stack, bins=None, log=False):
    bin_centers = get_bin_centers(bins)
    bin_widths = 2*get_bin_widths(bins)
    bar = ax.bar(bin_centers, stack[0].histo,bin_widths, label=stack[0].label,log=log,**stack[0].attrs)
    histo = stack[0].histo
    errors = [stack[0].error]
    for i,sample in enumerate(stack[1:]):
        bar = ax.bar(bin_centers, sample.histo,bin_widths,label=sample.label, bottom=stack[i].histo,log=log,**sample.attrs)
        histo += sample.histo
        errors.append(sample.error)
        
    if errors[0] is None:
        return histo, None

    error = np.sqrt(np.sum(np.array(errors)**2, axis=0))
    ax.errorbar(bin_centers, histo, yerr=error,
                fmt='none', color='grey', capsize=1)
    return histo, error

def build_stack(samples,bins=None,log=False,ax=None):
    stack = Stack()
    stack.add(*samples)
    stack.sort(key=lambda sample: sample.scaled_nevents, reverse=not log)
    histo, error = stack_error(ax, stack, bins=bins, log=log)
    stack.histo = histo
    stack.error = error
    stack.color = 'black'
    return stack

def draw_stats(ax,sample,bins=None):
    x,w = sample.data,sample.weight 
    if bins is not None: x,w = restrict_array(x,bins,w)
    mean = np.average(x,weights=w)
    stdv = np.sqrt(np.average((x-mean)**2,weights=w))
    stats = dict(
        mean=mean,stdv=stdv
    )
    sp = max(map(len,stats.keys()))
    stat_lines = [f"{stat:<{sp}}: {value:0.2f}" for stat,value in stats.items()]
    ax.text(0.9,0.9,'\n'.join(stat_lines),  transform=ax.transAxes)

def hist_multi(datalist, bins=None, weights=None, labels=None, is_datas=None, is_signals=None, density=False, cumulative=False, sumw2=True, scale=True,
               title=None, xlabel=None, ylabel=None, figax=None, log=0, ratio=False, stacked=False, lumikey=None, stats=False, stats_restricted=False, **kwargs):
    if figax is None:
        figax = plt.subplots()
    (fig, ax) = figax

    if scale is False:
        lumikey = None

    lumi, lumi_tag = lumiMap[lumikey]
    attrs = {key[2:]: value for key,
             value in kwargs.items() if key.startswith("s_")}
    samples = Samplelist(datalist, bins, weights=weights, density=density, cumulative=cumulative,lumi=lumi, labels=labels,
                         is_datas=is_datas, is_signals=is_signals, sumw2=sumw2, scale=scale, **attrs)

    bins = samples.bins
    bin_centers, bin_widths = get_bin_centers(bins), get_bin_widths(bins)

    if ratio:
        ratio = samples.nsample > 1
    if stacked:
        stacked = samples.nmc > 1
    if density:
        stacked = False
    denlist = []

    def get_extrema(h): return (np.max(h), np.min(h[np.nonzero(h)]))
    ymin, ymax = np.inf, 0
    

    if stacked:
        stack = build_stack([sample for sample in samples if sample.is_bkg],bins,log,ax)
        denlist.append(stack)

        hmax, hmin = get_extrema(stack.histo)
        ymax, ymin = max(ymax, hmax), min(ymin, hmin)


    num = None
    for sample in samples:
        histo = sample.histo
        hmax, hmin = get_extrema(histo)
        ymax, ymin = max(ymax, hmax), min(ymin, hmin)

        if sample.is_data:
            histo, error, label = sample.histo, sample.error, sample.label
            ax.errorbar(bin_centers, histo, yerr=None, xerr=bin_widths,
                        color='black', marker='o', linestyle='None', label=label)

            num = histo
        elif sample.is_signal or not stacked:
            data, histo, error, weight, label, attrs = sample.data, sample.histo, sample.error, sample.weight, sample.label, sample.attrs
            attrs["histtype"] = "step" if len(samples) > 1 else "bar"
            attrs["linewidth"] = 2
            histo,error = hist_error(ax, data, bins=bins, error=error,
                       weights=weight, label=label, log=log, cumulative=cumulative, **attrs)
            sample.histo = histo 
            sample.error = error

            if not samples.has_data or not sample.is_signal:
                denlist.append(sample)
                

    if ylabel is None:
        ylabel = "Events"
        if scale == "xs":
            ylabel = "Cross-Section (pb)"
        if density:
            ylabel = "Fraction of Events"
        if cumulative or cumulative == 1:
            ylabel = "Fraction of Events Below (CDF)"
        if cumulative == -1:
            ylabel = "Fraction of Events Above (CDF)"
    if lumi != 1:
        title = f"{lumi/1000:0.1f} $fb^{'{-1}'}$ {lumi_tag}"

    if kwargs.get('ylim', None) is None:
        ymin_scale,ymax_scale = kwargs.get('yscale',(0.1,10) if log else (0,1.5))
        kwargs['ylim'] = (ymin_scale*ymin, ymax_scale*ymax)
        
    if stats_restricted: draw_stats(ax,sample,sample.bins)
    elif stats: draw_stats(ax,sample)

    format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)
    ax.legend()
    if ratio: 
        build_ratio(num,denlist,bins,xlabel,figax=figax,
    **{key[2:]: value for key,value in kwargs.items() if key.startswith("r_")})

    return (fig, ax)

def boxplot_multi(datalist, bins=None, weights=None, labels=None, is_datas=None, is_signals=None, density=False, cumulative=False, sumw2=True, scale=True,
               title=None, xlabel=None, ylabel=None, figax=None, log=0, ratio=False, stacked=False, lumikey=None, stats=False, stats_restricted=False, **kwargs):
    if figax is None:
        figax = plt.subplots()
    (fig, ax) = figax

    if scale is False:
        lumikey = None

    lumi, lumi_tag = lumiMap[lumikey]
    attrs = {key[2:]: value for key,
             value in kwargs.items() if key.startswith("s_")}
    samples = Samplelist(datalist, bins, weights=weights, density=density, cumulative=cumulative,lumi=lumi, labels=labels,
                         is_datas=is_datas, is_signals=is_signals, sumw2=sumw2, scale=scale, **attrs)

    data = [ restrict_array(sample.data,sample.bins) for sample in samples ]    
    # data = [ sample.data for sample in samples ]
    
    ax.boxplot(data,showfliers=False,vert=False)

    if ylabel is None:
        ylabel = "Events"
        if scale == "xs":
            ylabel = "Cross-Section (pb)"
        if density:
            ylabel = "Fraction of Events"
    if lumi != 1:
        title = f"{lumi/1000:0.1f} $fb^{'{-1}'}$ {lumi_tag}"
    format_axis(ax, xlabel=xlabel, ylabel=None, title=title, xlim=(samples.bins[0],samples.bins[-1]), **kwargs)
    ax.set_yticklabels(labels)

    return (fig, ax)


def plot_mask_stack_comparison(datalist, bins=None, title=None, xlabel=None, figax=None, density=0, labels=None, histtype="bar", colors=None):
    if figax is None:
        figax = plt.subplots()
    (fig, ax) = figax

    if labels is None:
        labels = ["" for _ in datalist]

    labels = [f"{label} ({ak.size(data):.2e})"for data,
              label in zip(datalist, labels)]
    info = {"bins": bins, "label": labels, "density": density}
    if histtype:
        info["histtype"] = histtype
    if colors:
        info["color"] = colors
    ax.hist(datalist, stacked=True, **info)

    ax.set_ylabel("Fraction of Events" if density else "Events")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
#     if density: ax.set_ylim([0,1])
    ax.legend()
    return (fig, ax)


def hist2d_simple(xdata, ydata, xbins=None, ybins=None, title=None, xlabel=None, ylabel=None, figax=None, weights=None, 
                  lumikey=None, density=0, log=1, grid=False, label=None, cmap="YlOrRd", show_counts=False, **kwargs):
    if figax is None:
        figax = plt.subplots()
    (fig, ax) = figax

    xdata = flatten(xdata)
    ydata = flatten(ydata)

    nevnts = ak.size(xdata)
        
    lumi, lumi_tag = lumiMap[lumikey]
    if weights is not None:
        weights = lumi*flatten(weights)
    else:
        weights = np.ones((nevnts,))
        
    nevnts = ak.sum(weights)
    
    if density: weights = weights / nevnts

    if xbins is None:
        xbins = autobin(xdata)
    if ybins is None:
        ybins = autobin(ydata)


    n, bx, by, im = ax.hist2d(np.array(xdata), np.array(ydata), (xbins, ybins), weights=weights,
                               norm=clrs.LogNorm() if log else clrs.Normalize(), cmap=cmap)
    
    if show_counts:
        for i,(bx_lo,bx_hi) in enumerate(zip(bx[:-1],bx[1:])):
            for j,(by_lo,by_hi) in enumerate(zip(by[:-1],by[1:])):
                ax.text((bx_hi+bx_lo)/2,(by_hi+by_lo)/2,f'{n[i,j]:0.2}',ha="center", va="center", fontweight="bold")
        

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if label:
        ax.text(0.05, 1.01, f"{label} ({nevnts:0.2e})", transform=ax.transAxes)

    if grid:
        ax.set_yticks(ybins)
        ax.set_xticks(xbins)
        ax.grid()
    fig.colorbar(im, ax=ax)
    return (fig, ax)


def plot_barrel_display(eta, phi, weight, nbins=20, figax=None, cblabel=None, cmin=0.01):
    if figax is None:
        figax = plt.subplots()
    (fig, ax) = figax

    eta = ak.to_numpy(ak.flatten(eta, axis=None))
    phi = ak.to_numpy(ak.flatten(phi, axis=None))
    weight = ak.to_numpy(ak.flatten(weight, axis=None))

    max_eta = max(ak.max(np.abs(eta)), 2.5)

    xbins = np.linspace(-max_eta, max_eta, nbins)
    ybins = np.linspace(-3.14159, 3.14159, nbins)

    n, bx, by, im = ax.hist2d(eta, phi, bins=(
        xbins, ybins), weights=weight, cmin=cmin)
    ax.set_xlabel("Jet Eta")
    ax.set_ylabel("Jet Phi")
    ax.grid()

    cb = fig.colorbar(im, ax=ax)
    if cblabel:
        cb.ax.set_ylabel(cblabel)
    return (fig, ax)


def plot_endcap_display(eta, phi, weight, nbins=20, figax=None):
    if figax is None:
        figax = plt.subplots(projection='polar')
    (fig, ax) = figax

    eta = ak.to_numpy(ak.flatten(eta, axis=None))
    phi = ak.to_numpy(ak.flatten(phi, axis=None))
    weight = ak.to_numpy(ak.flatten(weight, axis=None)) / \
        ak.max(weight, axis=None)

    for p, w in zip(phi, weight):
        ax.plot([p, p], [0, 1], linewidth=max(5*w, 1))

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["", ""])
    return (fig, ax)
