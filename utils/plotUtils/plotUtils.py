#!/usr/bin/env python
# coding: utf-8

from . import *

import matplotlib.colors as clrs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

lumiMap = {
    None:[1,None],
    2016:[35900,"(13 TeV,2016)"],
    2017:[41500,"(13 TeV,2017)"],
    2018:[59740,"(13 TeV,2018)"],
    20180:[14300,"(13 TeV,2018 A)"],
    20181:[7070,"(13 TeV,2018 B)"],
    20182:[6900,"(13 TeV,2018 C)"],
    20183:[13540,"(13 TeV,2018 D)"],
    "Run2":[101000,"13 TeV,Run 2)"],
}

def format_axis(ax,title=None,xlabel=None,ylabel=None,ylim=None,grid=False,**kwargs):
    ax.set_ylabel(ylabel)

    if grid: ax.grid()
    if type(xlabel) == str:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticks(range(len(xlabel)))

        rotation = 0
        if type(xlabel[0]) == str: rotation = -45
        ax.set_xticklabels(xlabel,rotation=rotation)
    ax.set_title(title)
    if ylim is not None: ax.set_ylim(ylim)

def graph_simple(xdata,ydata,xlabel=None,ylabel=None,title=None,label=None,marker='o',ylim=None,xticklabels=None,figax=None):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax
    
    ax.plot(xdata,ydata,label=label,marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if xticklabels is not None:
        ax.set_xticks(xdata)
        ax.set_xticklabels(xticklabels)
    
    if ylim: ax.set_ylim(ylim)
    if label: ax.legend()
    return (fig,ax)

def graph_multi(xdata,ydatalist,yerrs=None,title=None,labels=None,markers=None,colors=None,log=False,figax=None,**kwargs):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    ndata = len(ydatalist)
    labels = init_attr(labels,None,ndata)
    markers = init_attr(markers,"o",ndata)
    colors = init_attr(colors,None,ndata)
    yerrs = init_attr(yerrs,None,ndata)
    
    
    for i,(ydata,yerr,label,marker,color) in enumerate(zip(ydatalist,yerrs,labels,markers,colors)):
        ax.errorbar(xdata,ydata,yerr=yerr,label=label,marker=marker,color=color,capsize=1)

    if log: ax.set_yscale('log')
    if any(label for label in labels): ax.legend()
    format_axis(ax,**kwargs)
    
    return (fig,ax)

def plot_simple(data,bins=None,xlabel=None,title=None,label=None,figax=None):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax
    
    ax.hist(data,bins=bins,label=label)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if label: ax.legend()
    return (fig,ax)
    
def plot_branch(variable,tree,mask=None,selected=None,bins=None,xlabel=None,title=None,label=None,figax=None):
    if figax is None: figax = plt.subplots()
    if mask is None: mask = np.ones(ak.size(tree['Run']),dtype=bool)
    (fig,ax) = figax
    
    data = tree[variable][mask]
    if selected is not None: data = tree[variable][mask][selected]
    data = ak.flatten( data,axis=-1 )
    
    ax.hist(data,bins=bins,label=label)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    return (fig,ax)

def ratio_plot(num,dens,denerrs,bins,xlabel,figax,ylim=(0.1,1.9),grid=True,**kwargs):
    
    fig,ax = figax
    ax.get_xaxis().set_visible(0)
    divider = make_axes_locatable(ax)
    ax_ratio = divider.append_axes("bottom", size="20%", pad=0.1, sharex=ax)

    def calc_ratio(data_hist,hist,error):
        ratio = safe_divide(data_hist,hist)
        ratio_error = ratio * safe_divide(error,hist)
        return ratio,ratio_error
    
    xdata = get_bin_centers(bins)
    ratio_info = np.array([ calc_ratio(num,den,denerr) for den,denerr in zip(dens,denerrs) ])
    ratio_data,ratio_error = ratio_info[:,0],ratio_info[:,1]
    graph_multi(xdata,ratio_data,yerrs=ratio_error,figax=(fig,ax_ratio),xlabel=xlabel,ylabel="Ratio",ylim=ylim,grid=True,**kwargs)

def hist_error(ax,data,error=None,**attrs):
    histo,bins,container = ax.hist(data,**attrs)

    if error is None: return histo,error
    
    color = container[0].get_ec()
    histtype = attrs.get('histtype',None)
    if histtype != 'step': color = 'black'
    
    bin_centers,bin_widths  = get_bin_centers(bins),get_bin_widths(bins)
    ax.errorbar(bin_centers,histo,yerr=error,fmt='none',color=color,capsize=1)
    
    return histo,error

def stack_error(ax,datalist,errors=None,**attrs):
    histos,bins,container = ax.hist(datalist,stacked=True,**attrs)
    histo = histos[-1]

    if errors is None: return histo,None
    
    bin_centers,bin_widths  = get_bin_centers(bins),get_bin_widths(bins)
    error = np.sqrt(np.sum(errors**2,axis=0))
    ax.errorbar(bin_centers,histo,yerr=error,fmt='none',color='grey',capsize=1)
    return histo,error

def hist_multi(datalist,bins=None,weights=None,labels=None,is_datas=None,is_signals=None,density=0,sumw2=True,scale=True,
               title=None,xlabel=None,ylabel=None,figax=None,log=0,ratio=False,stacked=False,lumikey=None,**kwargs):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    if scale is False: lumikey=None

    lumi,lumi_tag = lumiMap[lumikey]
    attrs = { key[2:]:value for key,value in kwargs.items() if key.startswith("s_") }
    samples = Samplelist( datalist,bins,weights=weights,density=density,lumi=lumi,labels=labels,is_datas=is_datas,is_signals=is_signals,sumw2=sumw2,scale=scale,**attrs )

    bins = samples.bins
    bin_centers,bin_widths = get_bin_centers(bins),get_bin_widths(bins)

    if ratio: ratio = samples.nsample > 1
    if stacked: stacked = samples.nmc > 1
    if density: stacked = False
    denlist = []

    get_extrema = lambda h : (np.max(h),np.min(h[np.nonzero(h)]))
    ymin,ymax = np.inf,0

    if stacked:
        stack = Stack()
        stack.add( *[ sample for sample in samples if sample.is_bkg ] )
        stack.sort(key=lambda sample : sample.scaled_nevents,reverse=not log)
        datalist,errors,weights,labels,attrs = stack.datalist(),stack.errors(),stack.weights(),stack.labels(),stack.attrs()
        histo,error = stack_error(ax,datalist,bins=bins,errors=errors,weights=weights,label=labels,log=log,**attrs)
        stack.histo = histo; stack.error = error; stack.color = 'black'
        denlist.append(stack)

        hmax,hmin = get_extrema(histo)
        ymax,ymin = max(ymax,hmax),min(ymin,hmin)

    num = None
    for sample in samples:
        histo = sample.histo
        hmax,hmin = get_extrema(histo)
        ymax,ymin = max(ymax,hmax),min(ymin,hmin)

        if sample.is_data:
            histo,error,label = sample.histo,sample.error,sample.label
            ax.errorbar(bin_centers,histo,yerr=None,xerr=bin_widths,color='black',marker='o',linestyle='None',label=label)

            num = histo
        elif sample.is_signal or not stacked:
            data,histo,error,weight,label,attrs = sample.data,sample.histo,sample.error,sample.weight,sample.label,sample.attrs
            attrs["histtype"] = "step" if len(samples) > 1 else "bar"; attrs["linewidth"] = 2
            hist_error(ax,data,bins=bins,error=error,weights=weight,label=label,log=log,**attrs)
            
            if not samples.has_data or not sample.is_signal: denlist.append(sample)
            
        
    if ylabel is None: ylabel = "Fraction of Events" if density else "Events"
    if lumi != 1: title = f"{lumi/1000:0.1f} fb^{-1} {lumi_tag}"
    
    if kwargs.get('ylim',None) is None:
        if log: ymin,ymax = 0.1*ymin,10*ymax
        else:   ymin,ymax = 0.9*ymin,1.1*ymax
        kwargs['ylim'] = (ymin,ymax)
    
    format_axis(ax,xlabel=xlabel,ylabel=ylabel,title=title,**kwargs)
    ax.legend()
    if ratio:
        if num is None: num = denlist.pop(0).histo
        
        options = { key[2:]:value for key,value in kwargs.items() if key.startswith("r_") }
        denerrs = [ sample.error for sample in denlist ]
        colors = [ sample.color for sample in denlist ]
        denlist = [ sample.histo for sample in denlist ]
        ratio_plot(num,denlist,denerrs,bins,xlabel,figax,colors=colors,**options)
    
    return (fig,ax)
    
def plot_mask_stack_comparison(datalist,bins=None,title=None,xlabel=None,figax=None,density=0,labels=None,histtype="bar",colors=None):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax
    
    if labels is None: labels = [ "" for _ in datalist ]
        
    labels = [f"{label} ({ak.size(data):.2e})"for data,label in zip(datalist,labels)]
    info = {"bins":bins,"label":labels,"density":density}
    if histtype: info["histtype"] = histtype
    if colors: info["color"] = colors
    ax.hist(datalist,stacked=True,**info)
        
    ax.set_ylabel("Fraction of Events" if density else "Events")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
#     if density: ax.set_ylim([0,1])
    ax.legend()
    return (fig,ax)


def hist2d_simple(xdata,ydata,xbins=None,ybins=None,title=None,xlabel=None,ylabel=None,figax=None,weights=None,lumikey=None,density=0,log=1,grid=False,label=None,**kwargs):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    xdata = flatten(xdata)
    ydata = flatten(ydata)
    
    lumi,lumi_tag = lumiMap[lumikey]
    if weights is not None: weights = lumi*flatten(weights)

    if xbins is None: xbins = autobin(xdata)
    if ybins is None: ybins = autobin(ydata)

    nevnts = ak.size(xdata)
    if weights is not None: nevnts = ak.sum(weights)
    
    n,bx,by,im = ax.hist2d(np.array(xdata),np.array(ydata),(xbins,ybins),weights=weights,density=density,norm=clrs.LogNorm() if log else clrs.Normalize(),cmap="jet")
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if label: ax.text(0.05,0.9,f"{label} ({nevnts:0.2e})",transform=ax.transAxes)
        
    if grid:
        ax.set_yticks(ybins)
        ax.set_xticks(xbins)
        ax.grid()
    fig.colorbar(im,ax=ax)
    return (fig,ax)
    
def plot_barrel_display(eta,phi,weight,nbins=20,figax=None,cblabel=None,cmin=0.01):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax
    
    eta =    ak.to_numpy(ak.flatten(eta,axis=None))
    phi =    ak.to_numpy(ak.flatten(phi,axis=None))
    weight = ak.to_numpy(ak.flatten(weight,axis=None))

    max_eta = max(ak.max(np.abs(eta)),2.5)
    
    xbins = np.linspace(-max_eta,max_eta,nbins)
    ybins = np.linspace(-3.14159,3.14159,nbins)

    n,bx,by,im = ax.hist2d(eta,phi,bins=(xbins,ybins),weights=weight,cmin=cmin)
    ax.set_xlabel("Jet Eta")
    ax.set_ylabel("Jet Phi")
    ax.grid()

    cb = fig.colorbar(im,ax=ax)
    if cblabel: cb.ax.set_ylabel(cblabel)
    return (fig,ax)

def plot_endcap_display(eta,phi,weight,nbins=20,figax=None):
    if figax is None: figax = plt.subplots(projection='polar')
    (fig,ax) = figax
    
    eta =    ak.to_numpy(ak.flatten(eta,axis=None))
    phi =    ak.to_numpy(ak.flatten(phi,axis=None))
    weight = ak.to_numpy(ak.flatten(weight,axis=None))/ak.max(weight,axis=None)
    
    for p,w in zip(phi,weight):
        ax.plot([p,p],[0,1],linewidth=max(5*w,1))
        
    ax.set_ylim(0,1)
    ax.set_yticks([0,1])
    ax.set_yticklabels(["",""])
    return (fig,ax)
