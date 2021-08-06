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
    2018:[57900,"(13 TeV,2018)"],
    "Run2":[101000,"13 TeV,Run 2)"],
}

def autobin(data,nstd=3):
    ndata = ak.size(data)
    mean = ak.mean(data)
    stdv = ak.std(data)
    minim,maxim = ak.min(data),ak.max(data)
    xlo,xhi = max([minim,mean-nstd*stdv]),min([maxim,mean+nstd*stdv])
    nbins = min(int(1+np.sqrt(ndata)),50)
    return np.linspace(xlo,xhi,nbins)

def get_bin_centers(bins):
    return [ (lo+hi)/2 for lo,hi in zip(bins[:-1],bins[1:]) ]
def get_bin_widths(bins):
    return [ (hi-lo)/2 for lo,hi in zip(bins[:-1],bins[1:]) ]

def safe_divide(a,b):
    tmp = np.full_like(a,None,dtype=float)
    np.divide(a,b,out=tmp,where=(b!=0))
    return tmp

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

def graph_multi(xdata,ydatalist,xlabel=None,ylabel=None,title=None,labels=None,markers=None,colors=None,ylim=None,log=False,grid=False,figax=None):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    ndata = len(ydatalist)
    labels = init_atr(labels,None,ndata)
    markers = init_atr(markers,"o",ndata)
    colors = init_atr(colors,None,ndata)
    
    for i,(ydata,label,marker,color) in enumerate(zip(ydatalist,labels,markers,colors)):
        ax.plot(xdata,ydata,label=label,marker=marker,color=color)

    if log: ax.set_yscale('log')
    if grid: ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if any(label for label in labels): ax.legend()
    
    if ylim: ax.set_ylim(ylim)
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

def ratio_plot(histolist,bins,is_datas,xlabel,figax,**kwargs):
    options = { key[2:]:value for key,value in kwargs.items() if key.startswith("r_") }
    
    fig,ax = figax
    divider = make_axes_locatable(ax)
    ax_ratio = divider.append_axes("bottom", size="20%", pad=0.1, sharex=ax)
    data_hist = next(hist for i,hist in enumerate(histolist) if is_datas[i])
        
    xdata = get_bin_centers(bins)
    ratio_data = [ safe_divide(data_hist,hist) for i,hist in enumerate(histolist) if not is_datas[i] ]
    graph_multi(xdata,ratio_data,figax=(fig,ax_ratio),xlabel=xlabel,ylabel="Ratio",**options)

def hist_multi(datalist,bins=None,title=None,xlabel=None,ylabel=None,figax=None,density=0,log=0,ratio=False,
               weights=None,labels=None,histtypes=None,colors=None,ylim=None,lumikey=None,is_datas=None,**kwargs):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    lumi,lumi_tag = lumiMap[lumikey]
    nhist = len(datalist)

    histdef = "bar" if nhist == 1 else "step"

    datalist = [ ak.to_numpy(ak.flatten(data,axis=None)) for data in datalist ]
    labels = init_atr(labels,"",nhist)
    histtypes = init_atr(histtypes,histdef,nhist)
    colors = init_atr(colors,None,nhist)
    weights = init_atr(weights,None,nhist)
    is_datas = init_atr(is_datas,False,nhist)
        
    if ratio: ratio = any(is_data for is_data in is_datas)
        
    if bins is None: bins = autobin(datalist[0])

    histolist = []
    for i,(data,label,histtype,color,weight,is_data) in enumerate(zip(datalist,labels,histtypes,colors,weights,is_datas)):
        nevnts = ak.size(data)
        is_scaled = weight is not None
        weight = ak.to_numpy(ak.flatten(weight,axis=None)) if (weight is not None and not is_data) else ak.to_numpy(ak.ones_like(data))
        if is_scaled and not is_data: weight = lumi*weight
        scaled_nevnts = ak.sum(weight)
        
        histo,bins = np.histogram(data,bins=bins,weights=weight)
        error_scale = safe_divide(np.sqrt(histo),histo)
        
        if density: weight = weight * 1/scaled_nevnts
        histo,bins = np.histogram(data,bins=bins,weights=weight)
        error = error_scale*histo
        histolist.append(histo)
        
        info = {"bins":bins,"label":f"{label} ({scaled_nevnts:.2e})","weights":weight}
        if is_data: color = 'black'
        if histtype: info["histtype"] = histtype
        if color: info["color"] = color
        if histtype == "step": info["linewidth"] = 2
        if log: info["log"] = log
        
        if is_data: ax.errorbar(get_bin_centers(bins),histo,yerr=error,xerr=get_bin_widths(bins),color="black",marker="o",linestyle='None',label=info['label'])
        else: ax.hist(data,**info)
        
    if ylabel is None: ylabel = "Fraction of Events" if density else "Events"
    if lumi != 1: title = f"{lumi/1000:0.1f} fb^{-1} {lumi_tag}"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()

    if ylim is not None: ax.set_ylim(ylim)
    if ratio: ratio_plot(histolist,bins,is_datas,xlabel,figax,**kwargs)
    
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


def hist2d_simple(xdata,ydata,xbins=None,ybins=None,title=None,xlabel=None,ylabel=None,figax=None,weights=None,lumikey=None,density=0,log=1,grid=False,label=None):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    xdata = ak.to_numpy(ak.flatten(xdata,axis=None))
    ydata = ak.to_numpy(ak.flatten(ydata,axis=None))
    
    lumi,lumi_tag = lumiMap[lumikey]
    if weights is not None: weights = lumi*ak.to_numpy(weights)

    if xbins is None: xbins = autobin(xdata)
    if ybins is None: ybins = autobin(ydata)

    nevnts = ak.size(xdata)
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
