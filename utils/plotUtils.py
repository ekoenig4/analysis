#!/usr/bin/env python
# coding: utf-8

from . import *

import matplotlib.colors as clrs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

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
    xlo,xhi = max([minim,mean-nstd*stdv]),min([mean+nstd*stdv])
    nbins = min(int(1+np.sqrt(ndata)),50)
    return np.linspace(xlo,xhi,nbins)

def init_atr(atr,init,size):
    if atr is None: return [init]*size
    atr = list(atr)
    return (atr + size*[init])[:size]

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

def graph_multi(xdata,ydatalist,xlabel=None,ylabel=None,title=None,labels=None,markers=None,colors=None,ylim=None,figax=None):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    ndata = len(ydatalist)
    labels = init_atr(labels,"",ndata)
    markers = init_atr(markers,"o",ndata)
    colors = init_atr(colors,None,ndata)
    
    for i,(ydata,label,marker,color) in enumerate(zip(ydatalist,labels,markers,colors)):
        ax.plot(xdata,ydata,label=label,marker=marker,color=color)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
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

def hist_multi(datalist,bins=None,title=None,xlabel=None,ylabel=None,figax=None,density=0,log=0,
               weights=None,labels=None,histtypes=None,colors=None,lumikey=None):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    lumi,lumi_tag = lumiMap[lumikey]
    nhist = len(datalist)

    histdef = "bar" if nhist == 1 else "step"

    datalist = [ ak.flatten(data,axis=None) for data in datalist ]
    
    labels = init_atr(labels,"",nhist)
    histtypes = init_atr(histtypes,histdef,nhist)
    colors = init_atr(colors,None,nhist)
    weights = init_atr(weights,None,nhist)
        
    if bins is None: bins = autobin(datalist[0])
        
    for i,(data,label,histtype,color,weight) in enumerate(zip(datalist,labels,histtypes,colors,weights)):
        nevnts = ak.size(data)
        is_scaled = weight is not None
        weight = ak.flatten(weight,axis=None) if weight is not None else ak.ones_like(data)
        if is_scaled: weight = lumi*weight
        
        scaled_nevnts = ak.sum(weight)
        
        info = {"bins":bins,"label":f"{label} ({scaled_nevnts:.2e})","weights":weight}
        if histtype: info["histtype"] = histtype
        if color: info["color"] = color
        if histtype == "step": info["linewidth"] = 2
        if density: info["weights"] = weight * 1/scaled_nevnts
        if log: info["log"] = log
            
        ax.hist(data,**info)
        
    if ylabel is None: ylabel = "Fraction of Events" if density else "Events"
    if lumi != 1: title = f"{lumi/1000:0.1f} fb^{-1} {lumi_tag}"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
#     if density: ax.set_ylim([0,1])
    ax.legend()
    return (fig,ax)
    
def plot_mask_stack_comparison(datalist,bins=None,title=None,xlabel=None,figax=None,density=0,
                         labels=None,histtype="bar",colors=None):
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


def hist2d_simple(xdata,ydata,xbins=None,ybins=None,title=None,xlabel=None,ylabel=None,figax=None,density=0,log=1,grid=False,label=None):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    xdata = ak.to_numpy(ak.flatten(xdata,axis=None))
    ydata = ak.to_numpy(ak.flatten(ydata,axis=None))

    nevnts = ak.size(xdata)
    n,bx,by,im = ax.hist2d(np.array(xdata),np.array(ydata),(xbins,ybins),density=density,norm=clrs.LogNorm() if log else clrs.Normalize(),cmap="jet")
    
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
    
def plot_jet_display(jet_eta,jet_phi,jet_weight,nbins=20,figax=None,cblabel=None,cmin=0.01):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax
    xbins = np.linspace(-2.4,2.4,nbins)
    ybins = np.linspace(-3.14159,3.14159,nbins)

    n,bx,by,im = ax.hist2d(jet_eta,jet_phi,bins=(xbins,ybins),weights=jet_weight,cmin=cmin)
    ax.set_xlabel("Jet Eta")
    ax.set_ylabel("Jet Phi")
    ax.grid()

    cb = fig.colorbar(im,ax=ax)
    if cblabel: cb.ax.set_ylabel(cblabel)
    return (fig,ax)
