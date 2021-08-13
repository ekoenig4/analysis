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


def get_bin_centers(bins):
    return [ (lo+hi)/2 for lo,hi in zip(bins[:-1],bins[1:]) ]
def get_bin_widths(bins):
    return [ (hi-lo)/2 for lo,hi in zip(bins[:-1],bins[1:]) ]

def safe_divide(a,b):
    tmp = np.full_like(a,None,dtype=float)
    np.divide(a,b,out=tmp,where=(b!=0))
    return tmp

class Sample:
    def __init__(self,data,bins,lumi=1,label="",weight=None,is_data=False,is_signal=False,density=False,**attrs):
        self.data = ak.to_numpy(ak.flatten(data,axis=None))
        self.nevnts = ak.size(self.data)
        
        if bins is None: bins = autobin(self.data)

        self.lumi = lumi
        self.weight = weight
        self.is_data = is_data
        self.is_signal = is_signal
        self.density = density
        
        for key,value in attrs.items(): setattr(self,key,value)

        if is_data: self.color = "black"
        self.linewidth = 2

        is_scaled = self.weight is not None
        self.weight = ak.to_numpy(ak.flatten(self.weight,axis=None)) if (self.weight is not None) else ak.to_numpy(ak.ones_like(self.data))
        if is_scaled and not is_data: self.weight = lumi*self.weight
        self.scaled_nevnts = ak.sum(self.weight)
        self.label = f"{label} ({self.scaled_nevnts:.2e})"
        
        if density: self.weight = self.weight * 1/self.scaled_nevnts
        self.histo,self.bins = np.histogram(self.data,bins=bins,weights=self.weight)
        sumw2,_ = np.histogram(self.data,bins=self.bins,weights=self.weight**2)
        self.error = np.sqrt(sumw2)

        self.bin_centers,self.bin_widths = get_bin_centers(bins),get_bin_widths(bins)
        
    def hist_error(self):
        info = {"bins":self.bins,"label":self.label,"weights":self.weight,"linewidth":self.linewidth}
        if hasattr(self,"histtype"): info["histtype"] = self.histtype
        if hasattr(self,"color"): info["color"] = self.color
        return [self.data,self.error],info
    def errorbar(self):
        info = dict(yerr=None,xerr=self.bin_widths,color="black",marker="o",linestyle='None',label=self.label)
        return [self.bin_centers,self.histo],info
        
class Samplelist(list):
    def __init__(self,datalist,bins,lumi=1,density=False,**attrs):
        self.bins = bins
        self.density = density
        self.lumi = lumi

        nhist = len(datalist)
        defaults = {
            "labels":"",
            "histtypes":"bar" if nhist == 1 else "step",
            "is_datas":False,
        }

        for key in attrs: attrs[key] = init_attr(attrs[key],defaults.get(key,None),nhist)
        for i,data in enumerate(datalist):
            sample = Sample(data,self.bins,lumi=lumi,density=density,**{key[:-1]:value[i] for key,value in attrs.items()})
            if self.bins is None: self.bins = sample.bins
            self.append(sample)
            
        self.has_data = any( sample.is_data for sample in self )
        self.nmc = sum( not( sample.is_data or sample.is_signal ) for sample in self )

def format_axis(ax,title=None,xlabel=None,ylabel=None,ylim=None,grid=False,**kwargs):
    ax.set_ylabel(ylabel)

    if grid: ax.grid()
    if type(xlabel) == list:
        ax.set_xticks(range(len(xlabel)))

        rotation = 0
        if type(xlabel[0]) == str: rotation = -45
        ax.set_xticklabels(xlabel,rotation=rotation)
    else:
            ax.set_xlabel(xlabel)
    ax.set_title(title)
    if ylim is not None: ax.set_ylim(ylim)

def autobin(data,nstd=3):
    ndata = ak.size(data)
    mean = ak.mean(data)
    stdv = ak.std(data)
    minim,maxim = ak.min(data),ak.max(data)
    xlo,xhi = max([minim,mean-nstd*stdv]),min([maxim,mean+nstd*stdv])
    nbins = min(int(1+np.sqrt(ndata)),50)
    return np.linspace(xlo,xhi,nbins)

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

def ratio_plot(num,dens,denerrs,bins,xlabel,figax,**kwargs):
    options = { key[2:]:value for key,value in kwargs.items() if key.startswith("r_") }
    
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
    graph_multi(xdata,ratio_data,yerrs=ratio_error,figax=(fig,ax_ratio),xlabel=xlabel,ylabel="Ratio",**options)

def hist_error(ax,data,error=None,**kwargs):
    histo,bins,container = ax.hist(data,**kwargs)

    if error is None: return
    
    color = container[0].get_ec()
    histtype = kwargs['histtype'] if 'histtype' in kwargs else None
    if histtype != 'step': color = 'black'
    
    bin_centers,bin_widths  = get_bin_centers(bins),get_bin_widths(bins)
    ax.errorbar(bin_centers,histo,yerr=error,fmt='none',color=color,capsize=1)

def stack_error(ax,stack,log=False,**kwargs):
    bins = stack[0].bins
    
    stack.sort(key=lambda s:s.scaled_nevnts,reverse=not log)
    stack_data = [ sample.data for sample in stack ]
    stack_weights = [ sample.weight for sample in stack ]
    stack_labels = [ sample.label for sample in stack ]
    stack_colors = [ sample.color for sample in stack ]
    stack_error = np.array([ sample.error for sample in stack ])

    stack,bins,container = ax.hist(stack_data,bins=bins,weights=stack_weights,label=stack_labels,color=stack_colors,stacked=True,log=log,**kwargs)
    
    bin_centers,bin_widths  = get_bin_centers(bins),get_bin_widths(bins)
    stack = stack[-1]
    error = np.sqrt(np.sum(stack_error**2,axis=0))
    ax.errorbar(bin_centers,stack,yerr=error,fmt='none',color='black',capsize=1,log=log)
    return stack,error

def hist_multi(datalist,bins=None,title=None,xlabel=None,ylabel=None,figax=None,density=0,log=0,ratio=False,stacked=False,
               weights=None,labels=None,histtypes=None,colors=None,lumikey=None,is_datas=None,is_signals=None,**kwargs):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    lumi,lumi_tag = lumiMap[lumikey]
    samples = Samplelist( datalist,bins,lumi=lumi,density=density,weights=weights,colors=colors,histtypes=histtypes,labels=labels,is_datas=is_datas,is_signals=is_signals )
        
    if ratio: ratio = samples.has_data
    if stacked: stacked = samples.nmc > 1
    if density: stacked = False
    
    stack,dens,denerrs = [],[],[]
    
    for i,sample in enumerate(samples):
        if sample.is_data:
            num = sample.histo
            
            _args,_kwargs = sample.errorbar()
            ax.errorbar(*_args,**_kwargs)
        elif sample.is_signal:
            _args,_kwargs = sample.hist_error()
            hist_error(ax,*_args,log=log,**_kwargs)
        elif stacked:
            stack.append(sample)
        else:
            dens.append(sample.histo)
            denerrs.append(sample.error)
            
            _args,_kwargs = sample.hist_error()
            hist_error(ax,*_args,log=log,**_kwargs)
    if stacked:
        stack,stackerr = stack_error(ax,stack,log=log)
        dens,denerrs = [stack],[stackerr]
        kwargs["r_colors"]=["black"]
        
    if ylabel is None: ylabel = "Fraction of Events" if density else "Events"
    if lumi != 1: title = f"{lumi/1000:0.1f} fb^{-1} {lumi_tag}"
    format_axis(ax,xlabel=xlabel,ylabel=ylabel,title=title,**kwargs)
    ax.legend()
    
    bins = samples.bins
    if ratio: ratio_plot(num,dens,denerrs,bins,xlabel,figax,**kwargs)
    
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
