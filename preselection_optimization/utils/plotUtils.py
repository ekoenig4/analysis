#!/usr/bin/env python
# coding: utf-8

from . import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

def plot_simple(variable,branches,mask=None,selected=None,bins=None,xlabel=None,title=None,label=None,figax=None):
    if figax is None: figax = plt.subplots()
    if mask is None: mask = np.ones(ak.size(branches['Run']),dtype=bool)
    (fig,ax) = figax
    
    data = branches[variable][mask]
    if selected is not None: data = branches[variable][mask][selected]
    data = ak.flatten( data,axis=-1 )
    
    ax.hist(data,bins=bins,label=label)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()

def plot_mask_comparison(variable,branches,mask=None,jet_selected=None,sixb_selected=None
                              ,bins=None,xlabel=None,title=None,label="All Events",figax=None):
    if figax is None: figax = plt.subplots()
    if mask is None: mask = np.ones(ak.size(branches['Run']),dtype=bool)
    (fig,ax) = figax
    
    data1 = branches[variable][mask];                   
    data2 = branches[variable][mask & branches.sixb_found_mask]; 
    data3 = branches[variable][mask & branches.sixb_found_mask]; 
    
#     signal_non_sixb = exclude_jets(jet_selected,sixb_selected)
#     signal_tru_sixb = exclude_jets(jet_selected,signal_non_sixb)
    
    sixb_not_selected = exclude_jets(sixb_selected,jet_selected)
    sixb_selected = exclude_jets(sixb_selected,sixb_not_selected)
    
    data3_selected = ak.flatten(data3[sixb_selected[mask & branches.sixb_found_mask]],axis=-1)
    n_sixb_selected = ak.size(data3_selected)
    
    data1 = ak.flatten(data1[jet_selected[mask]],axis=-1)
    data2 = ak.flatten(data2[jet_selected[mask & branches.sixb_found_mask]],axis=-1)
    data3 = ak.flatten(data3[sixb_selected[mask & branches.sixb_found_mask]],axis=-1)
    
    nevts1 = ak.size(data1)
    nevts2 = ak.size(data2)
    nevts3 = ak.size(data3)
    
    ax.hist(data1,bins=bins,label=f"{label} ({nevts1:.2e})")
    ax.hist(data2,bins=bins,label=f"Pure Events ({nevts2:.2e})")
    ax.hist(data3_selected,bins=bins,label=f'Signal BJet Selected({n_sixb_selected:.2e})',color="black")
        
    ax.set_ylabel("Events")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()

def plot_mask_difference(variable,branches,mask=None,jet_selected=None,sixb_selected=None
                         ,bins=None,xlabel=None,title=None,label="All Events",figax=None):
    if figax is None: figax = plt.subplots()
    if mask is None: mask = np.ones(ak.size(branches['Run']),dtype=bool)
    (fig,ax) = figax
    
    data1 = branches[variable][mask];                   
    data2 = branches[variable][mask & branches.sixb_found_mask]; 
    data3 = branches[variable][mask & branches.sixb_found_mask]; 
    
    event_non_sixb = exclude_jets(jet_selected,sixb_selected)
#     signal_tru_sixb = exclude_jets(jet_selected,signal_non_sixb)
    
    signal_non_sixb = exclude_jets(jet_selected,sixb_selected)
#     signal_tru_sixb = exclude_jets(jet_selected,signal_non_sixb)
    
    sixb_not_found = exclude_jets(sixb_selected,signal_selected)
#     sixb_found = exclude_jets(sixb_selected,sixb_not_found)
    
    data1 = ak.flatten(data1[event_non_sixb[mask]],axis=-1)
    data2 = ak.flatten(data2[signal_non_sixb[mask & branches.sixb_found_mask]],axis=-1)
    data3 = ak.flatten(data3[sixb_not_found[mask & branches.sixb_found_mask]],axis=-1)
    
    nevts1 = ak.size(data1 )
    nevts2 = ak.size(data2 )
    nevts3 = ak.size(data3 )
    
    ax.hist(data1,bins=bins,label=f"{label} Non Signal Jets ({nevts1:.2e})")
    ax.hist(data2,bins=bins,label=f"Gen Matched Event Non Signal Jets ({nevts2:.2e})")
    ax.hist(data3,bins=bins,label=f'Signal BJet Missed({nevts3:.2e})',color="black",histtype="step")
        
    ax.set_ylabel("Events")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()

def plot_mask_simple_comparison(selected,signal_selected,bins=None,label="All Events",title=None,xlabel=None,figax=None,density=0):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax
        
    nevts1 = ak.size(selected)
    nevts2 = ak.size(signal_selected)
        
    ax.hist(selected,bins=bins,label=f"{label} ({nevts1:.2e})",density=density)
    ax.hist(signal_selected,bins=bins,label=f"Pure Events ({nevts2:.2e})",
            density=density,histtype="step" if density else "bar",linewidth=2 if density else None)
    ax.set_ylabel("Percent of Events" if density else "Events")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if density: ax.set_ylim([0,1])
    ax.legend()


def plot_mask_simple_2d_comparison(xdata,ydata,xbins=None,ybins=None,title=None,xlabel=None,ylabel=None,figax=None,density=0):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax

    n,bx,by,im = ax.hist2d(np.array(xdata),np.array(ydata),(xbins,ybins),density=density)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.colorbar(im,ax=ax)

