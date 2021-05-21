#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from . import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


# In[52]:


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


# In[70]:


def plot_mask_comparison(variable,branches,mask=None,selected=None,signal_selected=None,sixb_selected=None
                              ,bins=None,xlabel=None,title=None,label="All Events",figax=None):
    if figax is None: figax = plt.subplots()
    if mask is None: mask = np.ones(ak.size(branches['Run']),dtype=bool)
    (fig,ax) = figax
    
    data1 = branches[variable][mask];                   
    data2 = branches[variable][mask & sixb_found_mask]; 
    data3 = branches[variable][mask & sixb_found_mask]; 
    
#     signal_non_sixb = exclude_jets(signal_selected,sixb_selected)
#     signal_tru_sixb = exclude_jets(signal_selected,signal_non_sixb)
    
    sixb_not_found = exclude_jets(sixb_selected,signal_selected)
    sixb_found = exclude_jets(sixb_selected,sixb_not_found)
    
    data1 = ak.flatten(data1[selected[mask]],axis=-1)
    data2 = ak.flatten(data2[selected[mask & sixb_found_mask]],axis=-1)
    data3 = ak.flatten(data3[sixb_selected[mask & sixb_found_mask]],axis=-1)
    
    data3_found = ak.flatten(data3[sixb_found[mask & sixb_found_mask]],axis=-1)
    n_sixb_found = ak.size(data3_found)
    
    nevts1 = ak.size(data1)
    nevts2 = ak.size(data2)
    nevts3 = ak.size(data3)
    
    ax.hist(data1,bins=bins,label=f"{label} ({nevts1:.2e})")
    ax.hist(data2,bins=bins,label=f"Gen Matched Six BJets ({nevts2:.2e})")
    ax.hist(data3,bins=bins,label=f"Signal Six BJets ({nevts3:.2e})",histtype="step",color="red",linewidth=2)
    ax.hist(data3_found,bins=bins,label=f'Signal Six BJet Found({n_sixb_found:.2e})',color="black")
        
    ax.set_ylabel("Events")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()


# In[74]:


def plot_mask_difference(variable,branches,mask=None,selected=None,signal_selected=None,sixb_selected=None
                              ,bins=None,xlabel=None,title=None,label="All Events",figax=None):
    if figax is None: figax = plt.subplots()
    if mask is None: mask = np.ones(ak.size(branches['Run']),dtype=bool)
    (fig,ax) = figax
    
    data1 = branches[variable][mask];                   
    data2 = branches[variable][mask & sixb_found_mask]; 
    data3 = branches[variable][mask & sixb_found_mask]; 
    
    event_non_sixb = exclude_jets(selected,sixb_selected)
#     signal_tru_sixb = exclude_jets(signal_selected,signal_non_sixb)
    
    signal_non_sixb = exclude_jets(signal_selected,sixb_selected)
#     signal_tru_sixb = exclude_jets(signal_selected,signal_non_sixb)
    
    sixb_not_found = exclude_jets(sixb_selected,signal_selected)
#     sixb_found = exclude_jets(sixb_selected,sixb_not_found)
    
    data1 = ak.flatten(data1[event_non_sixb[mask]],axis=-1)
    data2 = ak.flatten(data2[signal_non_sixb[mask & sixb_found_mask]],axis=-1)
    data3 = ak.flatten(data3[sixb_not_found[mask & sixb_found_mask]],axis=-1)
    
    nevts1 = ak.size(data1 )
    nevts2 = ak.size(data2 )
    nevts3 = ak.size(data3 )
    
    ax.hist(data1,bins=bins,label=f"{label} Non Signal Jets ({nevts1:.2e})")
    ax.hist(data2,bins=bins,label=f"Gen Matched Event Non Signal Jets ({nevts2:.2e})")
    ax.hist(data3,bins=bins,label=f'Signal Six BJet Missed({nevts3:.2e})',color="black",histtype="step")
        
    ax.set_ylabel("Events")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()


# In[57]:


def plot_mask_simple_comparison(selected,signal_selected,bins=None,label="All Events",title=None,xlabel=None,figax=None,density=0):
    if figax is None: figax = plt.subplots()
    (fig,ax) = figax
        
    nevts1 = ak.size(selected)
    nevts2 = ak.size(signal_selected)
        
    ax.hist(selected,bins=bins,label=f"{label} ({nevts1:.2e})",density=density)
    ax.hist(signal_selected,bins=bins,label=f"Gen Matched Six BJets ({nevts2:.2e})",
            density=density,histtype="step" if density else "bar",linewidth=2 if density else None)
    ax.set_ylabel("Percent of Events" if density else "Events")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if density: ax.set_ylim([0,1])
    ax.legend()

