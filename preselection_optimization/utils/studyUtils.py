#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from . import *


# In[80]:


def presel_study(branches=None,mask=None,jet_mask=None,njets=6,
                 selected=None,sixb_selected=None,nsixb_selected=None,sixb_remaining=None,nsixb_remaining=None,
                 title=None,saveas=None,plot=True,missing=True,**kwargs):
    if mask is None: mask = np.ones( branches.nevents,dtype=bool )
    
    print(f"--- {title} ---")
    varinfo = {
        f"jet_m":{"bins":np.linspace(0,60,100),"xlabel":"Top Selected Jet Mass"},
        f"jet_ptRegressed":{"bins":np.linspace(0,200,100),"xlabel":"Top Selected Jet Pt (GeV)"},
        f"jet_E":{"bins":np.linspace(0,300,100),"xlabel":"Top Selected Jet Energy"},
        f"jet_btag":{"bins":np.linspace(0,1,100),"xlabel":"Top Selected Jet Btag"},
        f"jet_eta":{"bins":np.linspace(-3,3,100),"xlabel":"Top Selected Jet Eta"},
        f"jet_phi":{"bins":np.linspace(-3.14,3.14,100),"xlabel":"Top Selected Jet Phi"},
    }
    
    values = calculate_values(branches,mask,nsixb_selected,nsixb_remaining,njets)
    print_values(values)
    
    if not plot: return
    
    nsixb = min(6,njets)
    total_njet = branches["njet"][mask]
    if jet_mask is not None: total_njet = ak.sum(jet_mask[mask],axis=-1)
    
    signal_njet = branches["njet"][mask & branches.sixb_found_mask]
    if jet_mask is not None: signal_njet = ak.sum(jet_mask[mask & branches.sixb_found_mask],axis=-1)
    
    fig = plt.figure( figsize=(15,10) )
    nrows,ncols = 3,3
    gs = fig.add_gridspec(nrows=nrows,ncols=ncols)
    
    ax1 = fig.add_subplot( gs[0,0])
    ax2 = fig.add_subplot( gs[0,1])
    ax3 = fig.add_subplot( gs[0,2])
    
    plot_mask_simple_comparison(total_njet,signal_njet
                                ,figax=(fig,ax1),bins=np.linspace(-0.5,21.5,23),xlabel="Total Number of Jets in Selection")
    plot_mask_simple_comparison(nsixb_selected[mask],nsixb_selected[mask & branches.sixb_found_mask],density=1
                                ,figax=(fig,ax2),bins=np.linspace(-0.5,nsixb+0.5,nsixb+2),xlabel="N Signal Six B in Top Selection")
    plot_mask_simple_comparison(nsixb_remaining[mask],nsixb_remaining[mask & branches.sixb_found_mask],density=1
                                ,figax=(fig,ax3),bins=np.linspace(-0.5,6.5,8),xlabel="N Signal Six B Remaining after Selection")
    
    plot_method = plot_mask_difference if missing else plot_mask_comparison
    
    for i,(var,info) in enumerate(varinfo.items()):
        ax = fig.add_subplot( gs[int( (i+3)/ncols),(i+3)%ncols] )
        plot_method(var,branches=branches,mask=mask,selected=selected,signal_selected=selected,
                             sixb_selected=sixb_selected,figax=(fig,ax),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: fig.savefig(f"plots/{saveas}")
    

