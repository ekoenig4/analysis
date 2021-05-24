#!/usr/bin/env python
# coding: utf-8

from . import *

def njet_study(branches=None,mask=None,jet_mask=None,njets=None,
                 selected=None,sixb_selected=None,nsixb_selected=None,
                 sixb_remaining=None,nsixb_remaining=None,sixb_captured=None,nsixb_captured=None,
                 title=None,saveas=None,plot=True,missing=False,**kwargs):
    if mask is None: mask = np.ones( branches.nevents,dtype=bool )
    
    print(f"--- {title} ---")
    
    values = calculate_values(branches,mask,nsixb_selected,nsixb_remaining,nsixb_captured,njets)
    print_values(values)
    
    if not plot: return
    
    total_njet = branches["njet"][mask]
    if jet_mask is not None: total_njet = ak.sum(jet_mask[mask],axis=-1)
    
    signal_njet = branches["njet"][mask & branches.sixb_found_mask]
    if jet_mask is not None: signal_njet = ak.sum(jet_mask[mask & branches.sixb_found_mask],axis=-1)
    
    nrows,ncols = 1,3
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(20,5) )
    
    plot_mask_simple_comparison(total_njet,signal_njet
                                ,figax=(fig,axs[0]),bins=np.linspace(-0.5,21.5,23),xlabel="Total Number of Jets Captured in Selection")
    
    plot_mask_simple_comparison(nsixb_captured[mask],nsixb_captured[mask & branches.sixb_found_mask]
                                ,figax=(fig,axs[1]),bins=np.linspace(-0.5,6.5,8),xlabel="N Signal Six B Captured in Selection")

    plot_mask_simple_2d_comparison(nsixb_captured[mask & branches.sixb_found_mask],signal_njet,xbins=np.linspace(-0.5,6.5,8),ybins=np.linspace(-0.5,21.5,23),
                                   xlabel="N Signal Six B Captured in Selection",ylabel="Total Number of Jets Captured in Selection",figax=(fig,axs[2]))
    

def presel_study(branches=None,mask=None,jet_mask=None,njets=None,
                 jet_selected=None,njet_selected=None,sixb_selected=None,nsixb_selected=None,
                 sixb_remaining=None,nsixb_remaining=None,sixb_captured=None,nsixb_captured=None,
                 title=None,saveas=None,plot=True,missing=False,**kwargs):
    
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
    
    values = calculate_values(branches,mask,nsixb_selected,nsixb_remaining,nsixb_captured,njets)
    print_values(values)
    
    if not plot: return
    
    nsixb = min(6,njets)

    nrows,ncols = 3,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(20,15) )
    plot_mask_simple_comparison(nsixb_selected[mask],nsixb_selected[mask & branches.sixb_found_mask]
                                ,figax=(fig,axs[0,0]),bins=np.linspace(-0.5,nsixb+0.5,nsixb+2),xlabel="N Signal Six B in Top Selection")
    plot_mask_simple_comparison(nsixb_remaining[mask],nsixb_remaining[mask & branches.sixb_found_mask]
                                ,figax=(fig,axs[0,1]),bins=np.linspace(-0.5,6.5,8),xlabel="N Signal Six B Remaining after Selection")
    
    plot_method = plot_mask_difference if missing else plot_mask_comparison
    
    for i,(var,info) in enumerate(varinfo.items()):
        plot_method(var,branches=branches,mask=mask,jet_selected=jet_selected,sixb_selected=sixb_selected,
                    figax=(fig,axs[int(1+i/ncols),i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: fig.savefig(f"plots/{saveas}")
    

