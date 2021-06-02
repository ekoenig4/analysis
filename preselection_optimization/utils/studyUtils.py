#!/usr/bin/env python
# coding: utf-8

from . import *

from datetime import date
import os

date_tag = date.today().strftime("%Y%m%d")

def save_scores(score,saveas):
    directory = f"plots/{date_tag}_plots/scores"
    if not os.path.isdir(directory): os.makedirs(directory)
    score.savetex(f"{directory}/{saveas}")
    
def jet_order_simple_study(selection=None,title=None,saveas=None,plot=True,missing=False,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    varinfo = {
#         f"jet_m":{"bins":np.linspace(0,60,100),"xlabel":"Top Selected Jet Mass"},
#         f"jet_E":{"bins":np.linspace(0,300,100),"xlabel":"Top Selected Jet Energy"},
        f"jet_ptRegressed":{"bins":np.linspace(0,500,100),"xlabel":"Top Selected Jet Pt (GeV)"},
        f"jet_btag":{"bins":np.linspace(0,1,100),"xlabel":"Top Selected Jet Btag"},
        f"jet_eta":{"bins":np.linspace(-3,3,100),"xlabel":"Top Selected Jet Eta"},
        f"jet_phi":{"bins":np.linspace(-3.14,3.14,100),"xlabel":"Top Selected Jet Phi"},
    }
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    if saveas:
        directory = f"plots/{date_tag}_plots/order_simple"
        if not os.path.isdir(directory): os.makedirs(directory)
    
    jets_ordered = ak.pad_none(selection.jets_ordered,6,axis=-1)
    sixb_ordered = ak.pad_none(selection.sixb_ordered,6,axis=-1)

    njets = min(6,selection.njets) if selection.njets != -1 else 6
    match_ratio = np.array(())
    
    nrows,ncols=1,1
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10,5))
    
    for ijet in range(njets):
        nsixb_mask = selection.nsixb_selected > ijet
        ijets_mask = get_jet_index_mask(branches,jets_ordered[:,ijet][:,np.newaxis])
        isixb_mask = get_jet_index_mask(branches,sixb_ordered[:,ijet][:,np.newaxis])
        
        inter_mask = ijets_mask & isixb_mask
        
        ijets_nevents = ak.sum(nsixb_mask)
        
        ijets_mask = exclude_jets(ijets_mask,inter_mask)
        isixb_mask = exclude_jets(isixb_mask,inter_mask)
        
        imiss_nevents = ak.sum(ak.flatten(isixb_mask[nsixb_mask]))
        
        imatch_ratio = 1-imiss_nevents/float(ijets_nevents)
        match_ratio = np.append(match_ratio,imatch_ratio)
        
        if ak.sum(ak.flatten(isixb_mask[nsixb_mask])) == 0: 
            print(f"*** No incorrect selection in position {ijet} ***")
            continue
    graph_simple(range(njets),match_ratio,xlabel="Selected Jet Position",ylabel="Fraction of Correct Matches",ylim=(0,1),figax=(fig,axs))
    
    fig.suptitle(f"{title}")
    fig.tight_layout()
    plt.show()
    if saveas: fig.savefig(f"{directory}/{saveas}")

def jet_order_study(selection=None,title=None,saveas=None,plot=True,missing=False,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    varinfo = {
#         f"jet_m":{"bins":np.linspace(0,60,100),"xlabel":"Top Selected Jet Mass"},
#         f"jet_E":{"bins":np.linspace(0,300,100),"xlabel":"Top Selected Jet Energy"},
        f"jet_ptRegressed":{"bins":np.linspace(0,500,100),"xlabel":"Top Selected Jet Pt (GeV)"},
        f"jet_btag":{"bins":np.linspace(0,1,100),"xlabel":"Top Selected Jet Btag"},
        f"jet_eta":{"bins":np.linspace(-3,3,100),"xlabel":"Top Selected Jet Eta"},
        f"jet_phi":{"bins":np.linspace(-3.14,3.14,100),"xlabel":"Top Selected Jet Phi"},
    }
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    if saveas:
        directory = f"plots/{date_tag}_plots/order"
        if not os.path.isdir(directory): os.makedirs(directory)
    
    jets_ordered = ak.pad_none(selection.jets_ordered,6,axis=-1)
    sixb_ordered = ak.pad_none(selection.sixb_ordered,6,axis=-1)

    njets = min(6,selection.njets) if selection.njets != -1 else 6
    
    for ijet in range(njets):
        nsixb_mask = selection.nsixb_selected > ijet
        ijets_mask = get_jet_index_mask(branches,jets_ordered[:,ijet][:,np.newaxis])
        isixb_mask = get_jet_index_mask(branches,sixb_ordered[:,ijet][:,np.newaxis])
        
        inter_mask = ijets_mask & isixb_mask
        
        ijets_mask = exclude_jets(ijets_mask,inter_mask)
        isixb_mask = exclude_jets(isixb_mask,inter_mask)
        
        if ak.sum(ak.flatten(isixb_mask[nsixb_mask])) == 0: 
            print(f"*** No incorrect selection in position {ijet} ***")
            continue
            
        nrows,ncols=2,4
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(20,10))
        for i,(var,info) in enumerate(varinfo.items()):
            ord_info = dict(info)
            ord_info["xlabel"] = f"{ordinal(ijet+1)} {info['xlabel']}"
            jets_var = ak.flatten(branches[var][ijets_mask][nsixb_mask])
            sixb_var = ak.flatten(branches[var][isixb_mask][nsixb_mask])
            plot_mask_simple_comparison(jets_var,sixb_var,figax=(fig,axs[0,i]),**ord_info,
                                        label1="Non Signal Jets Selected",label2="Signal Jets Missed",
                                        color1="tab:orange",color2="black",histtype2="step")
            
            xlabel_2d = ord_info["xlabel"].replace("Selected","Missed Signal")
            ylabel_2d = ord_info["xlabel"].replace("Selected","Selected Non Signal")
            plot_mask_simple_2d_comparison(sixb_var,jets_var,xbins=info["bins"],ybins=info["bins"],
                                   figax=(fig,axs[1,i]),xlabel=xlabel_2d,ylabel=ylabel_2d,log=1)
        fig.suptitle(f"{ordinal(ijet+1)} {title}")
        fig.tight_layout()
        plt.show()
        if saveas: fig.savefig(f"{directory}/{ordinal(ijet+1)}_{saveas}")
    graph_simple(np.linspace(1,njets,njets),match_ratio,xlabel="Selected Jet Position",ylabel="Fraction of Correct Matches")

def njet_study(selection=None,title=None,saveas=None,plot=True,missing=False,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    full_njet = ak.sum(selection.jet_mask,axis=-1)
    
    total_njet = full_njet[selection.mask]
    signal_njet = full_njet[selection.mask & branches.sixb_found_mask]
        
    max_jets = ak.max(total_njet)
    sixb_pos = selection.sixb_position
    
    total_sixb_pos = ak.flatten( sixb_pos[selection.mask] )
    signal_sixb_pos = ak.flatten( sixb_pos[selection.mask & branches.sixb_found_mask] )
    
    last_index = get_jet_index_mask(branches,ak.count(sixb_pos,axis=-1)[:,np.newaxis]-1,jets=sixb_pos)
    total_last_pos = ak.flatten(sixb_pos[last_index][selection.mask])
    signal_last_pos = ak.flatten(sixb_pos[last_index][selection.mask & branches.sixb_found_mask])
    
    expand_signal_njets = ak.broadcast_arrays(signal_njet,selection.sixb_position[selection.mask & branches.sixb_found_mask])[0]
    expand_signal_njets = ak.flatten(expand_signal_njets)
    
    total_nsixb_captured = selection.nsixb_captured[selection.mask]
    signal_nsixb_captured = selection.nsixb_captured[selection.mask & branches.sixb_found_mask]
    expand_nsixb_captured = ak.broadcast_arrays(signal_nsixb_captured,selection.sixb_position[selection.mask & branches.sixb_found_mask])[0]
    expand_nsixb_captured = ak.flatten(expand_nsixb_captured)
    
    nrows,ncols = 1,3
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(20,5) )
    
    plot_mask_simple_comparison(total_njet,signal_njet,bins=np.linspace(0,16,17),
                                figax=(fig,axs[0]),xlabel="N Captured Jets")
    
    plot_mask_simple_comparison(total_nsixb_captured,signal_nsixb_captured,bins=np.linspace(0,7,8),
                                figax=(fig,axs[1]),xlabel="N Captured Signal Jets")
    
    
#     plot_mask_simple_comparison(total_sixb_pos,signal_sixb_pos,
#                                 figax=(fig,axs[2]),bins=np.linspace(0,11,12),xlabel="Captured Signal Jet Position")
    
    plot_mask_simple_comparison(total_last_pos,signal_last_pos,
                                 figax=(fig,axs[2]),bins=np.linspace(0,11,12),xlabel="Last Captured Signal Jet Position")
    
#     axs[1,0].set_visible(False)
#     plot_mask_simple_2d_comparison(expand_nsixb_captured,signal_sixb_pos,xbins=np.linspace(0,7,8),ybins=np.linspace(0,max_jets+1,max_jets+2),
#                                    figax=(fig,axs[1,0]),xlabel="N Captured Signal Jets",ylabel="Captured Signal Jet Position")
    
#     plot_mask_simple_2d_comparison(signal_nsixb_captured,signal_njet,xbins=np.linspace(0,7,8),ybins=np.linspace(0,16,17),
#                                    figax=(fig,axs[1,1]),xlabel="N Captured Signal Jets",ylabel="N Captured Jets",log=1,label="Pure Events",grid=True)

#     plot_mask_simple_2d_comparison(signal_sixb_pos,expand_signal_njets,xbins=np.linspace(0,11,12),ybins=np.linspace(0,16,17),
#                                    figax=(fig,axs[1,2]),xlabel="Captured Signal Jet Position",ylabel="N Captured Jets",log=1,label="Pure Events",grid=True)


    
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: 
        directory = f"plots/{date_tag}_plots/njet"
        if not os.path.isdir(directory): os.makedirs(directory)
        fig.savefig(f"{directory}/{saveas}")
    

def presel_study(selection=None,title=None,saveas=None,plot=True,missing=False,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    varinfo = {
        f"jet_m":{"bins":np.linspace(0,60,100),"xlabel":"Top Selected Jet Mass"},
        f"jet_E":{"bins":np.linspace(0,300,100),"xlabel":"Top Selected Jet Energy"},
        f"jet_ptRegressed":{"bins":np.linspace(0,200,100),"xlabel":"Top Selected Jet Pt (GeV)"},
        f"jet_btag":{"bins":np.linspace(0,1,100),"xlabel":"Top Selected Jet Btag"},
        f"jet_eta":{"bins":np.linspace(-3,3,100),"xlabel":"Top Selected Jet Eta"},
        f"jet_phi":{"bins":np.linspace(-3.14,3.14,100),"xlabel":"Top Selected Jet Phi"},
    }
    
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    nsixb = min(6,selection.njets if selection.njets != -1 else 6)
    nsixb_selected = selection.nsixb_selected
    nsixb_remaining = selection.nsixb_remaining

    nrows,ncols = 4,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(9,15) )
    plot_mask_simple_comparison(nsixb_selected[selection.mask],nsixb_selected[selection.mask & branches.sixb_found_mask]
                                ,figax=(fig,axs[0,0]),bins=np.linspace(0,nsixb+1,nsixb+2),xlabel="N Selected Signal Jets")
    plot_mask_simple_comparison(nsixb_remaining[selection.mask],nsixb_remaining[selection.mask & branches.sixb_found_mask]
                                ,figax=(fig,axs[0,1]),bins=np.linspace(0,7,8),xlabel="N Signal Jets Remaining after Selection")
    
    plot_method = plot_mask_difference if missing else plot_mask_comparison
    
    for i,(var,info) in enumerate(varinfo.items()):
        plot_method(var,branches=branches,mask=selection.mask,jet_selected=selection.jets_selected,sixb_selected=selection.sixb_selected,
                    figax=(fig,axs[int(1+i/ncols),i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: 
        directory = f"plots/{date_tag}_plots/presel"
        if not os.path.isdir(directory): os.makedirs(directory)
        fig.savefig(f"{directory}/{saveas}")
    

