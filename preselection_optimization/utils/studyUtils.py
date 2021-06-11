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
    

def signal_order_study(selection=None,title=None,saveas=None,plot=True,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    varinfo = {
#         f"jet_m":{"bins":np.linspace(0,60,50),"xlabel":"Top Selected Jet Mass"},
#         f"jet_E":{"bins":np.linspace(0,500,50),"xlabel":"Top Selected Jet Energy"},
        f"jet_ptRegressed":{"bins":None,"xlabel":"Top Selected Jet Pt (GeV)"},
        f"jet_eta":{"bins":np.linspace(-3,3,50),"xlabel":"Top Selected Jet Eta"},
        f"jet_btag":{"bins":np.linspace(0,1,50),"xlabel":"Top Selected Jet Btag"},
#         f"jet_phi":{"bins":np.linspace(-3.14,3.14,50),"xlabel":"Top Selected Jet Phi"},
    }
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    if saveas:
        directory = f"plots/{date_tag}_plots/order"
        if not os.path.isdir(directory): os.makedirs(directory)
            
    mask = selection.mask
    sixb_ordered = ak.pad_none(selection.sixb_selected_index,6,axis=-1)
    njets = min(6,selection.njets) if selection.njets != -1 else 6
    
    ie = 5
    for ijet in range(njets):
        labels = (f"{ordinal(ijet+1)} Signal Jets",)
        nsixb_mask = (selection.nsixb_selected > ijet) & mask
        isixb_mask = get_jet_index_mask(branches,sixb_ordered[:,ijet][:,np.newaxis])
        
        if ak.sum(ak.flatten(isixb_mask[nsixb_mask])) == 0: 
            print(f"*** No signal selection in position {ijet} ***")
            continue
            
        nrows,ncols=2,3
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(18,10))
        for i,(var,info) in enumerate(varinfo.items()):
            ord_info = dict(info)
            ord_info["xlabel"] = f"{ordinal(ijet+1)} {info['xlabel']}"
            sixb_var = branches[var][isixb_mask][nsixb_mask]
            sixb_data = ak.flatten(sixb_var)
            
            if ord_info["bins"] is None:
                mean = ak.mean(sixb_data)
                stdv = ak.std(sixb_data)
                ord_info["bins"] = np.linspace(0,mean+3*stdv,50)
                
            if var == "jet_ptRegressed": ptinfo = ord_info
            if var == "jet_eta": etainfo = ord_info
            if var == "jet_btag": btaginfo = ord_info
            
            datalist = (sixb_data,)
            plot_mask_comparison(datalist,labels=labels,figax=(fig,axs[0,i%ncols]),**ord_info)
            
        sixb_ptdata = ak.flatten( branches["jet_ptRegressed"][isixb_mask][nsixb_mask] )
        sixb_etadata = ak.flatten( branches["jet_eta"][isixb_mask][nsixb_mask] )
        sixb_btagdata = ak.flatten( branches["jet_btag"][isixb_mask][nsixb_mask] )
        
        
        plot_mask_simple_2d_comparison(sixb_etadata,sixb_btagdata,xbins=etainfo['bins'],ybins=btaginfo['bins'],
                                       xlabel=etainfo['xlabel'],ylabel=btaginfo['xlabel'],figax=(fig,axs[1,0]),log=1)
        
        plot_mask_simple_2d_comparison(sixb_ptdata,sixb_etadata,xbins=ptinfo['bins'],ybins=etainfo['bins'],
                                       xlabel=ptinfo['xlabel'],ylabel=etainfo['xlabel'],figax=(fig,axs[1,1]),log=1)
        
        plot_mask_simple_2d_comparison(sixb_ptdata,sixb_btagdata,xbins=ptinfo['bins'],ybins=btaginfo['bins'],
                                       xlabel=ptinfo['xlabel'],ylabel=btaginfo['xlabel'],figax=(fig,axs[1,2]),log=1)
            
        fig.suptitle(f"{title} {ordinal(ijet+1)} {selection.variable}")
        fig.tight_layout()
        plt.show()
        if saveas: fig.savefig(f"{directory}/{ordinal(ijet+1)}_{saveas}.pdf",format="pdf")
    
def selection_study(selection=None,title=None,saveas=None,plot=True,print_score=True,under6=False,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    if saveas:
        directory = f"plots/{date_tag}_plots/selection"
        if not os.path.isdir(directory): os.makedirs(directory)

    mask = selection.mask
    nevnts = ak.sum(mask)
    sixb_position = selection.sixb_position[mask]
    maxjets = ak.max(selection.njets_selected[mask])
    selection_purities = np.array(())
    selection_efficiencies = np.array(())
    min_ijet = 1 if under6 else 6
    
    ijet_cut = range(min_ijet,maxjets+1)
    
    for ijet in ijet_cut:
        minjet = min(ijet,6)
        atleast_ijet = selection.njets_selected[mask] >= minjet
        nsixb_at_ijet = ak.sum(sixb_position < ijet,axis=-1)[atleast_ijet]
        selection_purity = ak.sum(nsixb_at_ijet >= minjet)/ak.sum(atleast_ijet)
        selection_efficiency = ak.sum(atleast_ijet)/nevnts
        
        selection_purities = np.append(selection_purities,selection_purity)
        selection_efficiencies = np.append(selection_efficiencies,selection_efficiency)
    selection_scores = selection_purities * selection_efficiencies
    
    nrows,ncols = 1,2
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,5))
        
    graph_simple(ijet_cut,selection_purities,xlabel="N Jets Selected",ylabel="Purity",label="nSixb == min(6,nSelected) / Total Events",figax=(fig,axs[0]))
    graph_simple(ijet_cut,selection_efficiencies,xlabel="N Jets Selected",ylabel="Efficiency",label="nJets >= min(6,nSelected) / Total Events",figax=(fig,axs[1]))
#     graph_simple(ijet_cut,selection_scores,xlabel="N Jets Selected",ylabel="Purity * Efficiency",figax=(fig,axs[2]))
    
    fig.suptitle(f"{title}")
    fig.tight_layout()
    plt.show()
    if saveas: fig.savefig(f"{directory}/{saveas}.pdf",format="pdf")

def jet_order_study(selection=None,title=None,saveas=None,plot=True,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    varinfo = {
#         f"jet_m":{"bins":np.linspace(0,60,50),"xlabel":"Top Selected Jet Mass"},
#         f"jet_E":{"bins":np.linspace(0,500,50),"xlabel":"Top Selected Jet Energy"},
        f"jet_ptRegressed":{"bins":np.linspace(0,500,50),"xlabel":"Top Selected Jet Pt (GeV)"},
        f"jet_btag":{"bins":np.linspace(0,1,50),"xlabel":"Top Selected Jet Btag"},
        f"jet_eta":{"bins":np.linspace(-3,3,50),"xlabel":"Top Selected Jet Eta"},
        f"jet_phi":{"bins":np.linspace(-3.14,3.14,50),"xlabel":"Top Selected Jet Phi"},
    }
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    labels = ("Non Signal Jets Selected","Signal Jets Missed")
    colors = ("tab:orange","black")
    histtypes = ("bar","step")
    
    if saveas:
        directory = f"plots/{date_tag}_plots/order"
        if not os.path.isdir(directory): os.makedirs(directory)
    
    non_sixb_jets = exclude_jets(selection.jets_captured,selection.sixb_captured)
    sixb_ordered = ak.pad_none(selection.sixb_ordered,6,axis=-1)
    
    njets = min(6,selection.njets) if selection.njets != -1 else 6
    
    for ijet in range(njets):
        nsixb_mask = selection.nsixb_captured > ijet
        isixb_mask = get_jet_index_mask(branches,sixb_ordered[:,ijet][:,np.newaxis])
        
        if ak.sum(ak.flatten(isixb_mask[nsixb_mask])) == 0: 
            print(f"*** No incorrect selection in position {ijet} ***")
            continue
            
        nrows,ncols=1,4
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,5))
        for i,(var,info) in enumerate(varinfo.items()):
            ord_info = dict(info)
            ord_info["xlabel"] = f"{ordinal(ijet+1)} {info['xlabel']}"
            jets_var = ak.flatten(branches[var][non_sixb_jets][nsixb_mask])
            sixb_var = ak.flatten(branches[var][isixb_mask][nsixb_mask])
            plot_mask_comparison((jets_var,sixb_var),figax=(fig,axs[i]),**ord_info,
                                  labels=labels,colors=colors,histtypes=histtypes,density=1)
        fig.suptitle(f"{ordinal(ijet+1)} {title}")
        fig.tight_layout()
        plt.show()
        if saveas: fig.savefig(f"{directory}/{ordinal(ijet+1)}_{saveas}.pdf",format="pdf")
    
def x_reco_study(selection=None,title=None,saveas=None,plot=True,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    varinfo = {
        f"m":{"bins":np.linspace(0,1400,50),"xlabel":"X Reco Mass (GeV)"},
        f"pt":{"bins":np.linspace(0,500,50),"xlabel":"X Reco Pt (GeV)"},
        f"eta":{"bins":np.linspace(-3,3,50),"xlabel":"X Reco Eta"},
        f"phi":{"bins":np.linspace(-3.14,3.14,50),"xlabel":"X Reco Phi"},
    }
    
    mask = selection.mask
    sixb_real = (selection.nsixb_selected == 6)[mask]
    five_real = (selection.nsixb_selected <= 5)[mask]
    labels = ["All Events",f"True Reco X",f"6 Signal Selected",f"<6 Signal Selected"]
    colors = ["tab:blue","black","tab:orange","tab:red"]
    histtypes=["bar","step","step","step"]
    
    nrows,ncols = 2,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(15,9) )
    X_reco = selection.reco_X()
    
    for i,(var,info) in enumerate(varinfo.items()):
        X_tru = branches[f"X_{var}"][mask]
        X_var = X_reco[var][mask]
        sixb_real_X_var = X_var[sixb_real]
        five_real_X_var = X_var[five_real]
        datalist = [X_var,X_tru,sixb_real_X_var,five_real_X_var]
        plot_mask_comparison(datalist,labels=labels,histtypes=histtypes,colors=colors,figax=(fig,axs[int(i/ncols),i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: 
        directory = f"plots/{date_tag}_plots/x_reco"
        if not os.path.isdir(directory): os.makedirs(directory)
        fig.savefig(f"{directory}/{saveas}.pdf",format="pdf")

def x_res_study(selection=None,title=None,saveas=None,plot=True,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    varinfo = {
        f"m":{"bins":np.linspace(0,2,50),"xlabel":"X Reco Mass Resolution"},
        f"pt":{"bins":np.linspace(0,2,50),"xlabel":"X Reco Pt Resolution"},
        f"eta":{"bins":np.linspace(0,2,50),"xlabel":"X Reco Resolution"},
        f"phi":{"bins":np.linspace(0,2,50),"xlabel":"X Reco Resolution"},
    }
    
    mask = selection.mask
    sixb_real = (selection.nsixb_selected == 6)[mask]
    five_real = (selection.nsixb_selected == 5)[mask]
    fake = (selection.nsixb_selected < 5)[mask]
    labels = ["All Events",f"6 Signal Selected",f"<6 Signal Selected"]
    colors = ["tab:blue","tab:orange","tab:red"]
    histtypes=["bar","step","step"]
    
    nrows,ncols = 2,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(15,9) )
    X_reco = selection.reco_X()
    
    for i,(var,info) in enumerate(varinfo.items()):
        X_tru = branches[f"X_{var}"][mask]
        X_var = X_reco[var][mask]/X_tru
        sixb_real_X_var = X_var[sixb_real]
        five_real_X_var = X_var[five_real]
        datalist = [X_var,sixb_real_X_var,five_real_X_var]
        plot_mask_comparison(datalist,labels=labels,colors=colors,histtypes=histtypes,figax=(fig,axs[int(i/ncols),i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: 
        directory = f"plots/{date_tag}_plots/x_res"
        if not os.path.isdir(directory): os.makedirs(directory)
        fig.savefig(f"{directory}/{saveas}.pdf",format="pdf")

def njet_study(selection=None,title=None,saveas=None,plot=True,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    nsixb = min(6,selection.njets) if selection.njets != -1 else 6
    
    nrows,ncols = 1,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(15,5) )
    labels = ("All Events",)
    
    mask = selection.mask
    njets_captured = selection.njets_captured[mask]
    njets_selected = selection.njets_selected[mask]
    nsixb_selected = selection.nsixb_selected[mask]
    
    plot_mask_comparison([njets_captured],bins=range(13),xlabel="Number of Jets",labels=["All Events"],figax=(fig,axs[0]))
    plot_mask_comparison([nsixb_selected],bins=range(8),xlabel="Number of Signal Jets Selected",labels=["All Events"],figax=(fig,axs[1]))

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: 
        directory = f"plots/{date_tag}_plots/njet"
        if not os.path.isdir(directory): os.makedirs(directory)
        fig.savefig(f"{directory}/{saveas}.pdf",format="pdf")
    

def presel_study(selection=None,title=None,saveas=None,plot=True,missing=False,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    varinfo = {
        f"jet_m":{"bins":np.linspace(0,60,50),"xlabel":"Top Selected Jet Mass"},
        f"jet_E":{"bins":np.linspace(0,500,50),"xlabel":"Top Selected Jet Energy"},
        f"jet_ptRegressed":{"bins":np.linspace(0,200,50),"xlabel":"Top Selected Jet Pt (GeV)"},
        f"jet_btag":{"bins":np.linspace(0,1,50),"xlabel":"Top Selected Jet Btag"},
        f"jet_eta":{"bins":np.linspace(-3,3,50),"xlabel":"Top Selected Jet Eta"},
        f"jet_phi":{"bins":np.linspace(-3.14,3.14,50),"xlabel":"Top Selected Jet Phi"},
    }
    
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    nsixb = min(6,selection.njets) if selection.njets != -1 else 6
    mask = selection.mask
    signal_mask = mask & (selection.nsixb_selected == nsixb)

    nrows,ncols = 2,3
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(15,9) )
    for i,(var,info) in enumerate(varinfo.items()):
        jet_var = branches[var][selection.jets_selected]
        sixb_var = branches[var][selection.sixb_selected]
        all_data = ak.flatten(jet_var[mask])
        sixb_data = ak.flatten(sixb_var[mask])
        signal_data = ak.flatten(jet_var[signal_mask])
        datalist = (all_data,sixb_data)
        labels = ("All Selected Jets",f"Selected Signal Jets")
        plot_mask_comparison(datalist,labels=labels,figax=(fig,axs[int(i/ncols),i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: 
        directory = f"plots/{date_tag}_plots/presel"
        if not os.path.isdir(directory): os.makedirs(directory)
        fig.savefig(f"{directory}/{saveas}.pdf",format="pdf")
    
def jet_issue_study(selection=None,title=None,saveas=None,plot=True,missing=False,print_score=True,**kwargs):
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    
    mask = selection.mask
    jets = selection.jets_selected
    
    ptbins = np.linspace(0,150,50)
    etabins = np.linspace(-3,3,50)
    
    jet_pt = ak.flatten(branches["jet_ptRegressed"][jets][mask])
    jet_eta = ak.flatten(branches["jet_eta"][jets][mask])
    eta24 = np.abs(jet_eta) < 2.4
    
    
    nrows,ncols = 2,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(15,12) )
    
    plot_mask_comparison([jet_pt],bins=ptbins,xlabel="jet pt",figax=(fig,axs[0,0]))
    plot_mask_comparison([jet_eta],bins=etabins,xlabel="jet eta",figax=(fig,axs[0,1]))
    
    plot_mask_simple_2d_comparison(jet_pt,jet_eta,xbins=ptbins,ybins=etabins,xlabel="jet pt",ylabel="jet eta",figax=(fig,axs[1,0]))
    plot_mask_simple_2d_comparison(jet_pt[eta24],jet_eta[eta24],xbins=ptbins,ybins=etabins,xlabel="jet pt",ylabel="jet eta",title="|jet eta|<2.4",figax=(fig,axs[1,1]))
    return fig,axs
    