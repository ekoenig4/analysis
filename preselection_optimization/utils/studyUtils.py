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
    
def save_fig(fig,directory,saveas):
    directory = f"plots/{date_tag}_plots/{directory}"
    if not os.path.isdir(directory): os.makedirs(directory)
    fig.savefig(f"{saveas}.pdf",format="pdf")

def signal_order_study(selection=None,title=None,saveas=None,plot=True,print_score=True,subset="selected",**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
        
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
            
        nrows,ncols=1,3
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
        for i,(var,info) in enumerate(varinfo.items()):
            ord_info = dict(info)
            ord_info["xlabel"] = f"{ordinal(ijet+1)} {info['xlabel']}"
            sixb_var = branches[var][isixb_mask][nsixb_mask]
            sixb_data = ak.flatten(sixb_var)
                
            if var == "jet_ptRegressed": ptinfo = ord_info
            if var == "jet_eta": etainfo = ord_info
            if var == "jet_btag": btaginfo = ord_info
            
            datalist = (sixb_data,)
            plot_mask_comparison(datalist,labels=labels,figax=(fig,axs[i%ncols]),**ord_info)
            
#         sixb_ptdata = ak.flatten( branches["jet_ptRegressed"][isixb_mask][nsixb_mask] )
#         sixb_etadata = ak.flatten( branches["jet_eta"][isixb_mask][nsixb_mask] )
#         sixb_btagdata = ak.flatten( branches["jet_btag"][isixb_mask][nsixb_mask] )
        
        
#         plot_mask_simple_2d_comparison(sixb_etadata,sixb_btagdata,xbins=etainfo['bins'],ybins=btaginfo['bins'],
#                                        xlabel=etainfo['xlabel'],ylabel=btaginfo['xlabel'],figax=(fig,axs[1,0]),log=1)
        
#         plot_mask_simple_2d_comparison(sixb_ptdata,sixb_etadata,xbins=ptinfo['bins'],ybins=etainfo['bins'],
#                                        xlabel=ptinfo['xlabel'],ylabel=etainfo['xlabel'],figax=(fig,axs[1,1]),log=1)
        
#         plot_mask_simple_2d_comparison(sixb_ptdata,sixb_btagdata,xbins=ptinfo['bins'],ybins=btaginfo['bins'],
#                                        xlabel=ptinfo['xlabel'],ylabel=btaginfo['xlabel'],figax=(fig,axs[1,2]),log=1)
            
        fig.suptitle(f"{title} {ordinal(ijet+1)} {selection.variable}")
        fig.tight_layout()
        plt.show()
        if saveas: save_fig(fig,"order",f"{ordinal(ijet+1)}_{saveas}")
    
def selection_study(selection=None,title=None,saveas=None,plot=True,print_score=True,subset="selected",under6=False,latex=False,required=False,**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
        
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return

    mask = selection.mask
    nevnts = ak.sum(mask)
    sixb_position = selection.sixb_selected_position[mask]
    maxjets = ak.max(selection.njets_selected[mask])
    selection_purities = np.array(())
    selection_efficiencies = np.array(())
    min_ijet = 1 if under6 else 6
    
    ijet_cut = range(min_ijet,maxjets+1)
    
    for ijet in ijet_cut:
        minjet = min(ijet,6)
        atleast_ijet = selection.njets_selected[mask] >= minjet
        nsixb_at_ijet = ak.sum(sixb_position < ijet,axis=-1)
        nevnts_ijet = ak.sum(atleast_ijet) if required else nevnts
        selection_purity = ak.sum(nsixb_at_ijet >= minjet)/nevnts_ijet
        selection_efficiency = nevnts_ijet/nevnts
        
        selection_purities = np.append(selection_purities,selection_purity)
        selection_efficiencies = np.append(selection_efficiencies,selection_efficiency)
    selection_scores = selection_purities * selection_efficiencies
    
    # Print out for latex table
    if latex:
        print(" & ".join(f"{purity:.2f}" for purity in selection_purities[:4]))
    
    nrows,ncols = 1,(2 if under6 else 1)
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=( 16 if under6 else 8,5))
        
    ax0 = axs[0] if under6 else axs
    ax1 = axs[1] if under6 else None
        
    graph_simple(ijet_cut,selection_purities,xlabel="N Jets Selected",ylabel="Purity",label="nSixb == min(6,nSelected) / Total Events",figax=(fig,ax0))
    
    if under6:
        graph_simple(ijet_cut,selection_efficiencies,xlabel="N Jets Selected",ylabel="Efficiency",label="nJets >= min(6,nSelected) / Total Events",figax=(fig,ax1))
#     graph_simple(ijet_cut,selection_scores,xlabel="N Jets Selected",ylabel="Purity * Efficiency",figax=(fig,axs[2]))
    
    fig.suptitle(f"{title}")
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,"selection",saveas)
        
def selection_comparison_study(selections=[],labels=[],saveas=None,plot=True,print_score=True,subset="selected",under6=False,latex=False,required=False,**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
        
    if not plot: return

    selection_purities = []
    
    for selection in selections:
        mask = selection.mask
        nevnts = ak.sum(mask)
        sixb_position = selection.sixb_selected_position[mask]
        maxjets = ak.max(selection.njets_selected[mask])
        min_ijet = 1 if under6 else 6
        ijet_cut = range(min_ijet,11)
        
        purities = np.array(())
        for ijet in ijet_cut:
            minjet = min(ijet,6)
            atleast_ijet = selection.njets_selected[mask] >= minjet
            nsixb_at_ijet = ak.sum(sixb_position < ijet,axis=-1)
            nevnts_ijet = ak.sum(atleast_ijet) if required else nevnts
            purity = ak.sum(nsixb_at_ijet >= minjet)/nevnts_ijet

            purities = np.append(purities,purity)
            
        selection_purities.append(purities)
    
    nrows,ncols = 1,1
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=( 16 if under6 else 8,5))
        
    graph_multi(ijet_cut,selection_purities,xlabel="N Jets Selected",ylabel="Purity",labels=labels,figax=(fig,axs))
    
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,"selection",saveas)

def jets_study(selection=None,title=None,saveas=None,plot=True,print_score=True,subset="selected",density=0,**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
    
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    varinfo = {
#         f"jet_m":{"bins":np.linspace(0,60,50),"xlabel":"Jet Mass"},
#         f"jet_E":{"bins":np.linspace(0,500,50),"xlabel":"Jet Energy"},
        f"jet_ptRegressed":{"bins":None,"xlabel":"Jet Pt (GeV)"},
        f"jet_btag":{"bins":np.linspace(0,1,50),"xlabel":"Jet Btag"},
        f"jet_qgl":{"bins":np.linspace(0,1,50),"xlabel":"Jet Quark Gluon Disciminant"},
        f"jet_eta":{"bins":np.linspace(-3,3,50),"xlabel":"Jet Eta"},
#         f"jet_phi":{"bins":np.linspace(-3.14,3.14,50),"xlabel":"Jet Phi"},
    }
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    labels = (f"Non Signal Jets {subset.capitalize()}",f"Signal Jets {subset.capitalize()}")
    colors = ("tab:orange","black")
    histtypes = ("bar","step")
    
    mask = selection.mask
    non_sixb_jets = exclude_jets( getattr(selection,f"jets_{subset}"),getattr(selection,f"sixb_{subset}"))
    sixb_ordered = getattr(selection,f"sixb_{subset}")
            
    nrows,ncols=1,4
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
    for i,(var,info) in enumerate(varinfo.items()):
        jets_var = ak.flatten(branches[var][non_sixb_jets][mask])
        sixb_var = ak.flatten(branches[var][sixb_ordered][mask])
        plot_mask_comparison((jets_var,sixb_var),figax=(fig,axs[i]),**info,
                              labels=labels,colors=colors,histtypes=histtypes,density=density)
            
    fig.suptitle(f"{title}")
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,f"jets_{subset}",saveas)

def jets_ordered_study(selection=None,title=None,saveas=None,plot=True,print_score=True,subset="selected",density=0,**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
        
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    varinfo = {
#         f"jet_m":{"bins":np.linspace(0,60,50),"xlabel":"Jet Mass"},
#         f"jet_E":{"bins":np.linspace(0,500,50),"xlabel":"Jet Energy"},
        f"jet_ptRegressed":{"bins":None,"xlabel":"Jet Pt (GeV)"},
        f"jet_btag":{"bins":np.linspace(0,1,50),"xlabel":"Jet Btag"},
        f"jet_qgl":{"bins":np.linspace(0,1,50),"xlabel":"Jet Quark Gluon Disciminant"},
        f"jet_eta":{"bins":np.linspace(-3,3,50),"xlabel":"Jet Eta"},
#         f"jet_phi":{"bins":np.linspace(-3.14,3.14,50),"xlabel":"Jet Phi"},
    }
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    colors = ("tab:orange","black")
    histtypes = ("bar","step")
    
    mask = selection.mask
    jets = getattr(selection,f"jets_{subset}")
    sixb = getattr(selection,f"sixb_{subset}")
    nsixb= getattr(selection,f"nsixb_{subset}")
    sixb_ordered = getattr(selection,f"sixb_{subset}_index")
    
    non_sixb_jets = exclude_jets(jets,sixb)
    sixb_ordered = ak.pad_none(sixb_ordered,6,axis=-1)
    
    njets = min(6,selection.njets) if selection.njets != -1 else 6
    
    for ijet in range(njets):
        labels = (f"Non Signal Jets {subset.capitalize()}",f"{ordinal(ijet+1)} Signal Jets {subset.capitalize()}")
        nsixb_mask = mask & (nsixb > ijet)
        isixb_mask = get_jet_index_mask(branches,sixb_ordered[:,ijet][:,np.newaxis])
        
        if ak.sum(ak.flatten(isixb_mask[nsixb_mask])) == 0: 
            print(f"*** No incorrect selection in position {ijet} ***")
            continue
            
        nrows,ncols=1,4
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
        for i,(var,info) in enumerate(varinfo.items()):
            jets_var = ak.flatten(branches[var][non_sixb_jets])
            sixb_var = ak.flatten(branches[var][isixb_mask][nsixb_mask])
            plot_mask_comparison((jets_var,sixb_var),figax=(fig,axs[i]),**info,
                                  labels=labels,colors=colors,histtypes=histtypes,density=density)
            
        fig.suptitle(f"{title} {ordinal(ijet+1)} Signal Jet")
        fig.tight_layout()
        plt.show()
        if saveas: save_fig(fig,f"jets_{subset}",f"{ordinal(ijet+1)}_{saveas}")
    
def x_reco_study(selection=None,title=None,saveas=None,plot=True,print_score=True,**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
        
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
    five_real = (selection.nsixb_selected == 5)[mask]
    less_real = (selection.nsixb_selected <  5)[mask]
    labels = ["All Events",f"Signal Selected",f"1 Wrong Jet Selected",f"More Wrong Jets Selected"]
    colors = ["tab:blue","tab:orange","tab:red","tab:green"]
    histtypes=["bar","step","step","step"]
    
    nrows,ncols = 2,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,10) )
    X_reco = selection.reco_X()
    
    for i,(var,info) in enumerate(varinfo.items()):
        X_var = X_reco[var][mask]
        sixb_real_X_var = X_var[sixb_real]
        five_real_X_var = X_var[five_real]
        less_real_X_var = X_var[less_real]
        datalist = [X_var,sixb_real_X_var,five_real_X_var,less_real_X_var]
        plot_mask_comparison(datalist,labels=labels,histtypes=histtypes,colors=colors,figax=(fig,axs[int(i/ncols),i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,"x_reco",saveas)

def x_res_study(selection=None,title=None,saveas=None,plot=True,print_score=True,subset="selected",**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
        
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
    labels = ["All Events",f"Signal Selected",f"Background Selected"]
    colors = ["tab:blue","tab:orange","tab:red"]
    histtypes=["bar","step","step"]
    
    nrows,ncols = 2,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,10) )
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
    if saveas: save_fig(fig,"x_res",saveas)

def njet_study(selection=None,title=None,saveas=None,plot=True,print_score=True,subset="selected",**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
        
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    nsixb = min(6,selection.njets) if selection.njets != -1 else 6
    
    nrows,ncols = 1,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,5) )
    labels = ("All Events",)
    
    mask = selection.mask
    njets = getattr(selection,f"njets_{subset}")[mask]
    nsixb = getattr(selection,f"nsixb_{subset}")[mask]
    
    plot_mask_comparison([njets],bins=range(12),xlabel=f"Number of Jets {subset.capitalize()}",labels=["All Events"],figax=(fig,axs[0]))
    plot_mask_comparison([nsixb],bins=range(8),xlabel=f"Number of Signal Jets {subset.capitalize()}",labels=["All Events"],figax=(fig,axs[1]))

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,f"njets_{subset}",saveas)
    

def presel_study(selection=None,title=None,saveas=None,plot=True,missing=False,print_score=True,subset="selected",**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
        
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    varinfo = {
#         f"jet_m":{"bins":np.linspace(0,60,50),"xlabel":"Top Selected Jet Mass"},
#         f"jet_E":{"bins":np.linspace(0,500,50),"xlabel":"Top Selected Jet Energy"},
        f"jet_ptRegressed":{"bins":np.linspace(0,500,50),"xlabel":"Jet Pt (GeV)"},
        f"jet_btag":{"bins":np.linspace(0,1,50),"xlabel":"Jet Btag"},
        f"jet_eta":{"bins":np.linspace(-3,3,50),"xlabel":"Jet Eta"},
        f"jet_phi":{"bins":np.linspace(-3.14,3.14,50),"xlabel":"Jet Phi"},
    }
    
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    nsixb = min(6,selection.njets) if selection.njets != -1 else 6
    mask = selection.mask
    
    jets = getattr(selection,f"jets_{subset}")
    sixb = getattr(selection,f"sixb_{subset}")
    
    signal_mask = mask & (getattr(selection,f"nsixb_{subset}") == nsixb)

    nrows,ncols = 1,4
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,5) )
    for i,(var,info) in enumerate(varinfo.items()):
        jet_var = branches[var][jets]
        sixb_var = branches[var][sixb]
        all_data = ak.flatten(jet_var[mask])
        sixb_data = ak.flatten(sixb_var[mask])
        signal_data = ak.flatten(jet_var[signal_mask])
        datalist = (all_data,signal_data)
        labels = (f"All Jets {subset.capitalize()}",f"Signal Jets {subset.capitalize()}")
        plot_mask_comparison(datalist,labels=labels,figax=(fig,axs[i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,f"presel_{subset}",saveas)
    
def jet_issue_study(selection=None,title=None,saveas=None,plot=True,missing=False,print_score=True,subset="selected",**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
        
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
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,10) )
    
    plot_mask_comparison([jet_pt],bins=ptbins,xlabel="jet pt",figax=(fig,axs[0,0]))
    plot_mask_comparison([jet_eta],bins=etabins,xlabel="jet eta",figax=(fig,axs[0,1]))
    
    plot_mask_simple_2d_comparison(jet_pt,jet_eta,xbins=ptbins,ybins=etabins,xlabel="jet pt",ylabel="jet eta",figax=(fig,axs[1,0]))
    plot_mask_simple_2d_comparison(jet_pt[eta24],jet_eta[eta24],xbins=ptbins,ybins=etabins,xlabel="jet pt",ylabel="jet eta",title="|jet eta|<2.4",figax=(fig,axs[1,1]))
    return fig,axs
    
def signal_b_study(selection=None,title=None,saveas=None,plot=True,print_score=True,subset="selected",**kwargs):
    if subset not in ["selected","captured","remaining"]: raise ValueError(f"{subset} not available")
        
    branches = selection.branches
    if title is None: title = selection.title()
    print(f"--- {title} ---")
    
    score = selection.score()
    if print_score: print(score)
    if saveas: save_scores(score,saveas)
    
    if not plot: return
    
    mask = selection.mask
    sixb_selected = getattr(selection,f"sixb_{subset}")[mask]
    signal_tags = ["HX_b1","HX_b2","HY1_b1","HY1_b2","HY2_b1","HY2_b2"]
    signal_index = { tag: get_jet_index_mask(branches,branches[f"gen_{tag}_recojet_index"][:,np.newaxis])[mask] for tag in signal_tags }
    signal_b_selected = { tag:sixb_selected & b_index for tag,b_index in signal_index.items() }
    
    varinfo = {
#         f"jet_m":{"bins":np.linspace(0,60,50),"xlabel":"Top Selected Jet Mass"},
#         f"jet_E":{"bins":np.linspace(0,500,50),"xlabel":"Top Selected Jet Energy"},
        f"jet_ptRegressed":{"bins":np.linspace(0,500,50),"xlabel":"Jet Pt (GeV)"},
        f"jet_btag":{"bins":np.linspace(0,1,50),"xlabel":"Jet Btag"},
        f"jet_eta":{"bins":np.linspace(-3,3,50),"xlabel":"Jet Eta"},
        f"jet_phi":{"bins":np.linspace(-3.14,3.14,50),"xlabel":"Jet Phi"},
    }
    
    nrows,ncols = 1,1
    fig0,ax0 = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,2.5))
    nsignal_b_selected = np.array([ak.sum(ak.sum(signal_mask,axis=-1)) for signal_mask in signal_b_selected.values() ])
    graph_simple(signal_tags,nsignal_b_selected,ylabel="Number Jets",figax=(fig0,ax0))
    
    for i,(var,info) in enumerate(varinfo.items()):
        
        nrows,ncols = 1,6
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,5),sharey=True )
        for i,(tag,b_selected) in enumerate(signal_b_selected.items()):
            b_info = dict(info)
            b_info["xlabel"] = f"{tag} {info['xlabel']}"
            b_info["labels"] = [tag]
            if i != 0: b_info["ylabel"] = ""
            b_var = ak.flatten(branches[var][mask][b_selected])
            plot_mask_comparison([b_var],**b_info,figax=(fig,axs[i]))
