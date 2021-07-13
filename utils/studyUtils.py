#!/usr/bin/env python
# coding: utf-8

from . import *

from datetime import date
import os

    
varinfo = {
    f"jet_m":     {"bins":np.linspace(0,60,50)      ,"xlabel":"Jet Mass"},
    f"jet_E":     {"bins":np.linspace(0,300,50)     ,"xlabel":"Jet Energy"},
    f"jet_pt":    {"bins":np.linspace(0,300,50)     ,"xlabel":"Jet Pt (GeV)"},
    f"jet_btag":  {"bins":np.linspace(0,1,50)       ,"xlabel":"Jet Btag"},
    f"jet_qgl":   {"bins":np.linspace(0,1,50)       ,"xlabel":"Jet QGL"},
    f"jet_min_dr":{"bins":np.linspace(0,3,50)       ,"xlabel":"Jet Min dR"},
    f"jet_eta":   {"bins":np.linspace(-3,3,50)      ,"xlabel":"Jet Eta"},
    f"jet_phi":   {"bins":np.linspace(-3.14,3.14,50),"xlabel":"Jet Phi"},
}

date_tag = date.today().strftime("%Y%m%d")

def save_scores(score,saveas):
    directory = f"plots/{date_tag}_plots/scores"
    if not os.path.isdir(directory): os.makedirs(directory)
    score.savetex(f"{directory}/{saveas}")
    
def save_fig(fig,directory,saveas):
    directory = f"plots/{date_tag}_plots/{directory}"
    if not os.path.isdir(directory): os.makedirs(directory)
    fig.savefig(f"{directory}/{saveas}.pdf",format="pdf")
    
class Study:
    def __init__(self,selection,title=None,saveas=None,print_score=True,subset="selected",mask=None,varlist=["jet_pt","jet_btag","jet_qgl","jet_eta"],autobin=False,**kwargs):
        if subset not in ["selected","passed","remaining","failed"]: raise ValueError(f"{subset} not available")
        if mask is not None: selection = selection.masked(mask)
        self.selection = selection
        self.scale = selection.nevents*selection.scale
        self.subset = subset
        self.saveas = saveas
        self.varinfo = { var:dict(**varinfo[var]) for var in varlist }
        
        if autobin: 
            for var in self.varinfo.values(): var["bins"] = None
        
        if title is None: title = selection.title()
        self.title = title
        print(f"--- {title} ---")
        
        score = selection.score()
        if print_score: print(score)
        if saveas: save_scores(score,saveas)

def signal_order_study(selection,plot=True,saveas=None,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
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
                
            if var == "jet_pt": ptinfo = ord_info
            if var == "jet_eta": etainfo = ord_info
            if var == "jet_btag": btaginfo = ord_info
            
            datalist = (sixb_data,)
            hist_multi(datalist,labels=labels,figax=(fig,axs[i%ncols]),**ord_info)
            
#         sixb_ptdata = ak.flatten( branches["jet_pt"][isixb_mask][nsixb_mask] )
#         sixb_etadata = ak.flatten( branches["jet_eta"][isixb_mask][nsixb_mask] )
#         sixb_btagdata = ak.flatten( branches["jet_btag"][isixb_mask][nsixb_mask] )
        
        
#         hist2d_simple(sixb_etadata,sixb_btagdata,xbins=etainfo['bins'],ybins=btaginfo['bins'],
#                                        xlabel=etainfo['xlabel'],ylabel=btaginfo['xlabel'],figax=(fig,axs[1,0]),log=1)
        
#         hist2d_simple(sixb_ptdata,sixb_etadata,xbins=ptinfo['bins'],ybins=etainfo['bins'],
#                                        xlabel=ptinfo['xlabel'],ylabel=etainfo['xlabel'],figax=(fig,axs[1,1]),log=1)
        
#         hist2d_simple(sixb_ptdata,sixb_btagdata,xbins=ptinfo['bins'],ybins=btaginfo['bins'],
#                                        xlabel=ptinfo['xlabel'],ylabel=btaginfo['xlabel'],figax=(fig,axs[1,2]),log=1)
            
        fig.suptitle(f"{title} {ordinal(ijet+1)} {selection.variable}")
        fig.tight_layout()
        plt.show()
        if saveas: save_fig(fig,"order",f"{ordinal(ijet+1)}_{saveas}")
    
def selection_study(selection,plot=True,saveas=None,under6=False,latex=False,required=False,scaled=False,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
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
        
        if scaled: selection_purity *= minjet/6
        
        selection_purities = np.append(selection_purities,selection_purity)
        selection_efficiencies = np.append(selection_efficiencies,selection_efficiency)
    selection_scores = selection_purities * selection_efficiencies
    
    # Print out for latex table
    if latex:
        print(" & ".join(f"{ijet:<4}" for ijet in ijet_cut))
        print(" & ".join(f"{purity:.2f}" for purity in selection_purities))
    
    extra = under6 and required
    nrows,ncols = 1,(2 if extra else 1)
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=( 16 if extra else 8,5))
        
    ax0 = axs[0] if extra else axs
    ax1 = axs[1] if extra else None
        
    ylabel = "Purity" if not scaled else "Scaled Purity"
    graph_simple(ijet_cut,selection_purities,xlabel="N Jets Selected",ylabel=ylabel,label="nSixb == min(6,nSelected) / Total Events",figax=(fig,ax0))
    
    if extra and required:
        graph_simple(ijet_cut,selection_efficiencies,xlabel="N Jets Selected",ylabel="Efficiency",label="nJets >= min(6,nSelected) / Total Events",figax=(fig,ax1))
#     graph_simple(ijet_cut,selection_scores,xlabel="N Jets Selected",ylabel="Purity * Efficiency",figax=(fig,axs[2]))
    
    fig.suptitle(f"{title}")
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,"selection",saveas)
        
def selection_comparison_study(selections,plot=True,saveas=None,under6=False,latex=False,required=False,title=None,labels=None,**kwargs):
    if not plot: return

    selection_purities = []
    if labels is None: labels = [ selection.tag for selection in selections ]
    
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

def jets_study(selection,plot=True,saveas=None,density=0,log=0,scaled=False,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    scale = study.scale if scaled else 1
    
    if not plot: return
    
    labels = (f"Background Jets",f"Signal Jets")
    colors = ("tab:orange","black")
    histtypes = ("bar","step")
    
    mask = selection.mask
    bkgs_jets = getattr(selection,f"bkgs_{subset}")
    sixb_ordered = getattr(selection,f"sixb_{subset}")
            
    nrows,ncols=1,4
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
    for i,(var,info) in enumerate(varinfo.items()):
        bkgs_var = ak.flatten(branches[var][bkgs_jets][mask])
        sixb_var = ak.flatten(branches[var][sixb_ordered][mask])
        hist_multi((bkgs_var,sixb_var),figax=(fig,axs[i]),**info,scale=scale,
                   labels=labels,colors=colors,histtypes=histtypes,density=density,log=log)
            
    fig.suptitle(f"{title}")
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,f"jets_{subset}",saveas)
        

def jets_2d_study(selection,plot=True,saveas=None,density=0,log=1,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    scale = study.scale if scaled else 1
    
    if not plot: return
    
    labels = (f"Background Jets",f"Signal Jets")
    colors = ("tab:orange","black")
    histtypes = ("bar","step")
    
    mask = selection.mask
    bkgs_jets = getattr(selection,f"bkgs_{subset}")
    sixb_ordered = getattr(selection,f"sixb_{subset}")
    
    plot2d = hist2d_simple
    ptinfo = varinfo["jet_pt"]
    btaginfo = varinfo["jet_btag"]
            
    nrows,ncols=1,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
    
    jets_btag = ak.flatten(branches["jet_btag"][bkgs_jets][mask])
    jets_pt = ak.flatten(branches["jet_pt"][bkgs_jets][mask])

    sixb_btag = ak.flatten(branches["jet_btag"][sixb_ordered][mask])
    sixb_pt =     ak.flatten(branches["jet_pt"][sixb_ordered][mask])

    plot2d(jets_pt,jets_btag,ybins=btaginfo["bins"],xbins=ptinfo["bins"],ylabel=btaginfo["xlabel"],xlabel=ptinfo["xlabel"],title=labels[0],density=density,log=log,figax=(fig,axs[0]))
    plot2d(sixb_pt,sixb_btag,ybins=btaginfo["bins"],xbins=ptinfo["bins"],ylabel=btaginfo["xlabel"],xlabel=ptinfo["xlabel"],title=labels[1],density=density,log=log,figax=(fig,axs[1]))
            
    fig.suptitle(f"{title}")
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,f"jets_2d_{subset}",saveas)
        
def ijets_study(selection,plot=True,saveas=None,njets=-1,show_ijet=None,topbkg=False,density=0,log=0,scaled=0,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    scale = study.scale if scaled else 1
    
    if not plot: return
    
    colors = ("tab:orange","black")
    histtypes = ("bar","step")
    
    mask = selection.mask
    bkgs_mask = getattr(selection,f"bkgs_{subset}")
    sixb_mask = getattr(selection,f"sixb_{subset}")
    
    jets = getattr(selection,f"jets_{subset}")
    
    maxjets = ak.max(selection.njets_selected)
    if njets == -1: njets = maxjets
    else: njets = min(njets,selection.njets)
    
    jets_ordered = ak.pad_none(getattr(selection,f"jets_{subset}_index"),njets)
    
    for ijet in range(njets):
        if show_ijet and ijet not in show_ijet: continue
        labels = (f"Background Jet",f"Signal Jet")
        
        ijet_mask = get_jet_index_mask(branches,jets_ordered[:,ijet][:,np.newaxis])
        isixb_mask = ijet_mask & sixb_mask
        ibkgs_mask = ijet_mask & bkgs_mask
            
        nrows,ncols=1,4
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
        for i,(var,info) in enumerate(varinfo.items()):
            ord_info = dict(**info)
            ord_info["xlabel"] = f"{ordinal(ijet+1)} {info['xlabel']}"
            bkgs_var = ak.flatten(branches[var][ibkgs_mask][mask])
            sixb_var = ak.flatten(branches[var][isixb_mask][mask])
            hist_multi((bkgs_var,sixb_var),figax=(fig,axs[i]),**ord_info,scale=scale,
                       labels=labels,colors=colors,histtypes=histtypes,density=density,log=log)
            
        fig.suptitle(f"{title} {ordinal(ijet+1)} Jet")
        fig.tight_layout()
        plt.show()
        if saveas: save_fig(fig,f"ijets_{subset}",f"{ordinal(ijet+1)}_{saveas}")

def jets_ordered_study(selection,plot=True,saveas=None,njets=6,topbkg=True,density=0,log=0,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
    if not plot: return
    
    label1 = "Top Background Jet" if topbkg else "Background Jets"
    colors = ["tab:orange","black"]
    histtypes = ["bar","step"]
    
    mask = selection.mask
    
    bkgs = getattr(selection,f"bkgs_{subset}")
    nbkgs= getattr(selection,f"nbkgs_{subset}")
    bkgs_position = getattr(selection,f"bkgs_{subset}_position")
    bkgs_ordered = getattr(selection,f"bkgs_{subset}_index")
    
    sixb = getattr(selection,f"sixb_{subset}")
    nsixb= getattr(selection,f"nsixb_{subset}")
    sixb_position = getattr(selection,f"sixb_{subset}_position")
    sixb_ordered = getattr(selection,f"sixb_{subset}_index")
    
    
    if topbkg: bkgs_ordered = ak.pad_none(bkgs_ordered,1,axis=-1,clip=1)
    
    bkgs_mask = get_jet_index_mask(branches,bkgs_ordered)
    sixb_ordered = ak.pad_none(sixb_ordered,6,axis=-1)
    
    njets = min(njets,selection.njets if selection.njets != -1 else 6) 
    
    for ijet in range(njets):
        labels = [label1,f"{ordinal(ijet+1)} Signal Jet"]
        nsixb_mask = mask & (nsixb > ijet)
        isixb_mask = get_jet_index_mask(branches,sixb_ordered[:,ijet][:,np.newaxis])
        
        if ak.sum(ak.flatten(isixb_mask[nsixb_mask])) == 0: 
            print(f"*** No incorrect selection in position {ijet} ***")
            continue
            
        nrows,ncols=1,4
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
        for i,(var,info) in enumerate(varinfo.items()):
            jets_var = ak.flatten(branches[var][bkgs_mask][mask])
            sixb_var = ak.flatten(branches[var][isixb_mask][nsixb_mask])
            hist_multi((jets_var,sixb_var),figax=(fig,axs[i]),**info,
                                  labels=labels,colors=colors,histtypes=histtypes,density=density,log=log)
            
        fig.suptitle(f"{title} {ordinal(ijet+1)} Signal Jet")
        fig.tight_layout()
        plt.show()
        if saveas: save_fig(fig,f"jets_{subset}",f"{ordinal(ijet+1)}_{saveas}")
            
def jets_2d_ordered_study(selection,plot=True,saveas=None,njets=6,topbkg=True,compare=False,density=0,log=1,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
    if not plot: return
    
    label1 = "Top Background Jet" if topbkg else "Background Jets"
    colors = ("tab:orange","black")
    histtypes = ("bar","step")
    
    mask = selection.mask
    
    bkgs = getattr(selection,f"bkgs_{subset}")
    nbkgs= getattr(selection,f"nbkgs_{subset}")
    bkgs_position = getattr(selection,f"bkgs_{subset}_position")
    bkgs_ordered = getattr(selection,f"bkgs_{subset}_index")
    
    sixb = getattr(selection,f"sixb_{subset}")
    nsixb= getattr(selection,f"nsixb_{subset}")
    sixb_position = getattr(selection,f"sixb_{subset}_position")
    sixb_ordered = getattr(selection,f"sixb_{subset}_index")
    
    
    if topbkg: bkgs_ordered = ak.pad_none(bkgs_ordered,1,axis=-1,clip=1)
    
    bkgs_mask = get_jet_index_mask(branches,bkgs_ordered)
    sixb_ordered = ak.pad_none(sixb_ordered,6,axis=-1)
    
    njets = min(njets,selection.njets if selection.njets != -1 else 6) 
    
    plot2d = hist2d_simple
    ptinfo = varinfo["jet_pt"]
    btaginfo = varinfo["jet_btag"]
    
    for ijet in range(njets):
        labels = (label1,f"{ordinal(ijet+1)} Signal Jet")
        nsixb_mask = mask & (nsixb > ijet)
        isixb_mask = get_jet_index_mask(branches,sixb_ordered[:,ijet][:,np.newaxis])
        
        if ak.sum(ak.flatten(isixb_mask[nsixb_mask])) == 0: 
            print(f"*** No incorrect selection in position {ijet} ***")
            continue
            
        nrows,ncols=1,2
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
        
        if not (compare and topbkg):
            jets_btag = ak.flatten(branches["jet_btag"][bkgs_mask][mask])
            jets_pt = ak.flatten(branches["jet_pt"][bkgs_mask][mask])

            sixb_btag = ak.flatten(branches["jet_btag"][isixb_mask][nsixb_mask])
            sixb_pt =     ak.flatten(branches["jet_pt"][isixb_mask][nsixb_mask])

            plot2d(jets_pt,jets_btag,ybins=btaginfo["bins"],xbins=ptinfo["bins"],ylabel=btaginfo["xlabel"],xlabel=ptinfo["xlabel"],
                   title=labels[0],density=density,log=log,figax=(fig,axs[0]))
            plot2d(sixb_pt,sixb_btag,ybins=btaginfo["bins"],xbins=ptinfo["bins"],ylabel=btaginfo["xlabel"],xlabel=ptinfo["xlabel"],
                   title=labels[1],density=density,log=log,figax=(fig,axs[1]))
        else:
            compared_jets = nsixb_mask &  (selection.nbkgs_selected > 0)
            njets_compared = ak.sum(compared_jets)

            jets_btag = ak.flatten(branches["jet_btag"][bkgs_mask][compared_jets])
            jets_pt = ak.flatten(branches["jet_pt"][bkgs_mask][compared_jets])

            sixb_btag = ak.flatten(branches["jet_btag"][isixb_mask][compared_jets])
            sixb_pt =     ak.flatten(branches["jet_pt"][isixb_mask][compared_jets])
            
            pt_bias = ak.sum(sixb_pt > jets_pt)/njets_compared
            btag_bias = ak.sum(sixb_btag > jets_btag)/njets_compared
            
            plot2d(sixb_pt,jets_pt,xbins=ptinfo["bins"],ybins=ptinfo["bins"],
                   xlabel=f"{ordinal(ijet+1)} Signal {ptinfo['xlabel']}",ylabel=f"Top Background {ptinfo['xlabel']}",
                   title=f"Comparison Signal Bias: {pt_bias:0.2f}",density=density,log=log,figax=(fig,axs[0]))
            plot2d(sixb_btag,jets_btag,xbins=btaginfo["bins"],ybins=btaginfo["bins"],
                   xlabel=f"{ordinal(ijet+1)} Signal {btaginfo['xlabel']}",ylabel=f"Top Background {btaginfo['xlabel']}",
                   title=f"Comparison Signal Bias: {btag_bias:0.2f}",density=density,log=log,figax=(fig,axs[1]))
            
        fig.suptitle(f"{title} {ordinal(ijet+1)} Signal Jet")
        fig.tight_layout()
        plt.show()
        if saveas: save_fig(fig,f"jets_2d_{subset}",f"{ordinal(ijet+1)}_{saveas}")
    
def x_reco_study(selection,plot=True,saveas=None,scaled=False,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    scale = study.scale if scaled else 1
    
    if not plot: return
    
    varinfo = {
        f"m":{"bins":np.linspace(0,1400,50),  "xlabel":"X Reco Mass"},
        f"pt":{"bins":np.linspace(0,1000,50), "xlabel":"X Reco Pt"},
        f"eta":{"bins":np.linspace(-3,3,50),"xlabel":"X Reco Eta"},
        f"phi":{"bins":np.linspace(-3.14,3.14,50),"xlabel":"X Reco Phi"},
    }
    
    mask = selection.mask
    sixb_mask = (selection.nsixb_selected == 6)[mask]
    bkgs_mask = (selection.nsixb_selected < 6)[mask]
    labels = [f"Background Selected",f"Signal Selected"]
    colors = ["tab:orange","black"]
    histtypes=["bar","step"]
    
    nrows,ncols = 2,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,10) )
    X_reco = selection.reco_X()
    
    for i,(var,info) in enumerate(varinfo.items()):
        X_var = X_reco[var][mask]
        sixb_X_var = X_var[sixb_mask]
        bkgs_X_var = X_var[bkgs_mask]
        datalist = [bkgs_X_var,sixb_X_var]
        hist_multi(datalist,labels=labels,histtypes=histtypes,colors=colors,scale=scale,figax=(fig,axs[int(i/ncols),i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,"x_reco",saveas)

def x_res_study(selection,plot=True,saveas=None,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    scale = study.scale if scaled else 1
    
    if not plot: return
    
    varinfo = {
        f"m":{"bins":np.linspace(0,2,50),"xlabel":"X Reco Mass Resolution"},
        f"pt":{"bins":np.linspace(0,2,50),"xlabel":"X Reco Pt Resolution"},
        f"eta":{"bins":np.linspace(0,2,50),"xlabel":"X Reco Resolution"},
        f"phi":{"bins":np.linspace(0,2,50),"xlabel":"X Reco Resolution"},
    }
    
    mask = selection.mask
    sixb_mask = (selection.nsixb_selected == 6)[mask]
    bkgs_mask = (selection.nsixb_selected < 6)[mask]
    labels = [f"Background Selected",f"Signal Selected"]
    colors = ["tab:orange","black"]
    histtypes=["bar","step"]
    
    nrows,ncols = 2,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,10) )
    X_reco = selection.reco_X()
    
    for i,(var,info) in enumerate(varinfo.items()):
        X_tru = branches[f"X_{var}"][mask]
        X_var = X_reco[var][mask]/X_tru
        sixb_X_var = X_var[sixb_mask]
        bkgs_X_var = X_var[bkgs_mask]
        datalist = [bkgs_X_var,sixb_X_var]
        hist_multi(datalist,labels=labels,colors=colors,histtypes=histtypes,scale=scale,figax=(fig,axs[int(i/ncols),i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,"x_res",saveas)

def njet_study(selection,plot=True,saveas=None,density=0,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
    if not plot: return
    
    nsixb = min(6,selection.njets) if selection.njets != -1 else 6
    
    nrows,ncols = 1,3
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,5) )
    labels = ("Events",)
    
    mask = selection.mask
    njets = getattr(selection,f"njets_{subset}")[mask]
    nsixb = getattr(selection,f"nsixb_{subset}")[mask]
    nbkgs = getattr(selection,f"nbkgs_{subset}")[mask]
    
    hist_multi([njets],bins=range(16),xlabel=f"Number of Jets {subset.capitalize()}",labels=["Events"],density=density,figax=(fig,axs[0]))
    hist_multi([nsixb],bins=range(8),xlabel=f"Number of Signal Jets {subset.capitalize()}",labels=["Events"],density=density,figax=(fig,axs[1]))
    hist_multi([nbkgs],bins=range(8),xlabel=f"Number of Background Jets {subset.capitalize()}",labels=["Events"],density=density,figax=(fig,axs[2]))

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,f"njets_{subset}",saveas)
    

def presel_study(selection,plot=True,saveas=None,density=0,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    scale = study.scale if scaled else 1
    
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
        hist_multi(datalist,labels=labels,figax=(fig,axs[i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,f"presel_{subset}",saveas)
    
def jet_issue_study(selection,plot=True,saveas=None,density=0,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
    if not plot: return
    
    mask = selection.mask
    jets = selection.jets_selected
    
    ptbins = np.linspace(0,150,50)
    etabins = np.linspace(-3,3,50)
    
    jet_pt = ak.flatten(branches["jet_pt"][jets][mask])
    jet_eta = ak.flatten(branches["jet_eta"][jets][mask])
    eta24 = np.abs(jet_eta) < 2.4
    
    
    nrows,ncols = 2,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,10) )
    
    hist_multi([jet_pt],bins=ptbins,xlabel="jet pt",figax=(fig,axs[0,0]))
    hist_multi([jet_eta],bins=etabins,xlabel="jet eta",figax=(fig,axs[0,1]))
    
    hist2d_simple(jet_pt,jet_eta,xbins=ptbins,ybins=etabins,xlabel="jet pt",ylabel="jet eta",figax=(fig,axs[1,0]))
    hist2d_simple(jet_pt[eta24],jet_eta[eta24],xbins=ptbins,ybins=etabins,xlabel="jet pt",ylabel="jet eta",title="|jet eta|<2.4",figax=(fig,axs[1,1]))
    return fig,axs
    
def jet_comp_study(selection,plot=True,saveas=None,signal=False,density=0,**kwargs):
    study = Study(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    branches = selection.branches     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
    if not plot: return
    
    mask = selection.mask
    jets = getattr(selection,f"jets_{subset}")[mask]
    signal_tags = ["HX_b1","HX_b2","HY1_b1","HY1_b2","HY2_b1","HY2_b2"]
    signal_index = { tag: get_jet_index_mask(branches,branches[f"gen_{tag}_recojet_index"][:,np.newaxis])[mask] for tag in signal_tags }
    
    if not signal:
        signal_tags.append("Background")
        signal_index["Background"] = branches.bkgs_jet_mask[mask]
    
    signal_b_selected = { tag:jets & b_index for tag,b_index in signal_index.items() }
    
    nrows,ncols = 1,1
    fig0,ax0 = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,2.5))
    nsignal_b_selected = np.array([ak.sum(ak.sum(signal_mask,axis=-1)) for signal_mask in signal_b_selected.values() ])
    graph_simple(signal_tags,nsignal_b_selected,ylabel="Number Jets",figax=(fig0,ax0))
    
#     for i,(var,info) in enumerate(varinfo.items()):
        
#         nrows,ncols = 1,6
#         fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,5),sharey=True )
#         for i,(tag,b_selected) in enumerate(signal_b_selected.items()):
#             b_info = dict(info)
#             b_info["xlabel"] = f"{tag} {info['xlabel']}"
#             b_info["labels"] = [tag]
#             if i != 0: b_info["ylabel"] = ""
#             b_var = ak.flatten(branches[var][mask][b_selected])
#             hist_multi([b_var],**b_info,figax=(fig,axs[i]))

def compare_scores_study(scores,cutlist,cutlabel,values=("efficiency","purity"),prod=False,title=None,saveas=None,**kwargs):
    valuemap = { value:np.array([getattr(score,value) for score in scores]) for value in values }
    if prod and len(values) > 1: 
        prodkey = '*'.join(valuemap.keys())
        for scorelist in list(valuemap.values()):
            if prodkey not in valuemap: valuemap[prodkey] = scorelist
            else: valuemap[prodkey] = valuemap[prodkey] * scorelist 
    
    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(8,5))
    
    graph_multi(cutlist,valuemap.values(),xlabel=cutlabel,labels=valuemap.keys(),ylabel="A.U.",figax=(fig,axs))
    fig.suptitle(title)
    fig.tight_layout()
    
    if saveas: save_fig(fig,"compare",saveas)
    
    plt.show()
