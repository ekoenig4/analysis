#!/usr/bin/env python
# coding: utf-8

from . import *

class SignalStudy(Study):
    def __init__(self,selection,subset="selected",plot=True,**kwargs):
        Study.__init__(self,selection,**kwargs)
        selection.build_extra_collections()

        self.selection = self.selections[0]
        if self.title is None: self.title = self.selection.title()
        self.subset = subset
        self.tree = selection.tree
        self.plot = plot
        self.lumikey = None

    def get(self,varname,subset=None,mask=None):
        var = self.tree[varname]
        if subset is not None: var = var[subset]
        if mask is not None: var = var[mask]
        return var

    def get_jets(self,varname,jets="jets",subset=None,mask=None):
        if mask is None: mask = self.selection.mask
        
        jet_mask = self.get_subset(jets,subset)
        return self.get(varname,jet_mask,mask)

    def get_subset(self,jets,subset=None):
        if subset is None: subset = self.subset
        return getattr(self.selection,f"{jets}_{subset}")

    def get_subset_index(self,jets,subset=None):
        if subset is None: subset = self.subset
        return getattr(self.selection,f"{jets}_{subset}_index")

def signal_order(selection,plot=True,saveas=None,**kwargs):
    study = SignalStudy(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    tree = selection.tree     
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
        isixb_mask = get_jet_index_mask(tree,sixb_ordered[:,ijet][:,np.newaxis])
        
        if ak.sum(ak.flatten(isixb_mask[nsixb_mask])) == 0: 
            print(f"*** No signal selection in position {ijet} ***")
            continue
            
        nrows,ncols=1,3
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
        for i,(var,info) in enumerate(varinfo.items()):
            ord_info = dict(info)
            ord_info["xlabel"] = f"{ordinal(ijet+1)} {info['xlabel']}"
            sixb_var = tree[var][isixb_mask][nsixb_mask]
            sixb_data = ak.flatten(sixb_var)
                
            if var == "jet_pt": ptinfo = ord_info
            if var == "jet_eta": etainfo = ord_info
            if var == "jet_btag": btaginfo = ord_info
            
            datalist = (sixb_data,)
            hist_multi(datalist,labels=labels,figax=(fig,axs[i%ncols]),**ord_info)
            
#         sixb_ptdata = ak.flatten( tree["jet_pt"][isixb_mask][nsixb_mask] )
#         sixb_etadata = ak.flatten( tree["jet_eta"][isixb_mask][nsixb_mask] )
#         sixb_btagdata = ak.flatten( tree["jet_btag"][isixb_mask][nsixb_mask] )
        
        
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
    
def selection(selection,plot=True,saveas=None,under6=False,latex=False,required=False,scaled=False,**kwargs):
    study = SignalStudy(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    tree = selection.tree     
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
        
def selection_comparison(selections,plot=True,saveas=None,under6=False,latex=False,required=False,title=None,labels=None,**kwargs):
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

def jets(*args,varlist=["jet_pt","jet_eta","jet_btag","jet_qgl"],**kwargs):
    study = SignalStudy(*args,varlist=varlist,labels=("All Jets",f"Background Jets",f"Signal Jets"),s_colors=("blue","black","tab:orange"),**kwargs)    
    
    if not study.plot: return
            
    nrows,ncols=1,4
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
    for i,(var,info) in enumerate(study.varinfo.items()):
        bkgs_var = study.get_jets(var,jets="bkgs")
        sixb_var = study.get_jets(var,jets="sixb")
        jets_var = study.get_jets(Var,jets="jets")
        hist_multi((jets_var,bkgs_var,sixb_var),figax=(fig,axs[i]),**info,**study.attrs)
            
    fig.suptitle(f"{study.title}")
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,f"jets_{study.subset}",study.saveas)
        

def jets_2d(selection,log=True,**kwargs):
    study = SignalStudy(selection,log=log,**kwargs)   
    
    if not study.plot: return
    
    labels = (f"Background Jets",f"Signal Jets")

    plot2d = hist2d_simple
    ptinfo = study.varinfo["jet_pt"]
    btaginfo = study.varinfo["jet_btag"]
            
    nrows,ncols=1,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))

    bkgs_btag = study.get_jets("jet_btag",jets="bkgs")
    bkgs_pt =   study.get_jets("jet_pt",jets="bkgs")

    sixb_btag = study.get_jets("jet_btag",jets="sixb")
    sixb_pt =   study.get_jets("jet_pt",jets="sixb")

    hist2d_simple(bkgs_pt,bkgs_btag,ybins=btaginfo["bins"],xbins=ptinfo["bins"],ylabel=btaginfo["xlabel"],xlabel=ptinfo["xlabel"],title=labels[0],density=study.density,log=study.log,figax=(fig,axs[0]))
    hist2d_simple(sixb_pt,sixb_btag,ybins=btaginfo["bins"],xbins=ptinfo["bins"],ylabel=btaginfo["xlabel"],xlabel=ptinfo["xlabel"],title=labels[1],density=study.density,log=study.log,figax=(fig,axs[1]))
            
    fig.suptitle(f"{study.title}")
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,f"jets_2d_{subset}",study.saveas)
        
def ijets(selection,varlist=["jet_pt","jet_btag","jet_eta","jet_phi"],njets=6,**kwargs):
    study = SignalStudy(selection,varlist=varlist,s_colors=("blue","black","tab:orange"),**kwargs)
    
    if not study.plot: return

    mask = selection.mask
    bkgs_mask = study.get_subset("bkgs")
    sixb_mask = study.get_subset("sixb")
    jets_mask = study.get_subset("jets")

    maxjets = ak.max(study.selection.njets_selected)
    if type(njets) == int:

        njets = maxjets if njets == -1 else min(njets,maxjets)
        njets = range(njets)
    
    jets_ordered = ak.pad_none(study.get_subset_index("jets"),maxjets)
    for ijet in njets:
        study.labels = ("All Jets",f"Background Jet",f"Signal Jet")
        
        ijet_mask = get_jet_index_mask(jets_mask,jets_ordered[:,ijet][:,np.newaxis])
        isixb_mask = ijet_mask & sixb_mask
        ibkgs_mask = ijet_mask & bkgs_mask
            
        nrows,ncols=1,4
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
        for i,(var,info) in enumerate(study.varinfo.items()):
            ord_info = dict(**info)
            ord_info["xlabel"] = f"{ordinal(ijet+1)} {info['xlabel']}"
                               
            bkgs_var = study.get(var,ibkgs_mask,mask)
            sixb_var = study.get(var,isixb_mask,mask)
            jets_var = study.get(var,ijet_mask,mask)
            hist_multi((jets_var,bkgs_var,sixb_var),figax=(fig,axs[i]),**ord_info,**study.attrs)
            
        fig.suptitle(f"{study.title} {ordinal(ijet+1)} Jet")
        fig.tight_layout()
        plt.show()
        if study.saveas: save_fig(fig,f"ijets_{subset}",f"{ordinal(ijet+1)}_{study.saveas}")

def jets_ordered(*args,varlist=["jet_pt","jet_eta","jet_btag","jet_qgl"],njets=1,topbkg=True,**kwargs):
    study = SignalStudy(*args,varlist=varlist,**kwargs) 
    
    if not study.plot: return
    
    label1 = "Top Background Jet" if topbkg else "Background Jets"
    colors = ["tab:orange","black"]
    histtypes = ["bar","step"]
    
    mask = study.selection.mask
    
    bkgs = getattr(study.selection,f"bkgs_{study.subset}")
    nbkgs= getattr(study.selection,f"nbkgs_{study.subset}")
    bkgs_position = getattr(study.selection,f"bkgs_{study.subset}_position")
    bkgs_ordered = getattr(study.selection,f"bkgs_{study.subset}_index")
    
    sixb = getattr(study.selection,f"sixb_{study.subset}")
    nsixb= getattr(study.selection,f"nsixb_{study.subset}")
    sixb_position = getattr(study.selection,f"sixb_{study.subset}_position")
    sixb_ordered = getattr(study.selection,f"sixb_{study.subset}_index")
    
    
    if topbkg: bkgs_ordered = ak.pad_none(bkgs_ordered,1,axis=-1,clip=1)
    
    bkgs_mask = get_jet_index_mask(study.tree,bkgs_ordered)
    sixb_ordered = ak.pad_none(sixb_ordered,6,axis=-1)
    
    njets = min(njets,study.selection.njets if study.selection.njets != -1 else 6) 
    
    for ijet in range(njets):
        study.labels = [label1,f"{ordinal(ijet+1)} Signal Jet"]
        nsixb_mask = mask & (nsixb > ijet)
        isixb_mask = get_jet_index_mask(study.tree,sixb_ordered[:,ijet][:,np.newaxis])
        
        if ak.sum(ak.flatten(isixb_mask[nsixb_mask])) == 0: 
            print(f"*** No incorrect selection in position {ijet} ***")
            continue
            
        nrows,ncols=1,4
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
        for i,(var,info) in enumerate(study.varinfo.items()):
            jets_var = ak.flatten(study.tree[var][bkgs_mask][mask])
            sixb_var = ak.flatten(study.tree[var][isixb_mask][nsixb_mask])
            hist_multi((sixb_var,jets_var),figax=(fig,axs[i]),**info,
                        colors=colors,histtypes=histtypes,**study.attrs)
            
        fig.suptitle(f"{study.title} {ordinal(ijet+1)} Signal Jet")
        fig.tight_layout()
        plt.show()
        if study.saveas: save_fig(fig,f"jets_{study.subset}",f"{ordinal(ijet+1)}_{study.saveas}")
            
def jets_2d_ordered(selection,plot=True,saveas=None,njets=1,topbkg=True,compare=True,density=0,log=1,**kwargs):
    study = SignalStudy(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    tree = selection.tree     
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
    
    bkgs_mask = get_jet_index_mask(tree,bkgs_ordered)
    sixb_ordered = ak.pad_none(sixb_ordered,6,axis=-1)
    
    njets = min(njets,selection.njets if selection.njets != -1 else 6) 
    
    plot2d = hist2d_simple
    ptinfo = varinfo["jet_pt"]
    btaginfo = varinfo["jet_btag"]
    
    for ijet in range(njets):
        labels = (label1,f"{ordinal(ijet+1)} Signal Jet")
        nsixb_mask = mask & (nsixb > ijet)
        isixb_mask = get_jet_index_mask(tree,sixb_ordered[:,ijet][:,np.newaxis])
        
        if ak.sum(ak.flatten(isixb_mask[nsixb_mask])) == 0: 
            print(f"*** No incorrect selection in position {ijet} ***")
            continue
            
        nrows,ncols=1,2
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
        
        if not (compare and topbkg):
            jets_btag = ak.flatten(tree["jet_btag"][bkgs_mask][mask])
            jets_pt = ak.flatten(tree["jet_pt"][bkgs_mask][mask])

            sixb_btag = ak.flatten(tree["jet_btag"][isixb_mask][nsixb_mask])
            sixb_pt =     ak.flatten(tree["jet_pt"][isixb_mask][nsixb_mask])

            plot2d(jets_pt,jets_btag,ybins=btaginfo["bins"],xbins=ptinfo["bins"],ylabel=btaginfo["xlabel"],xlabel=ptinfo["xlabel"],
                   title=labels[0],density=density,log=log,figax=(fig,axs[0]))
            plot2d(sixb_pt,sixb_btag,ybins=btaginfo["bins"],xbins=ptinfo["bins"],ylabel=btaginfo["xlabel"],xlabel=ptinfo["xlabel"],
                   title=labels[1],density=density,log=log,figax=(fig,axs[1]))
        else:
            compared_jets = nsixb_mask &  (selection.nbkgs_selected > 0)
            njets_compared = ak.sum(compared_jets)

            jets_btag = ak.flatten(tree["jet_btag"][bkgs_mask][compared_jets])
            jets_pt = ak.flatten(tree["jet_pt"][bkgs_mask][compared_jets])

            sixb_btag = ak.flatten(tree["jet_btag"][isixb_mask][compared_jets])
            sixb_pt =     ak.flatten(tree["jet_pt"][isixb_mask][compared_jets])
            
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
    
def x_reco(selection,plot=True,saveas=None,scaled=False,**kwargs):
    study = SignalStudy(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    tree = selection.tree     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
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
        hist_multi(datalist,labels=labels,histtypes=histtypes,colors=colors,figax=(fig,axs[int(i/ncols),i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,"x_reco",saveas)

def x_res(selection,plot=True,saveas=None,**kwargs):
    study = SignalStudy(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    tree = selection.tree     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
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
        X_tru = tree[f"X_{var}"][mask]
        X_var = X_reco[var][mask]/X_tru
        sixb_X_var = X_var[sixb_mask]
        bkgs_X_var = X_var[bkgs_mask]
        datalist = [bkgs_X_var,sixb_X_var]
        hist_multi(datalist,labels=labels,colors=colors,histtypes=histtypes,figax=(fig,axs[int(i/ncols),i%ncols]),**info)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,"x_res",saveas)

def njet(selection,plot=True,saveas=None,density=0,**kwargs):
    study = SignalStudy(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    tree = selection.tree     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
    if not plot: return
    
    nrows,ncols = 1,3
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,5) )
    labels = ("Events",)
    
    mask = selection.mask
    njets = getattr(selection,f"njets_{subset}")[mask]
    nsixb = getattr(selection,f"nsixb_{subset}")[mask]
    nbkgs = getattr(selection,f"nbkgs_{subset}")[mask]
    
    hist_multi([njets],bins=range(16),xlabel=f"Number of Jets {subset.capitalize()}",figax=(fig,axs[0]),**study.attrs)
    hist_multi([nsixb],bins=range(8),xlabel=f"Number of Signal Jets {subset.capitalize()}",figax=(fig,axs[1]),**study.attrs)
    hist_multi([nbkgs],bins=range(8),xlabel=f"Number of Background Jets {subset.capitalize()}",figax=(fig,axs[2]),**study.attrs)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,f"njets_{subset}",saveas)
    

def presel(selection,plot=True,saveas=None,density=0,**kwargs):
    study = SignalStudy(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    tree = selection.tree     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
    if not plot: return
    
    nsixb = min(6,selection.njets) if selection.njets != -1 else 6
    mask = selection.mask
    
    jets = getattr(selection,f"jets_{subset}")
    sixb = getattr(selection,f"sixb_{subset}")
    
    signal_mask = mask & (getattr(selection,f"nsixb_{subset}") == nsixb)

    nrows,ncols = 1,4
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,5) )
    for i,(var,info) in enumerate(varinfo.items()):
        jet_var = tree[var][jets]
        sixb_var = tree[var][sixb]
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
    
def jet_issue(selection,plot=True,saveas=None,density=0,**kwargs):
    study = SignalStudy(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    tree = selection.tree     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
    if not plot: return
    
    mask = selection.mask
    jets = selection.jets_selected
    
    ptbins = np.linspace(0,150,50)
    etabins = np.linspace(-3,3,50)
    
    jet_pt = ak.flatten(tree["jet_pt"][jets][mask])
    jet_eta = ak.flatten(tree["jet_eta"][jets][mask])
    eta24 = np.abs(jet_eta) < 2.4
    
    
    nrows,ncols = 2,2
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols, figsize=(16,10) )
    
    hist_multi([jet_pt],bins=ptbins,xlabel="jet pt",figax=(fig,axs[0,0]))
    hist_multi([jet_eta],bins=etabins,xlabel="jet eta",figax=(fig,axs[0,1]))
    
    hist2d_simple(jet_pt,jet_eta,xbins=ptbins,ybins=etabins,xlabel="jet pt",ylabel="jet eta",figax=(fig,axs[1,0]))
    hist2d_simple(jet_pt[eta24],jet_eta[eta24],xbins=ptbins,ybins=etabins,xlabel="jet pt",ylabel="jet eta",title="|jet eta|<2.4",figax=(fig,axs[1,1]))
    return fig,axs
    
def jet_comp(selection,plot=True,saveas=None,signal=False,density=0,**kwargs):
    study = SignalStudy(selection,saveas=saveas,**kwargs)     
    selection = study.selection     
    tree = selection.tree     
    subset = study.subset
    title = study.title
    varinfo = study.varinfo
    
    if not plot: return
    
    mask = selection.mask
    jets = getattr(selection,f"jets_{subset}")[mask]
    signal_tags = ["HX_b1","HX_b2","HY1_b1","HY1_b2","HY2_b1","HY2_b2"]
    signal_index = { tag: get_jet_index_mask(tree,tree[f"gen_{tag}_recojet_index"][:,np.newaxis])[mask] for tag in signal_tags }
    
    if not signal:
        signal_tags.append("Background")
        signal_index["Background"] = tree.bkgs_jet_mask[mask]
    
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
#             b_var = ak.flatten(tree[var][mask][b_selected])
#             hist_multi([b_var],**b_info,figax=(fig,axs[i]))

def compare_scores(scores,cutlist,cutlabel,values=("efficiency","purity"),prod=False,title=None,saveas=None,**kwargs):
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
