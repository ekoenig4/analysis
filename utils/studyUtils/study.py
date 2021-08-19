#!/usr/bin/env python
# coding: utf-8

from . import *

def autodim(nvar,dim=None,flip=False):
    if nvar % 2 == 1 and nvar != 1: nvar += 1
    if dim is not None:
        nrows,ncols = dim
    elif nvar == 1:
        nrows,ncols = 1,1
    elif flip:
        ncols = nvar//2
        nrows = nvar//ncols
    else:
        nrows = nvar//2
        ncols = nvar//nrows
    return nrows,ncols

def cutflow(*args,size=(16,8),**kwargs):
    study = Study(*args,**kwargs)
    get_scaled_cutflow = lambda tree : np.array([cutflow*scale for cutflow,scale in zip(tree.cutflow,tree.scales)])
    scaled_cutflows = [ get_scaled_cutflow(tree) for tree in study.selections ]
    cutflow_bins = [ ak.local_index(cutflow,axis=-1) for cutflow in scaled_cutflows ]
    cutflow_labels = max((selection.cutflow_labels for selection in study.selections),key=lambda a:len(a))
    ncutflow = len(cutflow_labels)+1
    bins = np.arange(ncutflow)-0.5

    figax = None
    if size: figax = plt.subplots(figsize=size)
        
    fig,ax = hist_multi(cutflow_bins,bins=bins,weights=scaled_cutflows,xlabel=cutflow_labels,histtypes=["step"]*len(study.selections),**vars(study),figax=figax)
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,"cutflow",study.saveas)

def quick(*args,varlist=[],binlist=None,dim=None,flip=False,**kwargs):
    study = Study(*args,**kwargs)

    nvar = len(varlist)
    binlist = init_attr(binlist,None,nvar)

    nrows,ncols = autodim(nvar,dim,flip)
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=( int((16/3)*ncols),5*nrows ))

    event_weights = study.get("scale")
    jet_weights = study.get("jet_scale")
    higgs_weights = study.get("higgs_scale")
    for i,(var,bins) in enumerate(zip(varlist,binlist)):

        xlabel = var
        if var in study.varinfo:
            if bins is None: bins = study.varinfo[var]["bins"]
            xlabel = study.varinfo[var]["xlabel"]
            
        hists = study.get(var)
        weights = next( (weights for weights in [event_weights,jet_weights,higgs_weights] if ak.count(weights[0]) == ak.count(hists[0])),None )

        if ncols == 1 and nrows == 1: ax = axs
        elif bool(ncols >1) != bool(nrows > 1): ax = axs[i]
        else: ax = axs[i//ncols,i%ncols]
        
        hist_multi(hists,bins=bins,xlabel=xlabel,weights=weights,**vars(study),figax=(fig,ax))
    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,"",study.saveas)

def njets(*args,**kwargs):
    study = Study(*args,**kwargs)
    
    nrows,ncols = 1,4
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
    
    weights = study.get("scale")

    varlist = ["n_jet","nloose_btag","nmedium_btag","ntight_btag"]

    for i,var in enumerate(varlist):
        tree_vars = study.get(var)
        maxjet= int(max( ak.max(var) for var in tree_vars))
        hist_multi(tree_vars,weights=weights,bins=range(maxjet+3),xlabel=var,figax=(fig,axs[i]),**vars(study))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,"njets",study.saveas)

def jets(*args,**kwargs):
    study = Study(*args,**kwargs)
    
    varlist=["jet_pt","jet_btag","jet_eta","jet_phi","jet_qgl"]
    
    nrows,ncols = 2,3
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,10))
    
    weights = study.get("scale")
    jet_weights = study.get("jet_scale")
    for i,varname in enumerate(varlist):
        hists = study.get(varname)
        info = study.varinfo[varname]
        hist_multi(hists,weights=jet_weights,**info,figax=(fig,axs[i//ncols,i%ncols]),**vars(study))

    n_jet_list = study.get("n_jet")
    hist_multi(n_jet_list,bins=range(12),weights=weights,xlabel="N Jet",figax=(fig,axs[1,2]),**vars(study))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,"jets",study.saveas)
    

def ijets(*args,njets=6,**kwargs):
    study = Study(*args,**kwargs)
    
    varlist=["jet_pt","jet_btag","jet_eta","jet_phi"]
    weights = study.get("scale")
    
    for ijet in range(njets):
        nrows,ncols = 1,4
        fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))

        for i,varname in enumerate(varlist):
            hists = [ var[:,ijet] for var in study.get(varname) ]
            info = study.varinfo[varname]
            hist_multi(hists,weights=weights,**info,figax=(fig,axs[i]),**vars(study))
            
        fig.suptitle(f"{ordinal(ijet+1)} Jet Distributions")
        fig.tight_layout()
        plt.show()
        if study.saveas: save_fig(fig,"ijets",f"jet{ijet}_{study.saveas}")
    
def higgs(*args,**kwargs):
    study = Study(*args,**kwargs)
    
    varlist=["higgs_pt","higgs_m","higgs_eta","higgs_phi"]
    
    nrows,ncols = 2,2
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,10))

    higgs_weights = study.get("higgs_scale")
    for i,varname in enumerate(varlist):
        hists = study.get(varname)
        info = study.varinfo[varname]
        hist_multi(hists,weights=higgs_weights,**info,figax=(fig,axs[i//ncols,i%ncols]),**vars(study))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,"higgs",study.saveas)
    
def ihiggs(*args,nhiggs=3,**kwargs):
    study = Study(*args,**kwargs)
    
    varlist=["higgs_pt","higgs_m","higgs_eta","higgs_phi"]

    weights = [ selection["scale"] for selection in study.selections ]
    for ihigg  in range(nhiggs):
        
        nrows,ncols = 1,4
        fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
        for i,varname in enumerate(varlist):
            hists = [ var[:,ihigg] for var in study.get(varname) ]
            info = study.varinfo[varname]
            hist_multi(hists,weights=weights,**info,figax=(fig,axs[i]),**vars(study))

        fig.suptitle(f"{ordinal(ihigg+1)} Higgs Distributions")
        fig.tight_layout()
        plt.show()
        if study.saveas: save_fig(fig,"ihiggs",f"higgs{ihigg}_{study.saveas}")

def njet_var_sum(*args,variable="jet_btag",start=3,**kwargs):
    study = Study(*args,**kwargs)

    nrows,ncols = 2,2
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,10))
    info = study.varinfo[variable]
    binmax = info['bins'][-1]
    
    weights = [ selection["scale"] for selection in study.selections ]
    selection_vars = [ ak.fill_none(ak.pad_none(selection[variable],6,axis=-1,clip=1),0) for selection in study.selections ]
    for i in range(4):
        ijet = i+start
        ijet_var_sum = [ ak.sum(var[:,:ijet],axis=-1) for var in selection_vars ]

        varstd = max([ ak.std(var,axis=None) for var in ijet_var_sum ])
        varavg = max([ ak.mean(var,axis=None) for var in ijet_var_sum ])
        
        bins = np.linspace(varavg-varstd,varavg+varstd,50)
        if variable == "jet_btag": bins = np.linspace(0,binmax*ijet,50)
        
        hist_multi(ijet_var_sum,weights=weights,bins=bins,**vars(study),xlabel=f"{ijet} {info['xlabel']} Sum",figax=(fig,axs[i//ncols,i%ncols]))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,f"n{variable}_sum",study.saveas)

        
def jet_display(*args,ie=0,printout=[],boosted=False,**kwargs):
    study = Study(*args,title="",**kwargs)
    tree = study.selections[0]
    
    for out in printout:
        print(f"{out}: {tree[out][ie]}")

    njet = tree["n_jet"][ie]
    jet_pt = tree["jet_pt"][ie][np.newaxis]
    jet_eta = tree["jet_eta"][ie][np.newaxis]
    jet_phi = tree["jet_phi"][ie][np.newaxis]
    jet_m = tree["jet_m"][ie][np.newaxis]

    if boosted:
        boost = com_boost_vector(jet_pt,jet_eta,jet_phi,jet_m,njet=njet)
        boosted_jets = vector.obj(pt=jet_pt,eta=jet_eta,phi=jet_phi,m=jet_m).boost_p4(boost)
        jet_pt,jet_eta,jet_phi,jet_m = boosted_jets.pt,boosted_jets.eta,boosted_jets.phi,boosted_jets.m
    
    fig = plt.figure(figsize=(10,5))
    plot_barrel_display(jet_eta,jet_phi,jet_pt,figax=(fig,fig.add_subplot(1,2,1)))
    plot_endcap_display(jet_eta,jet_phi,jet_pt,figax=(fig,fig.add_subplot(1,2,2,projection='polar')))
    
    r,l,e,id = [ tree[info][ie] for info in ("Run","Event","LumiSec","sample_id") ]
    sample = tree.samples[id]

    title = f"{sample} | Run: {r} | Lumi: {l} | Event: {e}"
    if boosted: title = f"Boosted COM: {title}"
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    if study.saveas: save_fig(fig,"jet_display",study.saveas)
    
def jet_sphericity(*args,**kwargs):
    study = Study(*args,**kwargs)
    
    nrows,ncols = 2,3
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,10))
    shapes = ["M_eig_w1","M_eig_w2","M_eig_w3","event_S","event_St","event_A"]
    weights = [ selection["scale"] for selection in study.selections ]
    for i,shape in enumerate(shapes):
        shape_var = [ selection[shape] for selection in study.selections ]
        info = shapeinfo[shape]
        hist_multi(shape_var,weights=weights,**info,**vars(study),figax=(fig,axs[i//ncols,i%ncols]))
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,"sphericity",study.saveas)
        
def jet_thrust(*args,**kwargs):
    study = Study(*args,**kwargs)
    
    nrows,ncols = 1,3
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,5))
    shapes = ["thrust_phi","event_Tt","event_Tm"]
    weights = [ selection["scale"] for selection in study.selections ]
    
    for i,shape in enumerate(shapes):
        shape_var = [ selection[shape] for selection in study.selections ]
        info = shapeinfo[shape]
        hist_multi(shape_var,weights=weights,**info,**vars(study),figax=(fig,axs[i]))
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,"thrust",study.saveas)
