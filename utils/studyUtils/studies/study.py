#!/usr/bin/env python
# coding: utf-8

from . import *

def njets(*args,**kwargs):
    study = Study(*args,**kwargs)
    
    nrows,ncols = 1,1
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(8,5))
    
    weights = [ selection["scale"] for selection in study.selections ]
    n_jet_list = [ selection["n_jet"] for selection in study.selections ]
    hist_multi(n_jet_list,weights=weights,bins=range(12),xlabel="N Jet",figax=(fig,axs),**vars(study))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,"njets",study.saveas)

def jets(*args,**kwargs):
    study = Study(*args,**kwargs)
    
    varlist=["jet_pt","jet_phi","jet_eta","jet_btag","jet_qgl"]
    
    nrows,ncols = 2,3
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,10))
    
    weights = [ selection["scale"] for selection in study.selections ]
    jet_weights = [ selection["jet_scale"] for selection in study.selections ]
    for i,varname in enumerate(varlist):
        hists = [ selection[varname] for selection in study.selections ]
        info = study.varinfo[varname]
        hist_multi(hists,weights=jet_weights,**info,figax=(fig,axs[i//ncols,i%ncols]),**vars(study))

    n_jet_list = [ selection["n_jet"] for selection in study.selections ]
    hist_multi(n_jet_list,bins=range(12),weights=weights,xlabel="N Jet",figax=(fig,axs[1,2]),**vars(study))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,"jets",study.saveas)

def njet_var_sum(*args,variable="jet_btag",**kwargs):
    study = Study(*args,**kwargs)

    nrows,ncols = 2,2
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,10))
    info = study.varinfo[variable]
    binmax = info['bins'][-1]
    
    weights = [ selection["scale"] for selection in study.selections ]
    selection_vars = [ ak.fill_none(ak.pad_none(selection[variable],6,axis=-1,clip=1),0) for selection in study.selections ]
    for i in range(4):
        ijet = i+3
        ijet_var_sum = [ ak.sum(var[:,:ijet],axis=-1) for var in selection_vars ]

        bins = np.linspace(0,binmax*(1+0.05*ijet),50)
        if variable == "jet_btag": bins = np.linspace(0,binmax*ijet,50)
        
        hist_multi(ijet_var_sum,weights=weights,bins=bins,**vars(study),xlabel=f"{ijet} {info['xlabel']} Sum",figax=(fig,axs[i//ncols,i%ncols]))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,f"n{variable}_sum",study.saveas)
