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
    
    weights = [ selection["scale"] for selection in study.selection ]
    jet_weights = [ selection["jet_scale"] for selection in study.selection ]
    for i,varname in enumerate(varlist):
        hists = [ selection[varname] for selection in study.selection ]
        info = study.varinfo[varname]
        hist_multi(hists,weights=jet_weights,**vars(study),**info,figax=(fig,axs[i//ncols,i%ncols]))

    n_jet_list = [ selection["n_jet"] for selection in selections ]
    hist_multi(n_jet_list,labels=labels,weights=weights,density=0,lumikey=2018,bins=range(12),xlabel="N Jet",figax=(fig,axs[1,2]))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if saveas: save_fig(fig,"",saveas)

def data_driven(*args,**kwargs):
    study = Study(*args,**kwargs)

    nrows,ncols = 2,2
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,10))
    
    weights = [ selection["scale"] for selection in study.selections ]
    selection_btags = [ ak.fill_none(ak.pad_none(selection["jet_btag"],6,axis=-1,clip=1),0) for selection in study.selections ]
    for i in range(4):
        ijet = i+3
        ijet_btag_sum = [ ak.sum(btag[:,:ijet],axis=-1) for btag in selection_btags ]
        hist_multi(ijet_btag_sum,weights=weights,bins=np.linspace(0,ijet,50),**vars(study),xlabel=f"{ijet} Jet Btag Sum",figax=(fig,axs[i//ncols,i%ncols]))

    fig.suptitle(study.title)
    fig.tight_layout()
    plt.show()
    if study.saveas: save_fig(fig,"data_driven",study.saveas)
