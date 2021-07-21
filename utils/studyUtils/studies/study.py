#!/usr/bin/env python
# coding: utf-8

from . import *


def njets(selections,labels=None,density=0,scaled=True,lumikey=2018,saveas=None):
    
    nrows,ncols = 1,1
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(8,5))

    if labels is None: labels = [ selection.tag for selection in selections ]
    weights = [ selection["scale"] for selection in selections ]
    n_jet_list = [ selection["n_jet"] for selection in selections ]
    hist_multi(n_jet_list,labels=labels,weights=weights,density=density,lumikey=lumikey,bins=range(12),xlabel="N Jet",figax=(fig,axs))

    fig.tight_layout()
    if saveas: save_fig(fig,"",saveas)

def jets(selections,labels=None,density=0,scaled=True,lumikey=2018,saveas=None):
    varlist=["jet_pt","jet_phi","jet_eta","jet_btag","jet_qgl"]
    
    nrows,ncols = 2,3
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,10))

    if labels is None: labels = [ selection.tag for selection in selections ]
    
    weights = [ selection["scale"] for selection in selections ]
    jet_weights = [ selection["jet_scale"] for selection in selections ]
    for i,varname in enumerate(varlist):
        hists = [ selection[varname] for selection in selections ]
        info = varinfo[varname]
        hist_multi(hists,labels=labels,weights=jet_weights,density=density,lumikey=lumikey,**info,figax=(fig,axs[i//ncols,i%ncols]))

    n_jet_list = [ selection["n_jet"] for selection in selections ]
    hist_multi(n_jet_list,labels=labels,weights=weights,density=0,lumikey=2018,bins=range(12),xlabel="N Jet",figax=(fig,axs[1,2]))

    fig.tight_layout()
    if saveas: save_fig(fig,"",saveas)

def data_driven(selections,labels=None,density=0,scaled=True,lumikey=2018,saveas=None):

    nrows,ncols = 2,2
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,10))
    
    weights = [ selection["scale"] for selection in selections ]
    selection_btags = [ ak.fill_none(ak.pad_none(selection["jet_btag"],6,axis=-1,clip=1),0) for selection in selections ]
    for i in range(4):
        ijet = i+3
        ijet_btag_sum = [ ak.sum(btag[:,:ijet],axis=-1) for btag in selection_btags ]
        hist_multi(ijet_btag_sum,labels=labels,weights=weights,density=density,lumikey=lumikey,bins=np.linspace(0,ijet,50),xlabel=f"{ijet} Jet Btag Sum",figax=(fig,axs[i//ncols,i%ncols]))
    
