#!/usr/bin/env python
# coding: utf-8

from . import *

ordinal = lambda n : "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))
array_min = lambda array,value : ak.min(ak.concatenate(ak.broadcast_arrays(value,array[:,np.newaxis]),axis=-1),axis=-1)

def get_jet_ieta(jet_eta):
    cell_size = 2*1.4841/34
    jet_ieta = (jet_eta+1.4841)/cell_size
    return np.floor(jet_ieta)

def get_eta_from_ieta(ieta):
    cell_size = 2*1.4841/34
    eta = (ieta*cell_size)-1.4841+0.5*cell_size
    return eta

def get_jet_iphi(jet_phi):
    cell_size = 2*3.14159/72
    jet_iphi = (jet_phi+3.14159)/cell_size
    return np.floor(jet_iphi)

def get_phi_from_iphi(iphi):
    cell_size = 2*3.14159/72
    phi = (iphi*cell_size)-3.14159+0.5*cell_size
    return phi

def get_jet_index_mask(branches,index,jets=None):
    """ Generate jet mask for a list of indicies """
    if jets is None: jets = branches["jet_pt"]
    
    jet_index = ak.local_index( jets )
    compare , _ = ak.broadcast_arrays( index[:,None],jet_index )
    inter = (jet_index == compare)
    return ak.sum(inter,axis=-1) == 1

def exclude_jets(input_mask,exclude_mask):
    return (input_mask == True) & (exclude_mask == False)

def get_top_njet_index(branches,jet_index,njets=6):
    index_array = ak.local_index(jet_index)
    firstN_array = index_array < njets if njets != -1 else index_array != -1
    top_njet_index = jet_index[firstN_array]
    bot_njet_index = jet_index[firstN_array == False]
    return top_njet_index,bot_njet_index

def sort_jet_index_simple(branches,varbranch,jets=None):
    """ Mask of the top njet jets in varbranch """
    
    if jets is None: jets = branches["jet_pt"] > -999
    sorted_array = np.argsort(-varbranch)
    selected_sorted_array = jets[sorted_array]
    selected_array = sorted_array[selected_sorted_array]
    return selected_array

def sort_jet_index(branches,variable="jet_ptRegressed",jets=None):
    """ Mask of the top njet jets in variable """

    varbranch = branches[variable]
    if variable == "jet_eta": varbranch = -np.abs(varbranch)
    return sort_jet_index_simple(branches,varbranch,jets)

def count_sixb_index(jet_index,sixb_jet_mask):
    """ Number of signal b-jets in index list """
    
    nevts = ak.size(jet_index,axis=0)
    compare , _ = ak.broadcast_arrays( sixb_jet_mask[:,None], jet_index)
    inter = (jet_index == compare)
    count = ak.sum(ak.flatten(inter,axis=-1),axis=-1)
    return count

def count_sixb_mask(jet_mask,sixb_jet_mask):
    """ Number of signal b-jets in jet mask """
    
    inter = jet_mask & sixb_jet_mask
    return ak.sum(inter,axis=-1)

def get_sixb_position(jet_index,sixb_jet_mask):
    """ Get index positions of signal jets in sorted jet inde list"""
    ie = 5
    position = ak.local_index(jet_index,axis=-1)
    sixb_position = position[ sixb_jet_mask[jet_index] ]
    return sixb_position

# --- Standard Preselection --- #
def std_preselection(branches,ptcut=20,etacut=2.4,btagcut=None,jetid=1,puid=1,njetcut=0,passthrough=False,
                     exclude_events_mask=None,exclude_jet_mask=None,include_jet_mask=None,**kwargs):
    jet_mask = ak.broadcast_arrays(True,branches["jet_pt"])[0]
    
    if not passthrough:
        jet_mask = jet_mask & (branches["jet_ptRegressed"] > ptcut)
        jet_mask = jet_mask & (np.abs(branches["jet_eta"]) < etacut)
        if btagcut: jet_mask = jet_mask & (branches["jet_btag"] > btagcut)
        if jetid: jet_mask = jet_mask & ((1 << jetid) == branches["jet_id"] & ( 1 << jetid ))
        if puid: 
            puid_mask = (1 << puid) == branches["jet_puid"] & ( 1 << puid )
            low_pt_pu_mask = (branches["jet_ptRegressed"] < 50) & puid_mask
            jet_pu_mask = (branches["jet_ptRegressed"] >= 50) | low_pt_pu_mask
            jet_mask = jet_mask & jet_pu_mask
        
    if include_jet_mask is not None: jet_mask = jet_mask & include_jet_mask 
    if exclude_jet_mask is not None:     jet_mask = exclude_jets(jet_mask,exclude_jet_mask)
        
    event_mask = ak.sum(jet_mask,axis=-1) >= njetcut
    if exclude_events_mask is not None: event_mask = event_mask & exclude_events_mask
    return event_mask,jet_mask

