#!/usr/bin/env python
# coding: utf-8

from . import *

ordinal = lambda n : "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))
array_min = lambda array,value : ak.min(ak.concatenate(ak.broadcast_arrays(value,array[:,np.newaxis]),axis=-1),axis=-1)

def get_jet_index_mask(jets,index):
    """ Generate jet mask for a list of indicies """
    if hasattr(jets,'ttree'): jets = jets["jet_pt"]
    
    jet_index = ak.local_index( jets )
    compare , _ = ak.broadcast_arrays( index[:,None],jet_index )
    inter = (jet_index == compare)
    return ak.sum(inter,axis=-1) == 1

def exclude_jets(input_mask,exclude_mask):
    return (input_mask == True) & (exclude_mask == False)

def get_top_njet_index(tree,jet_index,njets=6):
    index_array = ak.local_index(jet_index)
    firstN_array = index_array < njets if njets != -1 else index_array != -1
    top_njet_index = jet_index[firstN_array]
    bot_njet_index = jet_index[firstN_array == False]
    return top_njet_index,bot_njet_index

def sort_jet_index_simple(tree,varbranch,jets=None,method=max):
    """ Mask of the top njet jets in varbranch """
    
    if jets is None: jets = tree.all_jets_mask
    if varbranch is None: return ak.local_index(jets,axis=-1)
        
    polarity = -1 if method is max else 1
        
    sorted_array = np.argsort(polarity * varbranch)
    selected_sorted_array = jets[sorted_array]
    selected_array = sorted_array[selected_sorted_array]
    return selected_array

def sort_jet_index(tree,variable=None,jets=None,method=max):
    """ Mask of the top njet jets in variable """
        
    varbranch = tree[variable] if variable else None
    if variable == "jet_eta": varbranch = np.abs(varbranch)
    return sort_jet_index_simple(tree,varbranch,jets,method=method)

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

def get_jet_position(jet_index,jet_mask):
    """ Get index positions of jets in sorted jet index list"""
    position = ak.local_index(jet_index,axis=-1)
    jet_position = position[ jet_mask[jet_index] ]
    return jet_position

def calc_dphi(phi_1,phi_2):
    phi_2,phi_1 = ak.broadcast_arrays(phi_2[:,np.newaxis],phi_1)
    dphi = phi_2 - phi_1
    shift_over = ak.where(dphi > np.pi,-2*np.pi,0)
    shift_under= ak.where(dphi <= -np.pi,2*np.pi,0)
    return dphi + shift_over + shift_under

def calc_deta(eta_1,eta_2):
    eta_2,eta_1 = ak.broadcast_arrays(eta_2[:,np.newaxis],eta_1)
    return eta_2 - eta_1

def calc_dr(eta_1,phi_1,eta_2,phi_2):
    deta = calc_deta(eta_1,eta_2)
    dphi = calc_dphi(phi_1,phi_2)
    dr = np.sqrt( deta**2 + dphi**2 )
    return dr

def get_ext_dr(eta_1,phi_1,eta_2,phi_2):
    dr = calc_dr(eta_1,phi_1,eta_2,phi_2)
    dr_index = ak.local_index(dr,axis=-1)
    
    dr_index = dr_index[dr!=0]
    dr_reduced = dr[dr!=0]
    
    imin_dr = ak.argmin(dr_reduced,axis=-1,keepdims=True)
    min_dr = ak.flatten(dr_reduced[imin_dr],axis=-1)
    imin_dr = ak.flatten(dr_index[imin_dr],axis=-1)
    
    imax_dr = ak.argmax(dr_reduced,axis=-1,keepdims=True)
    max_dr = ak.flatten(dr_reduced[imax_dr],axis=-1)
    imax_dr = ak.flatten(dr_index[imax_dr],axis=-1)
    return dr,min_dr,imin_dr,max_dr,imax_dr

# --- Standard Preselection --- #
def std_preselection(tree,ptcut=20,etacut=2.5,btagcut=None,jetid=1,puid=1,njetcut=0,min_drcut=None,qglcut=None,
                     passthrough=False,exclude_events_mask=None,exclude_jet_mask=None,include_jet_mask=None,**kwargs):
    def jet_pu_mask(puid=puid):
        puid_mask = (1 << puid) == tree["jet_puid"] & ( 1 << puid )
        low_pt_pu_mask = (tree["jet_pt"] < 50) & puid_mask
        return (tree["jet_pt"] >= 50) | low_pt_pu_mask
    
    jet_mask = tree.all_jets_mask
    
    if include_jet_mask is not None: jet_mask = jet_mask & include_jet_mask 
    if exclude_jet_mask is not None: jet_mask = exclude_jets(jet_mask,exclude_jet_mask)
        
    if not passthrough:
        if ptcut: jet_mask = jet_mask & (tree["jet_pt"] > ptcut)
        if etacut: jet_mask = jet_mask & (np.abs(tree["jet_eta"]) < etacut)
        if btagcut: jet_mask = jet_mask & (tree["jet_btag"] > btagcut)
        if jetid: jet_mask = jet_mask & ((1 << jetid) == tree["jet_id"] & ( 1 << jetid ))
        if puid: jet_mask = jet_mask & jet_pu_mask()
        if min_drcut: jet_mask = jet_mask & (tree["jet_min_dr"] > min_drcut)
        if qglcut: jet_mask = jet_mask & (tree["jet_qgl"] > qglcut)
        
    event_mask = ak.sum(jet_mask,axis=-1) >= njetcut
    if exclude_events_mask is not None: event_mask = event_mask & exclude_events_mask
    return event_mask,jet_mask

def xmass_selected_signal(tree,jets_index,njets=6,invm=700):
    top_jets_index, _ = get_top_njet_index(tree,jets_index,njets=njets)
    
    jet_pt = tree["jet_pt"]
    jet_m = tree["jet_m"]
    jet_eta = tree["jet_eta"]
    jet_phi = tree["jet_phi"]

    comb_jets_index = ak.combinations(top_jets_index,6)
    build_p4 = lambda index : vector.obj(pt=jet_pt[index],mass=jet_m[index],eta=jet_eta[index],phi=jet_phi[index])
    jet0, jet1, jet2, jet3, jet4, jet5 = [build_p4(jet) for jet in ak.unzip(comb_jets_index)]
    x_invm = (jet0+jet1+jet2+jet3+jet4+jet5).mass
    signal_comb = ak.argmin(np.abs(x_invm - invm),axis=-1)
    comb_mask = get_jet_index_mask(comb_jets_index,signal_comb[:,np.newaxis])
    jets_selected_index = ak.concatenate(ak.unzip(comb_jets_index[comb_mask]),axis=-1)
    
    selected_compare, _ = ak.broadcast_arrays(jets_selected_index[:,np.newaxis],jets_index)
    jets_remaining_index = jets_index[ ak.sum(jets_index==selected_compare,axis=-1)==0 ]
    
    return jets_selected_index,jets_remaining_index
