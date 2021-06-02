#!/usr/bin/env python
# coding: utf-8

from . import *

ordinal = lambda n : "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))
array_min = lambda array,value : ak.min(ak.concatenate(ak.broadcast_arrays(value,array[:,np.newaxis]),axis=-1),axis=-1)

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

def calculate_values(branches,mask,nsixb_selected,nsixb_remaining,nsixb_captured,njets):
    mask_evnts = ak.sum(mask)
    signal_evnts= ak.sum(mask & branches.sixb_found_mask)
    
    total_eff = mask_evnts/float(branches.nevents)
    signal_eff = signal_evnts/float(branches.nsignal)
    purity = signal_evnts/float(mask_evnts)
    
    if njets < 0: njets = 6
    
    nsixb = min(6,njets)
    
    total_nsixb = ak.sum(branches["nfound_all"][mask])
    signal_nsixb = ak.sum(branches["nfound_all"][mask & branches.sixb_found_mask])

    total_min_nsixb = ak.sum(array_min(branches["nfound_all"][mask],nsixb))
    
    total_remain = ak.sum(nsixb_remaining[mask])
    signal_remain = ak.sum(nsixb_remaining[mask & branches.sixb_found_mask])
    
    total_selected = ak.sum(nsixb_selected[mask])
    signal_selected = ak.sum(nsixb_selected[mask & branches.sixb_found_mask])
    
    total_captured = ak.sum(nsixb_captured[mask])
    signal_captured = ak.sum(nsixb_captured[mask & branches.sixb_found_mask])
    
    return {
        "event_eff":total_eff,
        "event_avg_selected":total_selected/float(mask_evnts),
        "event_per_selected":total_selected/float(total_min_nsixb),
        "event_total_selected":total_selected/float(total_nsixb),
        "event_avg_captured":total_captured/float(mask_evnts),
        "event_per_captured":total_captured/float(total_nsixb),
        "signal_eff":signal_eff,
        "signal_avg_selected":signal_selected/float(signal_evnts),
        "signal_per_selected":signal_selected/float(nsixb*signal_evnts),
        "signal_total_selected":signal_selected/float(signal_nsixb),
        "signal_avg_captured":signal_captured/float(signal_evnts),
        "signal_per_captured":signal_captured/float(signal_nsixb),
        "purity":purity,
    }

def calculate_selection(branches,selection):
    return calculate_values(branches,selection["mask"],selection["nsixb_selected"],selection["nsixb_remaining"],selection["nsixb_captured"],selection["njets"])

def print_values(values):
    prompt_list = [
        "Event  Efficiency:     {event_eff:0.2}",
        "Signal Efficiency:     {signal_eff:0.2f}",
        "Signal Purity:         {purity:0.2f}",
        "Event  Avg Selected:   {event_avg_selected:0.2f} -> {event_per_selected:0.2%}",
        "Signal Avg Selected:   {signal_avg_selected:0.2f} -> {signal_per_selected:0.2%}",
        "Event  Total Selected: {event_total_selected:0.2%}",
        "Signal Total Selected: {signal_total_selected:0.2%}",
        "Event  Avg Captured:   {event_avg_captured:0.2f} -> {event_per_captured:0.2%}",
        "Signal Avg Captured:   {signal_avg_captured:0.2f} -> {signal_per_captured:0.2%}",
    ]
    prompt = '\n'.join(prompt_list)
    
    print(prompt.format(**values))

def get_selected(branches,variable,mask=None,jets=None,sixb_found_mask=None,sixb_jet_mask=None,sixb_removed=None,njets=6):
    if sixb_found_mask is None: sixb_found_mask = branches.sixb_found_mask
    
    selected_index = sort_jet_index(branches,variable,jets=jets,njets=njets)
    selected = get_jet_index_mask(branches,selected_index)
    njet_selected = ak.sum(selected,axis=-1)
    
    sixb_position = get_sixb_position(selected_index,sixb_jet_mask)
    nsixb_selected = count_sixb_mask(selected,sixb_jet_mask)
    
    sixb_remaining = exclude_jets(sixb_jet_mask,selected)
    nsixb_remaining = ak.sum(sixb_remaining,axis=-1)

    sixb_captured = jets & sixb_jet_mask
    nsixb_captured = ak.sum(sixb_captured,axis=-1)
    
    if sixb_removed is not None: sixb_jet_mask = exclude_jets(sixb_jet_mask,sixb_removed)
    sixb_selected = sort_jet_mask(branches,variable,jets=sixb_jet_mask & jets,njets=njets)
    
    return {
        "mask":mask,
        "jet_mask":jets,
        "variable":variable,
        "njets":njets,
        "selected":selected,
        "njet_selected":njet_selected,
        "sixb_selected":sixb_selected,
        "sixb_position":sixb_position,
        "nsixb_selected":nsixb_selected,
        "sixb_remaining":sixb_remaining,
        "nsixb_remaining":nsixb_remaining,
        "sixb_captured":sixb_captured,
        "nsixb_captured":nsixb_captured
    }

# --- Standard Preselection --- #
def std_preselection(branches,ptcut=30,etacut=2.4,btagcut=None,jetid=1,puid=1,njetcut=0,exclude_events_mask=None,exclude_jet_mask=None,exclude=True,**kwargs):
    jet_pt_mask = branches["jet_ptRegressed"] > ptcut
    jet_eta_mask = np.abs(branches["jet_eta"]) < etacut
    jet_mask = jet_pt_mask & jet_eta_mask
    
    if btagcut: jet_mask = jet_mask & (branches["jet_btag"] > btagcut)
    if jetid: jet_mask = jet_mask & ((1 << jetid) == branches["jet_id"] & ( 1 << jetid ))
    if puid: 
        puid_mask = (1 << puid) == branches["jet_puid"] & ( 1 << puid )
        low_pt_pu_mask = (branches["jet_ptRegressed"] < 50) & puid_mask
        jet_pu_mask = (branches["jet_ptRegressed"] >= 50) | low_pt_pu_mask
        jet_mask = jet_mask & jet_pu_mask
        
    if exclude_jet_mask is not None and exclude:     jet_mask = exclude_jets(jet_mask,exclude_jet_mask)
    if exclude_jet_mask is not None and not exclude: jet_mask = jet_mask & exclude_jet_mask 
        
    event_mask = ak.sum(jet_mask,axis=-1) >= njetcut
    if exclude_events_mask is not None: event_mask = event_mask & exclude_events_mask
    return event_mask,jet_mask

