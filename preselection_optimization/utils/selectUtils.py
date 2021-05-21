#!/usr/bin/env python
# coding: utf-8

from . import *

# In[4]:


ordinal = lambda n : "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))
    
def get_jet_index_mask(branches,index):
    jet_index = ak.local_index( branches["jet_pt"] )
    compare , _ = ak.broadcast_arrays( index[:,None],jet_index )
    inter = (jet_index == compare)
    return ak.sum(inter,axis=-1) == 1

def exclude_jets(input_mask,exclude_mask):
    return (input_mask == True) & (exclude_mask == False)


# In[5]:


def sort_jets_mask_simple(branches,varbranch,jets=None,njets=6):
    if jets is None: jets = branches["jet_pt"] > -999
    sorted_array = np.argsort(-varbranch)
    selected_sorted_array = jets[sorted_array]
    firstN_array = ak.local_index(sorted_array[selected_sorted_array]) < njets
    selected_array = sorted_array[selected_sorted_array][firstN_array]
    return get_jet_index_mask(branches,selected_array)
def sort_jets_mask(branches,variable="jet_ptRegressed",jets=None,njets=6):
    varbranch = branches[variable]
    return sort_jets_mask_simple(branches,varbranch,jets,njets)


# In[6]:


def count_sixb_index(test_jet_index,sixb_jet_mask):
    nevts = ak.size(test_jet_index,axis=0)
    compare , _ = ak.broadcast_arrays( sixb_jet_mask[:,None], test_jet_index)
    inter = (test_jet_index == compare)
    count = ak.sum(ak.flatten(inter,axis=-1),axis=-1)
    return count
def count_sixb_mask(test_jet_mask,sixb_jet_mask):
    inter = test_jet_mask & sixb_jet_mask
    return ak.sum(inter,axis=-1)


# In[58]:


def calculate_values(branches,mask,nsixb_selected,nsixb_remaining,njets):
    mask_evnts = ak.sum(mask)
    signal_evnts= ak.sum(mask & branches.sixb_found_mask)
    
    total_eff = mask_evnts/float(branches.nevents)
    signal_eff = signal_evnts/float(branches.nsignal)
    purity = signal_evnts/float(mask_evnts)
    
    nsixb = min(6,njets)
    
    total_nsixb = ak.sum(branches["nfound_all"][mask])
    signal_nsixb = ak.sum(branches["nfound_all"][mask & branches.sixb_found_mask])
    
    total_remain = ak.sum(nsixb_remaining[mask])
    signal_remain = ak.sum(nsixb_remaining[mask & branches.sixb_found_mask])
    
    total_found = ak.sum(nsixb_selected[mask])
    signal_found = ak.sum(nsixb_selected[mask & branches.sixb_found_mask])
    
    return {
        "event_eff":total_eff,
        "event_avg_found":total_found/float(mask_evnts),
        "event_per_found":total_found/float(nsixb*mask_evnts),
        "event_total_found":total_found/float(total_nsixb),
        "signal_eff":signal_eff,
        "signal_avg_found":signal_found/float(signal_evnts),
        "signal_per_found":signal_found/float(nsixb*signal_evnts),
        "signal_total_found":signal_found/float(signal_nsixb),
        "purity":purity,
    }
def print_values(values):
    prompt_list = [
        "Event Efficiency:   {event_eff:0.2}",
        "Signal Efficiency:  {signal_eff:0.2f}",
        "Signal Purity:      {purity:0.2f}",
        "Event Avg Found:    {event_avg_found:0.2f} -> {event_per_found:0.2%}",
        "Signal Avg Found:   {signal_avg_found:0.2f} -> {signal_per_found:0.2%}",
        "Event Total Found:  {event_total_found:0.2%}",
        "Signal Total Found: {signal_total_found:0.2%}"
    ]
    prompt = '\n'.join(prompt_list)
    
    print(prompt.format(**values))


# In[9]:


def get_selected(branches,variable,jets=None,sixb_found_mask=None,sixb_jet_mask=None,sixb_removed=None,njets=6):
    if sixb_found_mask is None: sixb_found_mask = branches.sixb_found_mask
    
    selected = sort_jets_mask(branches,variable,jets=jets,njets=njets)
    nsixb_selected = count_sixb_mask(selected,sixb_jet_mask)
    
    sixb_remaining = exclude_jets(sixb_jet_mask,selected)
    nsixb_remaining = ak.sum(sixb_remaining,axis=-1)
    
    if sixb_removed is not None: sixb_jet_mask = exclude_jets(sixb_jet_mask,sixb_removed)
    sixb_selected = sort_jets_mask(branches,variable,jets=sixb_jet_mask & jets,njets=njets)
    
    return {
        "selected":selected,
        "nsixb_selected":nsixb_selected,
        "sixb_selected":sixb_selected,
        "sixb_remaining":sixb_remaining,
        "nsixb_remaining":nsixb_remaining
    }


# In[10]:


# --- Standard Preselection --- #
def std_preselection(branches,ptcut=30,etacut=2.4,btagcut=None,jetid=1,puid=1,njetcut=0,exclude_events_mask=None,exclude_jets_mask=None,**kwargs):
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
        
    if exclude_jets_mask is not None: jet_mask = exclude_jets(jet_mask,exclude_jets_mask)
    event_mask = ak.sum(jet_mask,axis=-1) >= njetcut
    if exclude_events_mask is not None: event_mask = event_mask & exclude_events_mask
    return event_mask,jet_mask


# In[11]:


def step_selection(branches,variable="jet_btag",njets=6,previous_selection=None,**cut_info):
    event_mask, jet_mask = std_preselection(branches,exclude_events_mask=previous_selection["mask"],exclude_jets_mask=previous_selection["selected"],**cut_info)
    selection_info = get_selected(branches,variable,jets=jet_mask,sixb_jet_mask=previous_selection["sixb_remaining"],
                                  sixb_removed=previous_selection["sixb_selected"],njets=njets)
    selection_info.update({"mask":event_mask,"jet_mask":jet_mask,"variable":variable,"njets":njets})
    return selection_info


# In[12]:


def iterative_selection(branches,selection_scheme,previous_info=None):
    # --- Initialize 0th Selection --- #
    previous_selection = {
        "mask":None,
        "selected":None,
        "sixb_remaining":branches.sixb_jet_mask,
        "sixb_selected":None
    }
    selection_step_info = []
    if previous_info is not None:
        selection_step_info = list(previous_info)
        previous_selection = selection_step_info[-1]
    
    for scheme in selection_scheme:
        
        order_info = { key:scheme[key] for key in ("variable","njets") }
        cut_info = { key:scheme[key] for key in scheme.keys() if key not in ("variable","njets") }
        
        selection_info = step_selection(branches,**order_info,previous_selection=previous_selection,**cut_info)
        selection_info["tag"] = scheme["tag"] if "tag" in scheme else None
        selection_step_info.append(selection_info)
        previous_selection = selection_info
        
    if len(selection_step_info) > 1:
    # --- Merge all selected jets --- #
        merged_selected = {}
    
        mask_to_merge = ("selected","sixb_selected","jet_mask")
        mask_to_add = ("nsixb_selected","njets")
        mask_to_copy = ("sixb_remaining","nsixb_remaining","mask","variable")
    
        for selection in selection_step_info:
            for key in selection.keys():
                if key in mask_to_merge:
                    if key not in merged_selected: merged_selected[key] = selection[key]
                    else: merged_selected[key] = merged_selected[key] | selection[key]
                if key in mask_to_add:
                    if key not in merged_selected: merged_selected[key] = selection[key]
                    else: merged_selected[key] = merged_selected[key] + selection[key]
        for key in mask_to_copy:
            merged_selected[key] = previous_selection[key]
        merged_selected["tag"] = "merged"
        
        selection_step_info.append(merged_selected)
    return selection_step_info

