import numpy as np
from tqdm import tqdm
import awkward as ak

def select_jets(branches,variable="jet_ptRegressed",mask=None,jets=None,njets=6):
    if mask is None: mask = np.ones(ak.size(branches['Run']),dtype=bool)
    if jets is None: jets = branches["jet_pt"] > -999
    sorted_array = np.argsort(-branches[variable][mask])
    selected_sorted_array = jets[mask][sorted_array]
    firstN_array = np.argsort(sorted_array[selected_sorted_array]) < njets
    selected_array = sorted_array[selected_sorted_array][firstN_array]
    return selected_array

def select_sixb(branches,mask=None):
    if mask is None: mask = np.ones(ak.size(branches['Run']),dtype=bool)
    b_list = ("HX_b1","HX_b2","HY1_b1","HY1_b2","HY2_b1","HY2_b2")
    sixb_index = []
    for event in tqdm(branches[mask],total=ak.sum(mask)):
        sixb_index.append( [event[f"gen_{b}_recojet_index"] for b in b_list if event[f"gen_{b}_recojet_index"] > -1 ] )
    return ak.Array(sixb_index)

def count_sixb(test_selected,sixb_selected,mask=None):
    nevts = ak.size(test_selected,axis=0)
    if mask is None: mask = np.ones(nevts,dtype=bool)
    sixb_selected = sixb_selected[mask]
    return ak.Array( len(set(test).intersection(sixb)) for test,sixb in tqdm(zip(test_selected,sixb_selected),total=nevts) )

def get_selected(branches,variable,mask=None,jets=None,sixb_selected=None,sixb_gen_match_mask=None):
    if mask is None: mask = np.ones(ak.size(branches['Run']),dtype=bool)
    selected = select_jets(branches,variable,mask=mask,jets=jets)
    sixb_count = count_sixb(selected,sixb_selected,mask=mask)
    
    signal_selected = select_jets(branches,variable,mask=mask & sixb_gen_match_mask,jets=jets)
    signal_sixb_count = count_sixb(signal_selected,sixb_selected,mask=mask & sixb_gen_match_mask)
    
    return (selected,sixb_count),(signal_selected,signal_sixb_count)

def calc_scores(branches,mask,selected,signal_selected,sixb_selected,sixb_count,signal_sixb_count,sixb_gen_match_mask):
    
    mask_evnts = ak.sum(mask)
    signal_evnts= ak.sum(mask & sixb_gen_match_mask)
    
    total_eff = mask_evnts/float(nevents)
    signal_eff = signal_evnts/float(nsignal)
    purity = signal_evnts/float(mask_evnts)

    total_score = sum( ak.sum(sixb_count == n)/float(6+1-n) for n in range(6+1))/float(mask_evnts)
    signal_score= sum( ak.sum(signal_sixb_count == n)/float(6+1-n) for n in range(6+1))/float(signal_evnts)

    return {"total_eff":total_eff,"signal_eff":signal_eff,"purity":purity,"total_score":total_score,"signal_score":signal_score}
