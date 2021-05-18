import numpy as np
from tqdm import tqdm
import awkward as ak



def calc_scores(branches,mask,selected,signal_selected,sixb_selected,sixb_count,signal_sixb_count,sixb_gen_match_mask):
    
    mask_evnts = ak.sum(mask)
    signal_evnts= ak.sum(mask & sixb_gen_match_mask)
    
    total_eff = mask_evnts/float(nevents)
    signal_eff = signal_evnts/float(nsignal)
    purity = signal_evnts/float(mask_evnts)

    total_score = sum( ak.sum(sixb_count == n)/float(6+1-n) for n in range(6+1))/float(mask_evnts)
    signal_score= sum( ak.sum(signal_sixb_count == n)/float(6+1-n) for n in range(6+1))/float(signal_evnts)

    return {"total_eff":total_eff,"signal_eff":signal_eff,"purity":purity,"total_score":total_score,"signal_score":signal_score}
