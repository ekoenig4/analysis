import awkward as ak

from ..utils import get_collection,reorder_collection
from ..selectUtils import build_all_dijets
from ..cutConfig import jet_btagWP

def select_topbtag(tree,njet=8):
    jets = get_collection(tree,'jet')
    jets = reorder_collection(jets,ak.argsort(-jets.jet_btag,axis=-1)[:,:njet])
    tree.extend(reorder_collection(jets,ak.argsort(-jets.jet_pt,axis=-1)))
    tree.extend(**build_all_dijets(tree))
    
def select_toppt(tree,njet=8):
    jets = get_collection(tree,'jet')
    jets = reorder_collection(jets,ak.argsort(-jets.jet_pt,axis=-1)[:,:njet])
    tree.extend(reorder_collection(jets,ak.argsort(-jets.jet_pt,axis=-1)))
    tree.extend(**build_all_dijets(tree))
        
def select_topbias(tree,njet=8):
    jets = get_collection(tree,'jet')
    
    btagwps = jet_btagWP + [1]

    jet_wps = [ jets[(wp_lo < jets.jet_btag) & (jets.jet_btag < wp_hi)] for wp_lo,wp_hi in zip(btagwps[:-1],btagwps[1:]) ]
    jet_wps = [ jet_wp[ak.argsort(-jet_wp.jet_pt,axis=-1)] for jet_wp in jet_wps[::-1] ]
    jets = ak.zip({ field:ak.concatenate([jet_wp[field] for jet_wp in jet_wps],axis=-1) for field in jets.fields })    
    jets = jets[:,:njet]
    tree.extend(reorder_collection(jets,ak.argsort(-jets.jet_pt,axis=-1)))
    tree.extend(**build_all_dijets(tree))