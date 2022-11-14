import awkward as ak

from ..utils import get_collection,reorder_collection
from ..selectUtils import build_all_dijets
from ..cutConfig import jet_btagWP
from ..classUtils.Filter import EventFilter

def select_topbtag(tree,njet=8):
    jets = get_collection(tree,'jet')
    jets = reorder_collection(jets,ak.argsort(-jets.jet_btag,axis=-1)[:,:njet])
    tree.extend(reorder_collection(jets,ak.argsort(-jets.jet_pt,axis=-1)))
    build_all_dijets(tree)
    
def select_toppt(tree,njet=8):
    jets = get_collection(tree,'jet')
    jets = reorder_collection(jets,ak.argsort(-jets.jet_pt,axis=-1)[:,:njet])
    tree.extend(reorder_collection(jets,ak.argsort(-jets.jet_pt,axis=-1)))
    build_all_dijets(tree)
        
def select_topbias(tree,njet=8):
    jets = get_collection(tree,'jet')
    
    btagwps = jet_btagWP + [1]

    jet_wps = [ jets[(wp_lo < jets.jet_btag) & (jets.jet_btag < wp_hi)] for wp_lo,wp_hi in zip(btagwps[:-1],btagwps[1:]) ]
    jet_wps = [ jet_wp[ak.argsort(-jet_wp.jet_pt,axis=-1)] for jet_wp in jet_wps[::-1] ]
    jets = ak.zip({ field:ak.concatenate([jet_wp[field] for jet_wp in jet_wps],axis=-1) for field in jets.fields })    
    jets = jets[:,:njet]
    tree.extend(reorder_collection(jets,ak.argsort(-jets.jet_pt,axis=-1)))
    build_all_dijets(tree)

def selected_jet_ptregressed(pts=[80,65,50,40,35,30,20,20]):
    def jet_pt_filter(t, pts=pts):
        jet_pt = ak.sort(t.jet_ptRegressed, axis=-1, ascending=False)

        mask = ak.ones_like(jet_pt[:,0]) == 1
        for i, pt in enumerate(pts):
            mask = mask & (jet_pt[:,i] > pt)

        return mask
    return EventFilter('jet_ptRegressed_'+'_'.join(map(str,pts)), filter=jet_pt_filter)
    
def selected_jet_btagwp(btagwps=[3,3,2,2,1]):
    def jet_btagwp_filter(t, btagwps=btagwps):
        jet_btag = ak.sort(t.jet_btag, axis=-1, ascending=False)

        mask = ak.ones_like(jet_btag[:,0]) == 1
        for i, btagwp in enumerate(btagwps):
            mask = mask & (jet_btag[:,i] > jet_btagWP[btagwp])

        return mask
    return EventFilter('jet_btagwp_'+'_'.join(map(str,btagwps)), filter=jet_btagwp_filter)