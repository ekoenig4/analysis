import awkward as ak

from ..ak_tools import get_collection,reorder_collection
from ..hepUtils import build_all_dijets
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

get_jetpt_wp = dict(
        loose =[75, 60, 45, 40, 30, 25], # 0.99^8 eff
        medium=[85, 65, 55, 45, 35, 30, 25], # 0.95^8 eff
        tight =[90, 75, 60, 50, 40, 30, 25] # 0.90^8 eff
    ).get
def selected_jet_pt(pts='loose'):
    pts = get_jetpt_wp(str(pts), pts)

    def jet_pt_filter(t):
        jet_pt = ak.sort(t.jet_pt, axis=-1, ascending=False)

        mask = ak.ones_like(jet_pt[:,0]) == 1
        for i, pt in enumerate(pts):
            mask = mask & (jet_pt[:,i] > pt)

        return mask
    return EventFilter('jet_pt_'+'_'.join(map(str,pts)), filter=jet_pt_filter)

def selected_jet_ptregressed(pts=[80,65,50,40,35,30,20,20]):
    def jet_pt_filter(t):
        jet_pt = ak.sort(t.jet_ptRegressed, axis=-1, ascending=False)

        mask = ak.ones_like(jet_pt[:,0]) == 1
        for i, pt in enumerate(pts):
            mask = mask & (jet_pt[:,i] > pt)

        return mask
    return EventFilter('jet_ptRegressed_'+'_'.join(map(str,pts)), filter=jet_pt_filter)

get_jetbtag_wps = dict(
        loose =[3, 3, 2, 2, 1], # 0.99^8 eff
        medium=[3, 3, 3, 2, 2, 1], # 0.95^8 eff
        tight =[3, 3, 3, 3, 2, 2, 1] # 0.85^8 eff
    ).get
def selected_jet_btagwp(btagwps='loose'):
    btagwps = get_jetbtag_wps(str(btagwps), btagwps)
    def jet_btagwp_filter(t):
        jet_btag = ak.sort(t.jet_btag, axis=-1, ascending=False)

        mask = ak.ones_like(jet_btag[:,0]) == 1
        for i, btagwp in enumerate(btagwps):
            if isinstance(btagwp, int):
                btagwp = jet_btagWP[btagwp]
            mask = mask & (jet_btag[:,i] > btagwp)

        return mask
    return EventFilter('jet_btagwp_'+'_'.join(map(str,btagwps)), filter=jet_btagwp_filter)