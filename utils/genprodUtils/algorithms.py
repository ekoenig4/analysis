import awkward as ak
import numpy as np

def cluster_next_jet(genparts, dr=0.4):
    seed_index = ak.argmax(genparts.e, axis=1, keepdims=True)
    seed_part = genparts[seed_index][:,0]
    
    dr_mask = genparts.deltaR(seed_part) < dr
    
    jet_cluster = genparts[dr_mask]
    jet_cluster = ak.zip(dict(
        px=ak.sum(jet_cluster.px, axis=1),
        py=ak.sum(jet_cluster.py, axis=1),
        pz=ak.sum(jet_cluster.pz, axis=1),
        E=ak.sum(jet_cluster.e, axis=1),
    ), with_name='Momentum4D')
    
    return jet_cluster, genparts[~dr_mask]

def cluster_jets(genparts, dr=0.4):
    jets = []
    while ak.any(ak.num(genparts) > 0):
        jet, genparts = cluster_next_jet(genparts, dr)
        jets.append(jet[:,None])
    
    jets = ak.concatenate(jets, axis=1)
    jets = jets[~ak.is_none(jets.pt, axis=1)]
    jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]

    return ak.zip(dict(
        pt=ak.fill_none(jets.pt, -999),
        eta=ak.fill_none(jets.eta, -999),
        phi=ak.fill_none(jets.phi, -999),
        m=ak.fill_none(jets.mass, -999),
    ), with_name='Momentum4D')

def gen_match_jets(jets, genobjs):
    n_jet = ak.num(jets, axis=1)

    jet_quark_dr = ak.concatenate([ jets.deltaR(obj) for obj in genobjs ], axis=1)
    jet_quark_index = ak.local_index(jet_quark_dr, axis=1)

    matched = []
    remaining = jet_quark_index > -1

    while ak.any(remaining):

        next_jet_quark_dr = ak.where(remaining, jet_quark_dr, 9999)
        mindr_index = ak.argmin(next_jet_quark_dr, axis=1)
        mindr = ak.min(next_jet_quark_dr, axis=1)

        matched.append( ak.zip(dict(index=mindr_index,dr=mindr))[:,None] )

        mindr_jet_index = mindr_index % n_jet
        mindr_quark_index = mindr_index // n_jet

        remaining = remaining & ( (jet_quark_index % n_jet) != mindr_jet_index ) & ( (jet_quark_index // n_jet) != mindr_quark_index )

    matched = ak.concatenate(matched, axis=1)
    matched['dr'] = ak.fill_none(matched.dr, 9999)

    matched = matched[matched.dr < 0.4]
    matched['jet_index'] = matched.index % n_jet
    matched['quark_index'] = matched.index // n_jet
    matched = matched[ak.argsort(matched.jet_index)]
    jet_index = ak.local_index(jets, axis=1)
    jet_signalId = - ak.ones_like(jet_index)
    offset = ak.concatenate([ak.zeros_like(n_jet[:1]), np.cumsum(n_jet)[:-1]])
    flat_signalId = ak.flatten(jet_signalId)
    flat_jet_index = ak.flatten(matched.jet_index + offset)
    flat_quark_index = ak.flatten(matched.quark_index)
    np.asarray(flat_signalId)[flat_jet_index] = flat_quark_index

    return ak.unflatten(flat_signalId, n_jet)