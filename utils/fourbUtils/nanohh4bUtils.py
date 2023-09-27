from ..ak_tools import build_p4, build_collection, get_collection, ak_rank
from ..hepUtils import calc_dr_p4, calc_deta, calc_dphi
from .. import weaverUtils as weaver

quarklist = [
    'h1b1','h1b2','h2b1','h2b2',
]

higgslist = [
    'H1','H2',
]

import awkward as ak
import numpy as np
import re

def match_ak4_gen(tree):
    build_collection(tree, 'genj_H\db\d', 'genjet_b')
    build_collection(tree, 'ak4_h\db\d', 'ak4jet_b')
    tree.extend(
        ak4jet_b_m=tree.ak4jet_b_mass,
    )

    genjet = build_p4(tree, 'genjet_b')
    ak4jet = build_p4(tree, 'ak4jet_b')
    ak4_gen_dr = calc_dr_p4(ak4jet, genjet[:,None])

    flat_ak4_gen_dr = ak.flatten(ak4_gen_dr, axis=2)
    flat_ak4_gen_dr_index = ak.local_index(flat_ak4_gen_dr, axis=1)
    flat_ak4_index = flat_ak4_gen_dr_index // 4
    flat_gen_index = flat_ak4_gen_dr_index % 4

    best_matches = []
    best_matches_dr = []

    flat_match_mask = ak.zeros_like(flat_ak4_gen_dr, dtype=bool)

    for i in range(4):
        next_flat_ak4_gen_dr = ak.where(flat_match_mask, np.inf, flat_ak4_gen_dr)
        next_index = ak.argmin( next_flat_ak4_gen_dr, axis=1)
        next_min_dr = ak.min( next_flat_ak4_gen_dr, axis=1)

        next_ak4_index = next_index // 4
        next_gen_index = next_index %  4

        best_matches.append( next_index )
        best_matches_dr.append( next_min_dr )

        is_next_ak4 = flat_ak4_index == next_ak4_index
        is_next_gen = flat_gen_index == next_gen_index
        is_next_ak4_gen = is_next_ak4 | is_next_gen

        flat_match_mask = flat_match_mask | is_next_ak4_gen

    best_matches = np.stack(best_matches, axis=1)
    best_matches_dr = np.stack(best_matches_dr, axis=1)
    best_matches_dr4_mask = best_matches_dr < 0.4
    ak4jet_ak4id = best_matches // 4

    ak4jet_genid = ak.where(best_matches_dr4_mask, best_matches % 4, -1)

    ak4jet_b_index = ak.argsort(ak4jet_ak4id, axis=1)
    ak4jet_b_genid = ak4jet_genid[ak4jet_b_index]
    ak4jet_b_gendr = best_matches_dr[ak4jet_b_index]

    ak4j1_b_hid = ak4jet_b_genid[:,::2] // 2
    ak4j2_b_hid = ak4jet_b_genid[:,1::2] // 2

    h_genid = ak.where( ak4j1_b_hid == ak4j2_b_hid, ak4j1_b_hid, -1)

    tree.extend(
        **{
            f'ak4_{q}_signalId' : ak4jet_b_genid[:,i]
            for i, q in enumerate(quarklist)
        },
        **{
            f'ak4_{q}_gendr' : ak4jet_b_gendr[:,i]
            for i, q in enumerate(quarklist)
        },

        dHH_H1_genid=h_genid[:,0],
        dHH_H2_genid=h_genid[:,1],
        
        nfound_select=ak.sum(ak4jet_b_genid >= 0, axis=1),
        nfound_paired=ak.sum(h_genid >= 0, axis=1),
    )


def get_ak4_jets(tree):
    ak4_fields = [
        field
        for field in tree.fields
        if re.match(r'ak4_.*', field) and not re.match(r'ak4_h\db\d_.*', field)
    ]

    jet = tree[ak4_fields]

    regp4 = jet.ak4_bRegCorr * ak.zip(dict(
        pt=jet.ak4_pt,
        eta=jet.ak4_eta,
        phi=jet.ak4_phi,
        mass=jet.ak4_mass,
    ), with_name='Momentum4D')

    jet['ak4_regpt'] = regp4.pt
    jet['ak4_regmass'] = regp4.mass

    return jet


def reconstruct(jets, assignment):
    jets = ak.zip({
        field.replace('ak4_', '') : jets[field]
        for field in jets.fields
    })
    jets = jets[assignment]

    j_p4 = ak.zip(dict(
        pt=jets.pt,
        eta=jets.eta,
        phi=jets.phi,
        mass=jets.mass,
    ), with_name='Momentum4D')

    j_regp4 = jets.bRegCorr * j_p4

    h_p4 = j_p4[:,::2] + j_p4[:,1::2]
    h_regp4 = j_regp4[:,::2] + j_regp4[:,1::2]

    higgs = ak.zip(dict(
        pt=h_p4.pt,
        eta=h_p4.eta,
        phi=h_p4.phi,
        mass=h_p4.mass,
        regpt=h_regp4.pt,
        regmass=h_regp4.mass,
    ))

    h_pt_order = ak_rank(higgs.pt, axis=1)
    j_pt_order = ak_rank(jets.pt, axis=1)

    h_j_pt_order = j_pt_order + 10*h_pt_order[:,[0,0,1,1]]

    j_order = ak.argsort(h_j_pt_order, axis=1, ascending=False)
    h_order = ak.argsort(h_pt_order, axis=1, ascending=False)

    jets = jets[j_order]
    higgs = higgs[h_order]

    quarks = {
        f'ak4_{q}_{field}' : jets[field][:,i]
        for i, q in enumerate(['h1b1', 'h1b2', 'h2b1', 'h2b2'])
        for field in jets.fields
    }
    higgs = {
        f'dHH_{h}_{field}' : higgs[field][:,i]
        for i, h in enumerate(['H1', 'H2'])
        for field in higgs.fields
    }
    arrays = dict(**quarks, **higgs)

    arrays['dHH_H1_H2_deltaEta'] = calc_deta(arrays['dHH_H1_eta'], arrays['dHH_H2_eta'])
    arrays['dHH_H1_H2_deltaPhi'] = calc_dphi(arrays['dHH_H1_phi'], arrays['dHH_H2_phi'])
    arrays['dHH_H1_H2_deltaR'] = np.sqrt(arrays['dHH_H1_H2_deltaEta']**2 + arrays['dHH_H1_H2_deltaPhi']**2)

    arrays['dHH_H1b1_H1b2_deltaEta'] = calc_deta(arrays['ak4_h1b1_eta'], arrays['ak4_h1b2_eta'])
    arrays['dHH_H1b1_H1b2_deltaPhi'] = calc_dphi(arrays['ak4_h1b1_phi'], arrays['ak4_h1b2_phi'])
    arrays['dHH_H1b1_H1b2_deltaR'] = np.sqrt(arrays['dHH_H1b1_H1b2_deltaEta']**2 + arrays['dHH_H1b1_H1b2_deltaPhi']**2)

    arrays['dHH_H2b1_H2b2_deltaEta'] = calc_deta(arrays['ak4_h2b1_eta'], arrays['ak4_h2b2_eta'])
    arrays['dHH_H2b1_H2b2_deltaPhi'] = calc_dphi(arrays['ak4_h2b1_phi'], arrays['ak4_h2b2_phi'])
    arrays['dHH_H2b1_H2b2_deltaR'] = np.sqrt(arrays['dHH_H2b1_H2b2_deltaEta']**2 + arrays['dHH_H2b1_H2b2_deltaPhi']**2)

    return arrays

from ..classUtils import ParallelMethod
class f_evaluate_feynnet(ParallelMethod):
    def __init__(self, model_path, onnxdir='onnx'):
        super().__init__()
        self.model = weaver.WeaverONNX(model_path, onnxdir=onnxdir)

    def start(self, tree):
        jets = get_ak4_jets(tree)
        jets['ak4_sinphi'] = np.sin(jets['ak4_phi'])
        jets['ak4_cosphi'] = np.cos(jets['ak4_phi'])

        return dict(
            jets=jets,
            model=self.model,
        )

    def run(self, jets, model):
        jets = jets[ak_rank(-jets.ak4_bdisc) < 4]

        results = model(jets)
        assignments = ak.from_regular(results['sorted_j_assignments'].astype(int), axis=2)
        best_assignment = assignments[:,0]
        return reconstruct(jets, best_assignment)

    def end(self, tree, **results):
        tree.extend(**results)