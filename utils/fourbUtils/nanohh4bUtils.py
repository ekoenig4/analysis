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

    if not 'ak4_regpt' in jet.fields:
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

    h_pt_order = ak_rank(higgs.regpt, axis=1)
    j_pt_order = ak_rank(jets.regpt, axis=1)

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

    arrays['_higgs_order_'] = h_order
    arrays['_jet_order_'] = j_order

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
    def __init__(self, model_path, onnxdir='onnx', batch_size=5000, accelerator='cpu'):
        super().__init__()

        self.model_path = model_path
        self.onnxdir = onnxdir
        self.batch_size = batch_size
        self.accelerator = accelerator

        self.start = {
            'onnx':self.start_onnx,
            'predict':self.start_predict,
        }.get(onnxdir)

        self.run = {
            'onnx':self.run_onnx,
            'predict':self.run_predict,
        }.get(onnxdir)

        try:
            self.metadata = weaver.WeaverONNX(self.model_path, onnxdir=self.onnxdir).metadata
        except:
            self.metadata = {'invalid':'invalid'}

    def start_onnx(self, tree):
        jets = get_ak4_jets(tree)
        jets['ak4_log_pt'] = np.log(jets['ak4_pt'])
        jets['ak4_sinphi'] = np.sin(jets['ak4_phi'])
        jets['ak4_cosphi'] = np.cos(jets['ak4_phi'])

        return dict(
            jets=jets,
        )

    def run_onnx(self, jets):
        jets = jets[ ak.argsort(-jets.ak4_bdisc, axis=1) ]
        model = weaver.WeaverONNX(self.model_path, onnxdir=self.onnxdir, accelerator=self.accelerator)
        results = model.predict(jets, batch_size=self.batch_size)
        best_assignment = ak.from_regular(results['sorted_j_assignments'], axis=1)
        best_assignment = ak.values_astype(best_assignment, np.int32)
        return reconstruct(jets, best_assignment)

    def start_predict(self, tree):
        jets = get_ak4_jets(tree)
        filelist = [ fn.true_fname for fn in tree.filelist ]
        return dict(
            jets=jets,
            filelist=filelist,
        )
    
    def run_predict(self, jets, filelist):
        jets = jets[ak.argsort(-jets.ak4_bdisc, axis=1)]
        results = weaver.load_predict_filelist(filelist, self.model_path, fields=['sorted_j_assignments'])
        best_assignment = ak.from_regular(results['sorted_j_assignments'], axis=1)
        best_assignment = ak.values_astype(best_assignment, np.int32)
        return reconstruct(jets, best_assignment)

    def end(self, tree, **results):
        tree.extend(**results)

class f_evaluate_spanet(ParallelMethod):
    def __init__(self, model_path, onnxdir=''):
        super().__init__()

        self.model_path = model_path
        self.onnxdir = onnxdir

    def start(self, tree):
        jets = get_ak4_jets(tree)

        return dict(
            jets=jets,
        )

    def get_reconstruction(self, jets, h1_assignment_probability, h2_assignment_probability, **results):
        _, nj1, nj2 = h1_assignment_probability.shape
        h1_assignment_probability = h1_assignment_probability.reshape(-1, nj1 * nj2)
        h2_assignment_probability = h2_assignment_probability.reshape(-1, nj1 * nj2)

        h1_maxprob, h2_maxprob = np.max(h1_assignment_probability, axis=1), np.max(h2_assignment_probability, axis=1)
        max_mask = h1_maxprob >= h2_maxprob
        leading_pair = np.where(max_mask, np.argmax(h1_assignment_probability, axis=1), np.argmax(h2_assignment_probability, axis=1))
        h1_j1_index = leading_pair // nj1
        h1_j2_index = leading_pair % nj2
        index = ak.local_index(h1_assignment_probability, axis=1).to_numpy()
        mask_pair = ( index // nj1 == h1_j1_index ) & ( index % nj2 == h1_j2_index )
        next_leading_pair = np.where(max_mask, np.argmax( np.where(mask_pair, 0, h2_assignment_probability) , axis=1), np.argmax( np.where(mask_pair, 0, h1_assignment_probability), axis=1))
        h2_j1_index = next_leading_pair // nj1
        h2_j2_index = next_leading_pair % nj2
        assignment = ak.from_regular(np.stack([h1_j1_index, h1_j2_index, h2_j1_index, h2_j2_index], axis=1))
        return reconstruct(jets, assignment)
    
    def get_detection(self, h1_detection_probability, h2_detection_probability, _higgs_order_, **results):

        higgs_dp = ak.from_regular(np.stack([h1_detection_probability, h2_detection_probability], axis=1))
        higgs_dp = higgs_dp[_higgs_order_]

        return dict(
            dHH_H1_dp = higgs_dp[:,0],
            dHH_H2_dp = higgs_dp[:,1],
        )

    def run(self, jets):
        jets = jets[ak_rank(-jets.ak4_bdisc) < 6]
        jets['ak4_sinphi'] = np.sin(jets['ak4_phi'])
        jets['ak4_cosphi'] = np.cos(jets['ak4_phi'])
        jets['ak4_mask'] = ak.ones_like(jets.ak4_pt, dtype=bool)

        model = weaver.WeaverONNX(self.model_path, onnxdir=self.onnxdir)
        results = model.predict(jets)

        reconstruction = self.get_reconstruction(jets, **results)
        detection = self.get_detection(**results, **reconstruction)

        return dict(
            **reconstruction,
            **detection,
        )

    def end(self, tree, **results):
        tree.extend(**results)

def n_loose_btag(t):
    nL = t.ak4_h1b1_btag_L + t.ak4_h1b2_btag_L + t.ak4_h2b1_btag_L + t.ak4_h2b2_btag_L
    return ak.values_astype(nL, np.int32)

def n_medium_btag(t):
    nM = t.ak4_h1b1_btag_M + t.ak4_h1b2_btag_M + t.ak4_h2b1_btag_M + t.ak4_h2b2_btag_M
    return ak.values_astype(nM, np.int32)

def n_tight_btag(t):
    nT = t.ak4_h1b1_btag_T + t.ak4_h1b2_btag_T + t.ak4_h2b1_btag_T + t.ak4_h2b2_btag_T
    return ak.values_astype(nT, np.int32)