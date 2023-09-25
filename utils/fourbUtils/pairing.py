import vector
import awkward0 as ak0
import awkward as ak
import glob, os

from ..ak_tools import *
from ..hepUtils import build_all_dijets
from ..combinatorics import combinations
from ..classUtils import ParallelMethod
from .. import weaverUtils

def reconstruct(jet_p4, assignment, tag='', order='pt'):
    if tag and not tag.endswith('_'): tag += '_'
    
    j_p4 = jet_p4[assignment]

    h_j1_arg, h_j2_arg = assignment[:, ::2], assignment[:, 1::2]
    j1_p4, j2_p4 = jet_p4[h_j1_arg], jet_p4[h_j2_arg]

    h_p4 = j1_p4 + j2_p4
    h_signalId = ak.where( j1_p4.signalId//2 == j2_p4.signalId//2, j1_p4.signalId//2, -1 )
    h1_signalId, h2_signalId = h_signalId[:, 0], h_signalId[:, 1]
    x_signalId = ak.where( h1_signalId//2 == h2_signalId//2, h1_signalId//2, -1 )

    if order == 'pt':
        h_pt_order = ak_rank(h_p4.pt, axis=1)
        j_pt_order = ak_rank(j_p4.pt, axis=1)
    else:
        h_pt_order = ak.argsort( ak_rand_like(h_p4.pt), axis=1 )
        j_pt_order = ak.argsort( ak_rand_like(j_p4.pt), axis=1 )
    
    h_j_pt_order = j_pt_order + 10*h_pt_order[:,[0,0,1,1]]
    
    j_order = ak.argsort(h_j_pt_order, axis=1, ascending=False)
    h_order = ak.argsort(h_pt_order, axis=1, ascending=False)

    j_p4 = j_p4[j_order]
    h_p4 = h_p4[h_order]
    h_signalId = h_signalId[h_order]

    p4vars = ['pt','eta','phi','m','mass']
    return dict(
        **{f'{tag}h_{var}': getattr(h_p4, var) for var in p4vars},
        **{f'{tag}h_signalId': h_signalId},
        **{f'{tag}j_{var}': getattr(j_p4, var) for var in j_p4.fields},
        x_signalId=x_signalId,
    )

quarklist = [
    'H1_b1','H1_b2','H2_b1','H2_b2',
]

higgslist = [
    'H1','H2',
]

def assign(tree, tag=''):
    if tag and not tag.endswith('_'): tag += '_'
    j = get_collection(tree, tag+'j', named=False)
    h = get_collection(tree, tag+'h', named=False)

    tree.extend(
        **{
            f'{tag}{J}_{field}': j[field][:,i]
            for field in j.fields
            for i, J in enumerate(quarklist)
        },
        **{
            f'{tag}{H}_{field}': h[field][:,i]
            for field in h.fields
            for i, H in enumerate(higgslist)
        },
    )


class f_load_feynnet_assignment(ParallelMethod):
    def __init__(self, model, extra=[], reco_event=True, onnx=False, onnxdir='export', prefix='jet', use_regressed=True):
        super().__init__()
        self.model = model
        self.extra = extra
        self.reco_event = reco_event
        self.prefix = prefix
        self.use_regressed = use_regressed

        if onnx:
            self.start = self.start_onnx
            self.run = self.run_onnx
            self.onnxdir = onnxdir
            
    def start(self, tree):
        fields = ['maxcomb','maxscore','minscore'] + self.extra
        return dict(
            ranker=weaverUtils.load_output(tree, self.model, fields=fields),
            extra=self.extra,
            jet_p4=build_p4(tree, prefix=self.prefix, use_regressed=self.use_regressed, extra=['signalId', 'btag']),
        )
    def start_onnx(self, tree):
        return dict(
            ranker=weaverUtils.WeaverONNX(self.model, self.onnxdir),
            jets=get_collection(tree, self.prefix),
        )
    
    def run(self, jet_p4, ranker, extra):
        score, assignment, minscore = ranker['maxscore'], ranker['maxcomb'], ranker['minscore']
        assignment = ak.values_astype(ak.from_regular(assignment), "int64")
        reconstruction = reconstruct(jet_p4, assignment)
        return dict(
            feynnet_maxscore=score,
            feynnet_minscore=minscore,
            **{f'feynnet_{field}':ak.from_regular(ranker[field]) for field in extra},
            **reconstruction,
        )
    def run_onnx(self, jets, ranker):
        results = ranker(jets, batch_size=5000)

        ranks = ak.from_regular(results['sorted_rank'], axis=1)
        ranks = ranks[~np.isinf(ranks)]

        combs = results.get('sorted_combs', results['sorted_j_assignments'])

        max_rank = ak.max(ranks, axis=1)
        assignment = combs[:,0]

        min_rank = ak.min(ranks, axis=1)
        assignment = ak.values_astype(ak.from_regular(assignment), "int64")

        jet_p4 = build_p4(jets, prefix=self.prefix, use_regressed=self.use_regressed, extra=['signalId', 'btag'])
        reconstruction = reconstruct(jet_p4, assignment)

        result = dict(
            feynnet_maxscore=max_rank,
            feynnet_minscore=min_rank,
            **reconstruction,
        )

        if 'class_probs' in results:
            result['feynnet_class_probs'] = results['class_probs']

        return result

    def end(self, tree, **output):
        tree.extend(**output)

def load_true_assignment(tree, use_regressed=True, tag='true_'):
    jet_p4 = build_p4(tree, prefix='jet', use_regressed=use_regressed, extra=['signalId', 'btag'])

    true_assignment = ak.argsort(-jet_p4.signalId, axis=1)[:,:4]
    true_reconstruction = reconstruct(jet_p4, true_assignment, tag=tag)
    tree.extend(**true_reconstruction)

def load_random_assignment(tree, tag=''):
    jet_p4 = build_p4(tree, prefix='jet', use_regressed=True, extra=['signalId', 'btag'])

    rand = ak_rand_like(jet_p4.pt)
    random_assignment = ak.argsort(rand, axis=1)[:,:4]
    random_reconstruction = reconstruct(jet_p4, random_assignment, tag=tag)
    tree.extend(**random_reconstruction)