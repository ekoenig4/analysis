import vector
import awkward as ak
import glob, os

from ..ak_tools import *
from ..hepUtils import build_all_dijets
from ..combinatorics import combinations
from ..classUtils import ParallelMethod
from .. import weaverUtils

def reconstruct(jet_p4, assignment, tag=''):
    if tag and not tag.endswith('_'): tag += '_'

    j_p4 = jet_p4[assignment]

    h_j1_arg, h_j2_arg = assignment[:, ::2], assignment[:, 1::2]
    j1_p4, j2_p4 = jet_p4[h_j1_arg], jet_p4[h_j2_arg]

    h_p4 = j1_p4 + j2_p4
    h_signalId = ak.where( j1_p4.signalId//2 == j2_p4.signalId//2, j1_p4.signalId//2, -1 )

    h1_p4, h2_p4 = h_p4[:, ::2], h_p4[:, 1::2]
    h1_signalId, h2_signalId = h_signalId[:, ::2], h_signalId[:, 1::2]

    y_p4 = h1_p4 + h2_p4
    y_signalId = ak.where( h1_signalId//2 == h2_signalId//2, h1_signalId//2, -1 )

    x_p4 = y_p4[:,0] + y_p4[:,1]
    x_signalId = ak.where( y_signalId[:,0]//2 == y_signalId[:,1]//2, y_signalId[:,0]//2, -1 )

    y_pt_order = ak_rank(y_p4.pt, axis=1)
    h_pt_order = ak_rank(h_p4.pt, axis=1)
    j_pt_order = ak_rank(j_p4.pt, axis=1)

    y_h_pt_order = h_pt_order + 10*y_pt_order[:,[0,0,1,1]]
    y_h_j_pt_order = j_pt_order + 100*y_h_pt_order[:,[0,0,1,1,2,2,3,3]]

    j_order = ak.argsort(y_h_j_pt_order, axis=1, ascending=False)
    h_order = ak.argsort(y_h_pt_order, axis=1, ascending=False)
    y_order = ak.argsort(y_pt_order, axis=1, ascending=False)
    
    j_p4 = j_p4[j_order]
    h_p4 = h_p4[h_order]
    y_p4 = y_p4[y_order]
    h_signalId = h_signalId[h_order]
    y_signalId = y_signalId[y_order]

    p4vars = ['pt','eta','phi','m']
    return dict(
        **{f'{tag}x_{var}': getattr(x_p4, var) for var in p4vars},
        **{f'{tag}x_signalId': x_signalId},
        **{f'{tag}y_{var}': getattr(y_p4, var) for var in p4vars},
        **{f'{tag}y_signalId': y_signalId},
        **{f'{tag}h_{var}': getattr(h_p4, var) for var in p4vars},
        **{f'{tag}h_signalId': h_signalId},
        **{f'{tag}j_{var}': getattr(j_p4, var) for var in j_p4.fields},
    )


class f_load_feynnet_assignment(ParallelMethod):
    def __init__(self, model, extra=[], reco_event=True):
        super().__init__()
        self.model = model
        self.extra = extra
        self.reco_event = reco_event
    def start(self, tree):
        fields = ['maxcomb','maxscore','minscore'] + self.extra

        jet_p4 = build_p4(tree, prefix='jet', use_regressed=True, extra=['signalId', 'btag'])
        ranker = weaverUtils.load_output(tree, self.model, fields=fields)

        assert len(ranker['maxcomb']) == len(jet_p4), f'Ranker output and jet collection have different lengths. Got {len(ranker["maxcomb"])} and {len(jet_p4)} respectively.'

        return dict(
            jet_p4=jet_p4,
            ranker=ranker,
            extra=self.extra,
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
    def end(self, tree, **output):
        tree.extend(**output)

# def load_feynnet_assignment(tree, model, extra=[], reco_event=True):
#     fields = ['maxcomb','maxscore','minscore'] + extra
#     ranker = load_weaver_output(tree, model, fields=fields)
    
#     score, assignment, minscore = ranker['maxscore'], ranker['maxcomb'], ranker['minscore']
#     tree.extend(feynnet_maxscore=score, feynnet_minscore=minscore, **{f'feynnet_{field}':ak.from_regular(ranker[field]) for field in extra})

#     if not reco_event: return

#     assignment = ak.from_regular(assignment.astype(int))

#     jet_p4 = build_p4(tree, prefix='jet', use_regressed=True, extra=['signalId', 'btag'])
#     reconstruction = reconstruct(jet_p4, assignment)
#     tree.extend(**reconstruction)

def load_true_assignment(tree):
    jet_p4 = build_p4(tree, prefix='jet', use_regressed=True, extra=['signalId', 'btag'])

    true_assignment = ak.argsort(jet_p4.signalId, axis=1)
    true_reconstruction = reconstruct(jet_p4, true_assignment, tag='true_')
    tree.extend(**true_reconstruction)

def load_random_assignment(tree, tag=''):
    jet_p4 = build_p4(tree, prefix='jet', use_regressed=True, extra=['signalId', 'btag'])

    random_assignment = ak.from_regular( np.stack([ np.random.permutation(8) for _ in range(len(tree)) ]) )
    random_reconstruction = reconstruct(jet_p4, random_assignment, tag=tag)
    tree.extend(**random_reconstruction)

from .reco_genobjs import quarklist, higgslist, ylist

def assign(tree, tag=''):
    if tag and not tag.endswith('_'): tag += '_'
    j = get_collection(tree, tag+'j', named=False)
    h = get_collection(tree, tag+'h', named=False)
    y = get_collection(tree, tag+'y', named=False)
    x = get_collection(tree, tag+'x', named=False)

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
        **{
            f'{tag}{Y}_{field}': y[field][:,i]
            for field in y.fields
            for i, Y in enumerate(ylist)
        },
        **{
            f'{tag}X_{field}': x[field]
            for field in x.fields
        }
    )