import awkward as ak

from .. import weaverUtils
from ..ak_tools import ak_rank, ak_rand_like, get_collection, build_p4, ak_cumsum, ak_histogram
from ..classUtils.ObjIter import ParallelMethod
import numpy as np

def reconstruct(jet_p4, assignment, tag='', order='pt', higgs_only=False):
    if tag and not tag.endswith('_'): tag += '_'
    
    j_p4 = jet_p4[assignment]
    h_j1_arg, h_j2_arg = assignment[:, ::2], assignment[:, 1::2]
    j1_p4, j2_p4 = jet_p4[h_j1_arg], jet_p4[h_j2_arg]

    h_p4 = j1_p4 + j2_p4
    h_signalId = ak.where( j1_p4.signalId//2 == j2_p4.signalId//2, j1_p4.signalId//2, -1 )

    if not higgs_only:
        y_p4 = h_p4[:,1] + h_p4[:,2]
        # y_signalId = ak.where( (h_signalId[:,1]>0) == (h_signalId[:,2]>0), h_signalId[:,1]>0, -1 )
        y_signalId = ak.where( (h_signalId[:,1]>0) & (h_signalId[:,2]>0), 0, -1)

        x_p4 = h_p4[:,0] + y_p4
        # x_signalId = ak.where( (h_signalId[:,0]==0) == (y_signalId==1), h_signalId[:,0]>0, -1 )
        x_signalId = ak.where(y_signalId == h_signalId[:,0], y_signalId, -1)

    if order == 'pt':
        h_order = ak_rank(h_p4.pt, axis=1)
        j_order = ak_rank(j_p4.pt, axis=1)
    else:
        h_order = ak.argsort(ak_rand_like(h_p4.pt), axis=1)
        j_order = ak.argsort(ak_rand_like(j_p4.pt), axis=1)

    if not higgs_only:
        hy_h_order = h_order + np.array([[10,0,0]])
        hy_h_j_order = j_order + 10*hy_h_order[:,[0,0,1,1,2,2]]
    else:
        hy_h_order = h_order
        hy_h_j_order = j_order + 10*hy_h_order[:,[0,0,1,1,2,2]]

    j_order = ak.argsort(hy_h_j_order, axis=1, ascending=False)
    h_order = ak.argsort(hy_h_order, axis=1, ascending=False)

    j_p4 = j_p4[j_order]
    h_p4 = h_p4[h_order]
    h_signalId = h_signalId[h_order]

    p4vars = ['pt','eta','phi','m']

    if not higgs_only:
        return dict(
                **{f'{tag}x_{var}': getattr(x_p4, var) for var in p4vars},
                **{f'{tag}x_signalId': x_signalId},
                **{f'{tag}y_{var}': getattr(y_p4, var) for var in p4vars},
                **{f'{tag}y_signalId': y_signalId},
                **{f'{tag}h_{var}': getattr(h_p4, var) for var in p4vars},
                **{f'{tag}h_signalId': h_signalId},
                **{f'{tag}j_{var}': getattr(j_p4, var) for var in j_p4.fields},
            )
    else:
        return dict(
                **{f'{tag}h_{var}': getattr(h_p4, var) for var in p4vars},
                **{f'{tag}h_signalId': h_signalId},
                **{f'{tag}j_{var}': getattr(j_p4, var) for var in j_p4.fields},
            )


class f_load_x3h_feynnet(ParallelMethod):
    def __init__(self, model_path, onnxdir='onnx', batch_size=5000, accelerator='cpu'):
        super().__init__()

        self.model_path = model_path
        self.onnxdir = onnxdir
        self.batch_size = batch_size
        self.accelerator = accelerator

    def start(self, tree):
        jets = tree[[
            'jet_pt',
            'jet_ptRegressed',
            'jet_eta',
            'jet_phi',
            'jet_m',
            'jet_mRegressed',
            'jet_btag',
            'jet_signalId',
        ]]
        jets['jet_sinphi'] = np.sin(jets['jet_phi'])
        jets['jet_cosphi'] = np.cos(jets['jet_phi'])

        return dict(
            jets=jets,
        )

    def run(self, jets):
        import utils.weaverUtils as weaver
        import utils.sixbUtils as sixb

        jets = jets[ ak.argsort(-jets.jet_btag, axis=1) ]
        model = weaver.WeaverONNX(self.model_path, onnxdir=self.onnxdir, accelerator=self.accelerator)
        results = model.predict(jets, batch_size=self.batch_size)
        best_assignment = ak.from_regular(results['sorted_j_assignments'], axis=1)
        best_assignment = ak.values_astype(best_assignment, np.int32)

        rename = dict(jet_pt='jet_ptRegressed',jet_m='jet_mRegressed')
        jets = ak.zip(
            {
                field[4:]: jets[ rename.get(field, field) ] for field in jets.fields if field.startswith('jet_')
            }, with_name='Momentum4D'
        )
        return sixb.reconstruct(jets, best_assignment, higgs_only=True)

    def end(self, tree, **results):
        tree.extend(**results)

class f_load_feynnet_assignment(ParallelMethod):
    def __init__(self, model, extra=[], reco_event=True, try_base=False, try_year=True, onnx=False, onnx_sample=False, tag='', order='random'):
        super().__init__()
        self.model = model
        self.extra = extra
        self.reco_event = reco_event
        self.try_base = try_base
        self.try_year = try_year
        self.tag = tag
        self.order = order

        if onnx:
            self.start = self.start_onnx
            self.run = self.run_onnx
            self.onnx_sample = onnx_sample
    def start(self, tree):
        fields = ['max_comb','max_score','min_score'] + self.extra

        jet_p4 = build_p4(tree, prefix='jet', use_regressed=True, extra=['signalId', 'btag'])
        ranker = weaverUtils.load_output(tree, self.model, fields=fields, try_base=self.try_base, try_year=self.try_year)

        assert len(ranker['max_comb']) == len(jet_p4), f'Ranker output and jet collection have different lengths. Got {len(ranker["max_comb"])} and {len(jet_p4)} respectively.'

        return dict(
            jet_p4=jet_p4,
            ranker=ranker,
            extra=self.extra,
        )
    def start_onnx(self, tree):
        gen_masses = dict(
            gen_X_m=tree.gen_X_m,
            gen_Y_m=tree.gen_Y_m,
        ) if 'gen_X_m' in tree.fields else dict()

        return dict(
            jet_p4=build_p4(tree, prefix='jet', use_regressed=True, extra=['signalId', 'btag']),
            gen_masses=gen_masses,
            ranker=weaverUtils.WeaverONNX(self.model),
        )
    
    def run(self, jet_p4, ranker, extra):
        score, assignment, min_score = ranker['max_score'], ranker['max_comb'], ranker['min_score']
        assignment = ak.values_astype(ak.from_regular(assignment), "int64")
        reconstruction = reconstruct(jet_p4, assignment, tag=self.tag, order=self.order)
        return dict(
            feynnet_max_score=score,
            feynnet_min_score=min_score,
            **{f'feynnet_{field}':ak.from_regular(ranker[field]) for field in extra},
            **reconstruction,
        )
    
    def sample_ranks(self, ranks):
        probs = np.exp(-ranks) / ak.sum(np.exp(-ranks), axis=1)
        cumul = ak_cumsum(probs, axis=1)
        r = np.random.uniform(size=(len(ranks), 1))
        indices = ak.argmax(r < cumul, axis=1)
        return indices
    
    def run_onnx(self, jet_p4, gen_masses, ranker):
        tree = ak.zip(dict(
            jet_ptRegressed=jet_p4.pt,
            jet_mRegressed=jet_p4.mass,
            jet_eta=jet_p4.eta,
            jet_phi=jet_p4.phi,
            **gen_masses,
        ))
        results = ranker(tree, batch_size=5000)

        ranks = ak.from_regular(results['sorted_rank'], axis=1)
        combs = ak.from_regular(results['sorted_comb'], axis=1)

        is_inf = np.isinf(ranks)
        ranks = ranks[~is_inf]
        combs = combs[~is_inf]

        selected_comb = combs[:,0]
        assignment = ak.values_astype(ak.from_regular(selected_comb), "int64")
        reconstruction = reconstruct(jet_p4, assignment, tag=self.tag, order=self.order)
        return dict(
            feynnet_maxscore=ak.max(ranks, axis=1),
            feynnet_minscore=ak.min(ranks, axis=1),
            **reconstruction,
        )
    def end(self, tree, **output):
        tree.extend(**output)

quarklist = [
    'HX_b1','HX_b2','H1_b1','H1_b2','H2_b1','H2_b2',
]

higgslist = [
    'HX','H1','H2',
]

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
            f'{tag}Y_{field}': y[field]
            for field in y.fields
        },
        **{
            f'{tag}X_{field}': x[field]
            for field in x.fields
        }
    )

def load_true_assignment(tree, use_regressed=True, tag='true_'):
    jet_p4 = build_p4(tree, prefix='jet', use_regressed=use_regressed, extra=['signalId', 'btag'])

    true_assignment = ak.argsort(jet_p4.signalId, axis=1)[:,:6]
    true_reconstruction = reconstruct(jet_p4, true_assignment, tag=tag)
    tree.extend(**true_reconstruction)
def load_random_assignment(tree, tag=''):
    jet_p4 = build_p4(tree, prefix='jet', use_regressed=True, extra=['signalId', 'btag'])

    rand = ak_rand_like(jet_p4.pt)
    random_assignment = ak.argsort(rand, axis=1)[:,:6]
    random_reconstruction = reconstruct(jet_p4, random_assignment, tag=tag)
    tree.extend(**random_reconstruction)