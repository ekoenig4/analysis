from tqdm import tqdm
import awkward as ak
import numpy as np
from ..utils import get_collection, join_fields, unzip_records
from ..selectUtils import calc_dr
import vector


def build_dijet(jet_pair):
    p4_array = [vector.obj(pt=jet_pair.pt[:,i],eta=jet_pair.eta[:,i],phi=jet_pair.phi[:,i],m=jet_pair.m[:,i]) for i in range(2)]
    dijet_p4 = p4_array[0]+p4_array[1]
    return dijet_p4

def build_input(jet_pair):
    order = ak.argsort(-jet_pair.pt, axis=-1)
    jet_pair = jet_pair[order]
    dr = calc_dr(jet_pair.eta[:, 0], jet_pair.phi[:, 0],
                 jet_pair.eta[:, 1], jet_pair.phi[:, 1])[:, np.newaxis]
    features = ak.concatenate(
        [jet_pair.pt, jet_pair.eta, jet_pair.phi, jet_pair.btag, dr], axis=-1).to_numpy()

    signalId = ak.sort(jet_pair.signalId, axis=-1)
    target = ((signalId[:, 1]-signalId[:, 0]) == 1) & (signalId[:, 0] % 2 == 0)

    return features, target


def get_score(jet_pair, model):
    features, target = build_input(jet_pair)
    return model.predict(features)[:, 0][:, None], target[:, None]


def process_dijets(jets, model):
    dijet_pairs = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 3], [2, 4], [2, 5],
        [3, 4], [3, 5],
        [4, 5]
    ]

    pair_data = {'score':[],'pt':[],'target':[]}

    for pair in tqdm(dijet_pairs):
        pair_mask = ak.Array([pair]*len(jets))
        jet_pair = jets[pair_mask]
        score, target = get_score(jet_pair, model)
        dijet_p4 = build_dijet(jet_pair)

        pair_data['score'].append(score)
        pair_data['pt'].append(dijet_p4.pt)
        pair_data['target'].append(target)

    ndata = len(jets)
    for key in pair_data: pair_data[key] = ak.Array(np.array(pair_data[key]).reshape(ndata,15))

    dijet_record = ak.zip(pair_data)
    return dijet_record

def reorder_features(data,order):
    unzipped = unzip_records(data)
    nhiggs = unzipped.pop('nhiggs')
    reordered = ak.zip(unzipped)[order]
    return join_fields(reordered,nhiggs=nhiggs)

def prep_triH_data(data,reorder=True):
    if reorder:
        order = ak.argsort(data.score,axis=-1)
        data = data[order]
    nhiggs = ak.sum(data.target, axis=-1)
    return join_fields(data,nhiggs=nhiggs)


def process_triH(dijets):
    triH_combos = [[0,  9, 14], [0, 10, 13], [0, 11, 12],
                   [1,  6, 14], [1,  7, 13], [1,  8, 12],
                   [2,  5, 14], [2,  7, 11], [2,  8, 10],
                   [3,  5, 13], [3,  6, 11], [3,  8,  9],
                   [4,  5, 12], [4,  6, 10], [4,  7,  9]]
    triH_data = [prep_triH_data(dijets[:, combo]) for combo in triH_combos]

    fields = triH_data[0].fields
    unzipped = { field:ak.concatenate([data[field] for data in triH_data]) for field in fields }
    return ak.zip(unzipped, depth_limit=1)


def process(training, model, collection='jet'):
    jets = get_collection(training, collection, named=False)
    dijets = process_dijets(jets, model)
    triH = process_triH(dijets)
    return triH


def reshape_features(scores, nhiggs):
    ndata = len(nhiggs)//15
    scores = ak.Array(scores.to_numpy().reshape(ndata, 15, 3))
    nhiggs = ak.Array(nhiggs.to_numpy().reshape(ndata, 15))
    return ak.zip(dict(scores=scores, nhiggs=nhiggs), depth_limit=1)


def reshape_dataset(dataset):
    return reshape_features(dataset.scores, dataset.nhiggs)


def reshape(scores=None, nhiggs=None, dataset=None):
    if dataset is not None:
        return reshape_dataset(dataset)
    return reshape_features(scores, nhiggs)


def to_pandas(data):
    return ak.to_pandas(ak.zip({'score0': data.score[:, 0], 'score1': data.score[:, 1], 'score2': data.score[:, 2], 'nhiggs': data.nhiggs}))
