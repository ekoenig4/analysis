from tqdm import tqdm
import awkward as ak
import numpy as np
from ..utils import get_collection
from ..selectUtils import calc_dr


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

    dijet_pair_scores = []
    dijet_pair_target = []

    for pair in tqdm(dijet_pairs):
        pair_mask = ak.Array([pair]*len(jets))
        jet_pair = jets[pair_mask]
        score, target = get_score(jet_pair, model)
        dijet_pair_scores.append(score)
        dijet_pair_target.append(target)
    dijet_record = ak.zip({'score': ak.concatenate(
        dijet_pair_scores, axis=-1), 'target': ak.concatenate(dijet_pair_target, axis=-1)})
    return dijet_record


def prep_triH_data(data):
    scores = ak.sort(data.score, axis=-1)
    nhiggs = ak.sum(data.target, axis=-1)
    return ak.zip({'score': scores, 'nhiggs': nhiggs}, depth_limit=1)


def process_triH(dijets):
    triH_combos = [[0,  9, 14], [0, 10, 13], [0, 11, 12],
                   [1,  6, 14], [1,  7, 13], [1,  8, 12],
                   [2,  5, 14], [2,  7, 11], [2,  8, 10],
                   [3,  5, 13], [3,  6, 11], [3,  8,  9],
                   [4,  5, 12], [4,  6, 10], [4,  7,  9]]
    triH_data = [prep_triH_data(dijets[:, combo]) for combo in triH_combos]

    scores = ak.concatenate([data.score for data in triH_data])
    nhiggs = ak.concatenate([data.nhiggs for data in triH_data])
    return ak.zip(dict(scores=scores, nhiggs=nhiggs), depth_limit=1)


def process(training, model):
    jets = get_collection(training, 'jet', named=False)
    dijets = process_dijets(jets, model)
    triH = process_triH(dijets)
    return triH


def to_pandas(data):
    return ak.to_pandas(ak.zip({'score0': data.scores[:, 0], 'score1': data.scores[:, 1], 'score2': data.scores[:, 2], 'nhiggs': data.nhiggs}))
