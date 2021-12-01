from tqdm import tqdm
import awkward as ak
import numpy as np
from ..utils import get_collection, join_fields, unzip_records
from ..selectUtils import calc_dr
import vector
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

feature_names = ['m', 'pt', 'eta', 'phi', 'btag']
target_names = ['is_signal']


def get_features(jets):
    return jets[feature_names]


def get_target(jets):
    j1, j2 = ak.unzip(ak.combinations(jets.signalId, 2))
    j1_id, j2_id = ak.unzip(ak.argcombinations(jets.signalId, 2))
    jet_pairs = ak.sort(ak.concatenate(
        [j1[:, :, None], j2[:, :, None]], axis=-1), axis=-1)
    id_diff = jet_pairs[:, :, 1] - jet_pairs[:, :, 0]
    j1_even = jet_pairs[:, :, 0] % 2

    is_higgs = ((id_diff == 1) & (j1_even == 0))
    j_ids = ak.local_index(jets.signalId, axis=-1)
    j1_id_higgs = ak.fill_none(ak.pad_none(j1_id[is_higgs], 3), -1)
    j2_id_higgs = ak.fill_none(ak.pad_none(j2_id[is_higgs], 3), -1)

    is_h1 = (j_ids == j1_id_higgs[:, 0][:,None]) | (j_ids == j2_id_higgs[:, 0][:,None])
    is_h2 = (j_ids == j1_id_higgs[:, 1][:,None]) | (j_ids == j2_id_higgs[:, 1][:,None])
    is_h3 = (j_ids == j1_id_higgs[:, 2][:,None]) | (j_ids == j2_id_higgs[:, 2][:,None])
    is_none = ~(is_h1 | is_h2 | is_h3)
    return ak.zip(dict(is_h1=is_h1, is_h2=is_h2, is_h3=is_h3, is_none=is_none))


def scale_features(features, scaler=MinMaxScaler):
    partions = ak.count(features[features.fields[0]], axis=-1)
    flatfeat = ak.concatenate([ak.flatten(array)[:, None] for array in unzip_records(
        features).values()], axis=-1).to_numpy()
    flatfeat_scaled = scaler().fit_transform(flatfeat)
    scaled_features = ak.zip({field: ak.unflatten(
        flatfeat_scaled[:, i], partions) for i, field in enumerate(features.fields)})
    return scaled_features


def reshape_features(features, target):
    nfeatures = len(features.fields)
    maxjets = ak.max(ak.count(features[features.fields[0]], axis=-1))
    features = ak.concatenate([features[field][:, :, None]
                               for field in features.fields], axis=-1)

    maskint = np.int64(ak.max(features) + 128)
    features = ak.fill_none(ak.pad_none(
        features, maxjets, axis=-2, clip=True), ak.Array(nfeatures*[maskint])).to_numpy()
    target = 1*ak.concatenate([target.is_h1[:, :, None],
                               target.is_h2[:, :, None],
                               target.is_h3[:, :, None],
                               target.is_none[:, :, None]
                               ], axis=-1)
    target = ak.fill_none(ak.pad_none(
        target, maxjets, clip=True, axis=-2), ak.Array([maskint, maskint, maskint, maskint])).to_numpy()
    return features, target, maxjets, nfeatures, maskint


def higgs_reshape(y, y_pred, maskint):
    h1_mask = y[:, :, 0] != maskint
    njets = np.sum(h1_mask, axis=-1)
    h1_pred_ak = ak.unflatten(y_pred[h1_mask][:, 0], njets)
    h1_ak = ak.unflatten(y[h1_mask][:, 0], njets)
    return h1_ak, h1_pred_ak
