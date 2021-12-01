from tqdm import tqdm
import awkward as ak
import numpy as np
from ..utils import get_collection, join_fields, unzip_records, reorder_collection
from ..selectUtils import calc_dr
import vector
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score
from ..plotUtils import *


feature_names = ['m', 'pt', 'eta', 'phi', 'btag']
target_names = ['is_signal']


def get_features(jets):
    return jets[feature_names]


def get_target(jets):
    is_signal = jets.signalId > -1
    return ak.zip(dict(is_signal=is_signal))


def scale_features(features, scaler=MinMaxScaler):
    partions = ak.count(features[features.fields[0]], axis=-1)
    flatfeat = ak.concatenate([ak.flatten(array)[:, None] for array in unzip_records(
        features).values()], axis=-1).to_numpy()
    flatfeat_scaled = scaler().fit_transform(flatfeat)
    scaled_features = ak.zip({field: ak.unflatten(
        flatfeat_scaled[:, i], partions) for i, field in enumerate(features.fields)})
    return scaled_features


def reshape_features(features, target, maxjets=None, maskint=None):
    nfeatures = len(features.fields)
    if maxjets is None:
        maxjets = ak.max(ak.count(features[features.fields[0]], axis=-1))
    features = ak.concatenate([features[field][:, :, None]
                               for field in features.fields], axis=-1)

    if maskint is None:
        maskint = np.int64(ak.max(features) + 128)
    features = ak.fill_none(ak.pad_none(
        features, maxjets, axis=-2, clip=True), ak.Array(nfeatures*[maskint])).to_numpy()
    target = 1*ak.concatenate([target[:, :, None],
                              ~target[:, :, None]], axis=-1)
    target = ak.fill_none(ak.pad_none(
        target, maxjets, clip=True, axis=-2), ak.Array([maskint, maskint])).to_numpy()
    return features, target, maxjets, nfeatures, maskint


def tree_features_target(tree):
    jets = get_collection(tree, 'jet', named=False)
    jets = reorder_collection(jets, ak.argsort(-jets.pt, axis=-1))
    features = get_features(jets)
    target = get_target(jets)
    scaled_features = scale_features(features)
    return scaled_features, target


def jet_reshape(y, y_pred, maskint):
    mask = y[:, :, 0] != maskint
    njets = np.sum(mask, axis=-1)
    y_pred_ak = ak.unflatten(y_pred[mask][:, 0], njets)
    y_ak = ak.unflatten(y[mask][:, 0], njets)
    return y_ak, y_pred_ak

def plot_roc(y_ak,y_pred_ak):
    top6_pred = ak.argsort(-y_pred_ak,axis=-1)[:,:6]

    pred_top6_tru,pred_top6_pred = y_ak[top6_pred],y_pred_ak[top6_pred]
    flat_pred_top6_tru,flat_pred_top6_pred = ak.flatten(pred_top6_tru),ak.flatten(pred_top6_pred)

    fpr, tpr, thresholds = roc_curve(flat_pred_top6_tru,flat_pred_top6_pred)
    auc = roc_auc_score(flat_pred_top6_tru,flat_pred_top6_pred)

    fig,ax = graph_simple(fpr,tpr,xlabel="Mistag Rate",ylabel="Signal Efficiency",title=f"AUC = {auc:0.3}",marker=None)
    return fpr, tpr, thresholds

