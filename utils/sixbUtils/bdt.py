import awkward as ak
import numpy as np

from ..variableUtils.variable_tools import cache_variable
from ..ak_tools import build_p4
from ..hepUtils import calc_dr, calc_deta, calc_dphi
from ..bdtUtils import ABCD

@cache_variable
def h_dm(t):
    return np.sqrt( np.sum( (t.h_m - 125)**2, axis=1 ))

@cache_variable
def btag6bavg(t):
    return ak.mean(t.j_btag, axis=1)

@cache_variable
def pt6bsum(t):
    return ak.sum(t.j_pt, axis=1)

@cache_variable
def dR6bmin(t):
    j1_eta, j2_eta = ak.unzip(ak.combinations(t.j_eta, 2))
    j1_phi, j2_phi = ak.unzip(ak.combinations(t.j_phi, 2))

    dr = calc_dr(j1_eta, j2_eta, j1_phi, j2_phi)
    return ak.min(dr, axis=1)

@cache_variable
def dEta6bmax(t):
    j1_eta, j2_eta = ak.unzip(ak.combinations(t.j_eta, 2))
    deta = calc_deta(j1_eta, j2_eta)
    argmax = ak.argmax( abs(deta), axis=1)
    return deta[argmax]

@cache_variable
def HX_dr(t):
    j_eta = t.j_eta[:,[0,1]]
    j_phi = t.j_phi[:,[0,1]]

    return calc_dr(j_eta[:,0], j_eta[:,1], j_phi[:,0], j_phi[:,1])

@cache_variable
def H1_dr(t):
    j_eta = t.j_eta[:,[2,3]]
    j_phi = t.j_phi[:,[2,3]]

    return calc_dr(j_eta[:,0], j_eta[:,1], j_phi[:,0], j_phi[:,1])

@cache_variable
def H2_dr(t):
    j_eta = t.j_eta[:,[4,5]]
    j_phi = t.j_phi[:,[4,5]]

    return calc_dr(j_eta[:,0], j_eta[:,1], j_phi[:,0], j_phi[:,1])

@cache_variable
def HX_H1_dEta(t):
    return calc_deta(t.HX_eta, t.H1_eta)

@cache_variable
def H1_H2_dEta(t):
    return calc_deta(t.H1_eta, t.H2_eta)

@cache_variable
def H2_HX_dEta(t):
    return calc_deta(t.H2_eta, t.HX_eta)

@cache_variable
def HX_H1_dPhi(t):
    return calc_dphi(t.HX_phi, t.H1_phi)

@cache_variable
def H1_H2_dPhi(t):
    return calc_dphi(t.H1_phi, t.H2_phi)

@cache_variable
def H1_costheta(t):
    p4 = build_p4(t, 'H1')
    return np.cos(p4.theta)

bdt_features = [
    pt6bsum, dR6bmin, dEta6bmax, 
    'HX_pt', 'H1_pt', 'H2_pt', 
    HX_dr, H1_dr, H2_dr, 
    'H1_m', 'H2_m', 
    HX_H1_dEta, H1_H2_dEta, H2_HX_dEta, 
    HX_H1_dPhi, H1_H2_dPhi, H1_costheta,
]

hm_cfg = dict(
    ARcenter = 125,
    SRedge = 25,
    CRedge = 50,
    VRedge = 75,
)

btag_cfg = 0.65

def get_ar_bdt():
    hparams = dict(
        n_estimators = 70,
        learning_rate = 0.15,
        max_depth = 3,
        min_samples_leaf = 275,
        gb_args={'subsample':0.6},
        n_folds=2,
        seed = 2020,
    )


    return ABCD(
        bdt_features,
        a = lambda t : (h_dm(t) <= hm_cfg['SRedge']) & (btag6bavg(t) >= btag_cfg),
        b = lambda t : (h_dm(t) <= hm_cfg['SRedge']) & (btag6bavg(t) <  btag_cfg),
        c = lambda t : (h_dm(t) >  hm_cfg['SRedge']) & (h_dm(t) <= hm_cfg['CRedge']) & (btag6bavg(t) >=  btag_cfg),
        d = lambda t : (h_dm(t) >  hm_cfg['SRedge']) & (h_dm(t) <= hm_cfg['CRedge']) & (btag6bavg(t) <  btag_cfg),
        **hparams
    )