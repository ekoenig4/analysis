from .variable_tools import cache_variable
import awkward as ak
import numpy as np
from ..hepUtils import calc_deta, calc_dphi, calc_dr_p4, build_p4

@cache_variable(bins=(0,2000,30))
def jet_ht(t):
    return ak.sum(t.jet_pt, axis=1)

@cache_variable(bins=(-5,5,30))
def jet_deta(t):
    j1, j2 = ak.unzip(ak.combinations(t.jet_eta, 2))
    return calc_deta(j1, j2)

@cache_variable(bins=(-1, 1, 30))
def min_jet_deta(t):
    deta = jet_deta(t)
    return deta[ak.argmin(abs(deta), axis=1, keepdims=True)][:,0]

@cache_variable(bins=(-5,5,30))
def max_jet_deta(t):
    deta = jet_deta(t)
    return deta[ak.argmax(abs(deta), axis=1, keepdims=True)][:,0]

@cache_variable(bins=(0,5,30))
def jet_dr(t):
    j_p4 = build_p4(t, 'jet')
    j1, j2 = ak.unzip(ak.combinations(j_p4, 2))
    return calc_dr_p4(j1, j2)

@cache_variable(bins=(0,1.5,30))
def min_jet_dr(t):
    return ak.min(jet_dr(t), axis=1)

@cache_variable(bins=(0,5,30))
def max_jet_dr(t):
    return ak.max(jet_dr(t), axis=1)

@cache_variable(bins=(-4,4,30))
def h_deta(t):
    h1, h2 = ak.unzip(ak.combinations(t.h_eta, 2))
    return calc_deta(h1, h2)

@cache_variable(bins=(-4,4,30))
def h_dphi(t):
    h1, h2 = ak.unzip(ak.combinations(t.h_phi, 2))
    return calc_dphi(h1, h2)