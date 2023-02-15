#!/usr/bin/env python
# coding: utf-8

import awkward as ak
import numpy as np
import vector
import scipy

from .ak_tools import *


def calc_dphi(phi_1, phi_2):
    dphi = phi_2 - phi_1
    dphi = ak.where(dphi >= np.pi, dphi - 2.0*np.pi, dphi)
    dphi = ak.where(dphi < -np.pi, dphi + 2.0*np.pi, dphi)
    return dphi


def calc_deta(eta_1, eta_2):
    return eta_2 - eta_1


def calc_dr(eta_1, phi_1, eta_2, phi_2):
    deta = calc_deta(eta_1, eta_2)
    dphi = calc_dphi(phi_1, phi_2)
    dr = np.sqrt(deta**2 + dphi**2)
    return dr


def calc_dr_p4(a_p4, b_p4):
    return calc_dr(a_p4.eta, a_p4.phi, b_p4.eta, b_p4.phi)


def get_ext_dr(eta_1, phi_1, eta_2, phi_2):
    dr = calc_dr(eta_1, phi_1, eta_2, phi_2)
    dr_index = ak.local_index(dr, axis=-1)

    dr_index = dr_index[dr != 0]
    dr_reduced = dr[dr != 0]

    imin_dr = ak.argmin(dr_reduced, axis=-1, keepdims=True)
    min_dr = ak.flatten(dr_reduced[imin_dr], axis=-1)
    imin_dr = ak.flatten(dr_index[imin_dr], axis=-1)

    imax_dr = ak.argmax(dr_reduced, axis=-1, keepdims=True)
    max_dr = ak.flatten(dr_reduced[imax_dr], axis=-1)
    imax_dr = ak.flatten(dr_index[imax_dr], axis=-1)
    return dr, min_dr, imin_dr, max_dr, imax_dr


def com_boost_vector(jet_pt, jet_eta, jet_phi, jet_m, njet=-1):
    """
    Calculate the COM boost vector for the top njets
    """
    if njet == -1:
        njet = ak.max(ak.count(jet_pt, axis=-1))

    def fill_zero(arr): return ak.fill_none(ak.pad_none(arr, njet, axis=-1), 0)

    jet_pt = fill_zero(jet_pt)
    jet_eta = fill_zero(jet_eta)
    jet_phi = fill_zero(jet_phi)
    jet_m = fill_zero(jet_m)

    jet_vectors = [vector.obj(pt=jet_pt[:, i], eta=jet_eta[:, i],
                              phi=jet_phi[:, i], m=jet_m[:, i]) for i in range(njet)]
    boost = jet_vectors[0]
    for jet_vector in jet_vectors[1:]:
        boost = boost + jet_vector

    return boost


def calc_y23(jet_pt):
    """
    measure of the third-jet pT relative to the summed transverse momenta
    of the two leading jets in a multi-jet event
    """
    ht2 = ak.sum(jet_pt[:, :2], axis=-1)
    pt3 = jet_pt[:, 3]
    y23 = pt3**2/ht2**2

    return dict(event_y23=y23)


def calc_momentum_tensor(jet_px, jet_py, jet_pz):
    trace = ak.sum(jet_px**2+jet_py**2+jet_pz**2, axis=-1)
    def Mij(jet_pi, jet_pj): return ak.sum(jet_pi*jet_pj, axis=-1)/trace

    a = Mij(jet_px, jet_px)
    b = Mij(jet_py, jet_py)
    c = Mij(jet_pz, jet_pz)
    d = Mij(jet_px, jet_py)
    e = Mij(jet_px, jet_pz)
    f = Mij(jet_py, jet_pz)

    m1 = ak.concatenate(
        [a[:, np.newaxis], d[:, np.newaxis], e[:, np.newaxis]], axis=-1)
    m2 = ak.concatenate(
        [d[:, np.newaxis], b[:, np.newaxis], f[:, np.newaxis]], axis=-1)
    m3 = ak.concatenate(
        [e[:, np.newaxis], f[:, np.newaxis], c[:, np.newaxis]], axis=-1)
    M = ak.to_numpy(ak.concatenate(
        [m1[:, np.newaxis], m2[:, np.newaxis], m3[:, np.newaxis]], axis=-2))
    return M


def calc_sphericity(jet_pt, jet_eta, jet_phi, jet_m, njet=-1):
    """
    Calculate sphericity/aplanarity in the COM frame of the top njets

    Sphericity: Measures how spherical the event is
    0 -> Spherical | 1 -> Collimated

    Aplanarity: Measures the amount of transverse momentum in or out of the jet plane
    """

    boost = com_boost_vector(jet_pt, jet_eta, jet_phi, jet_m, njet)
    boosted_jets = vector.obj(pt=jet_pt, eta=jet_eta,
                              phi=jet_phi, m=jet_m).boost_p4(-boost)
    jet_px, jet_py, jet_pz = boosted_jets.px, boosted_jets.py, boosted_jets.pz

    M = calc_momentum_tensor(jet_px, jet_py, jet_pz)

    eig_w, eig_v = np.linalg.eig(M)
    # make sure we only look at the absolue magnitude of values
    eig_w = np.abs(eig_w)
    # make sure eigenvalues are normalized
    eig_w = eig_w/np.sum(eig_w, axis=-1)[:, np.newaxis]
    eig_w = ak.sort(eig_w)

    eig_w1 = eig_w[:, 2]
    eig_w2 = eig_w[:, 1]
    eig_w3 = eig_w[:, 0]

    S = 3/2 * (eig_w2+eig_w3)
    St = 2 * eig_w2 / (eig_w1 + eig_w2)
    A = 3/2 * eig_w3
    F = eig_w2/eig_w1

    return dict(M_eig_w1=eig_w1, M_eig_w2=eig_w2, M_eig_w3=eig_w3, sphericity=S, sphericity_t=St, aplanarity=A, F=F)


def find_thrust_phi(jet_px, jet_py, tol=1e-05, niter=10, gr=(1+np.sqrt(5))/2):
    """
    Maximizing thrust via golden-selection search
    """

    jet_ones = ak.ones_like(jet_px)

    a = -(np.pi/2)*jet_ones
    b = (np.pi/2)*jet_ones

    c = b-(b-a)/gr
    d = a+(b-a)/gr

    def f(phi): return -ak.sum(np.abs(jet_px *
                                      np.cos(phi)+jet_py*np.sin(phi)), axis=-1)

    it = 0
    while ak.all(np.abs(b - a) > tol) and it < niter:
        is_c_low = f(c) < f(d)
        is_d_low = is_c_low == False

        b = b*(is_d_low) + d*(is_c_low)
        a = c*(is_d_low) + a*(is_c_low)

        c = b-(b-a)/gr
        d = a+(b-a)/gr

        it += 1

    return (b[:, 0]+a[:, 0])/2


def calc_thrust(jet_pt, jet_eta, jet_phi, jet_m):
    """
    The total thrust of the jets in the event
    """

    boost = com_boost_vector(jet_pt, jet_eta, jet_phi, jet_m)
    boosted_jets = vector.obj(pt=jet_pt, eta=jet_eta,
                              phi=jet_phi, m=jet_m).boost_p4(-boost)
    jet_pt, jet_eta, jet_phi, jet_m = boosted_jets.pt, boosted_jets.eta, boosted_jets.phi, boosted_jets.m

    jet_ht = ak.sum(jet_pt, axis=-1)
    jet_vectors = vector.obj(pt=jet_pt, eta=jet_eta, phi=jet_phi, m=jet_m)
    jet_px, jet_py = jet_vectors.px, jet_vectors.py

    thrust_phi = find_thrust_phi(jet_px, jet_py)
    Tt = 1 - ak.sum(np.abs(jet_px*np.cos(thrust_phi)+jet_py *
                    np.sin(thrust_phi)), axis=-1)/jet_ht
    Tm = ak.sum(np.abs(jet_px*np.sin(thrust_phi)-jet_py *
                np.cos(thrust_phi)), axis=-1)/jet_ht

    return dict(thrust_phi=thrust_phi, thrust_t=Tt, thrust_axis=Tm)


def calc_asymmetry(jet_pt, jet_eta, jet_phi, jet_m, njet=-1):
    """
    Calculate the asymmetry of the top njets in their COM frame
    """

    boost = com_boost_vector(jet_pt, jet_eta, jet_phi, jet_m, njet)
    boosted_jets = vector.obj(pt=jet_pt, eta=jet_eta,
                              phi=jet_phi, m=jet_m)  # .boost_p4(-boost)
    jet_px, jet_py, jet_pz = boosted_jets.px, boosted_jets.py, boosted_jets.pz

    jet_p = np.sqrt(jet_px**2+jet_py**2+jet_pz**2)

    AL = ak.sum(jet_pz, axis=-1)/ak.sum(jet_p, axis=-1)

    return dict(asymmetry=AL)


def optimize_var_cut(selections, variable, varmin=None, varmax=None, method=min, plot=False):
    from .plotUtils import graph_simple
    varmin = min([ak.min(selection[variable])
                 for selection in selections]) if varmin == None else varmin
    varmax = max([ak.max(selection[variable])
                 for selection in selections]) if varmax == None else varmax

    if method is min:
        def method(arr, cut): return arr < cut
    elif method is max:
        def method(arr, cut): return arr > cut

    def function(cut):
        nevents = [ak.sum(selection["scale"][method(selection[variable], cut)])
                   for selection in selections]
        bkg_eff = sum(nevents[1:])/sum([ak.sum(selection["scale"])
                                        for selection in selections[1:]])
        bovers = sum(nevents[1:])/nevents[0] if nevents[0] != 0 else 0
        score = bovers*bkg_eff
        return -score

    if plot:

        x = np.linspace(varmin, varmax, 100)
        y = np.vectorize(function)(x)

        graph_simple(x, -y, xlabel=f'{variable} cut',
                     ylabel="b/s*b_eff", marker=None)

    f_min = scipy.optimize.fmin(function, (varmax+varmin)/2)
    return f_min


def build_all_dijets(tree, pairs=None, ordered=None, name='dijet'):
    """Build all possible dijet pairings with the jet collection

    Args:
        tree (Tree): Tree class object

    Returns:
        awkward.Record: Awkward collection of dijets
    """
    jets = get_collection(tree, 'jet', False)[
        ['mRegressed', 'ptRegressed', 'eta', 'phi', 'signalId', 'btag']]
    jets = join_fields(jets, idx=ak.local_index(jets.ptRegressed, axis=-1))

    if pairs is None:
        pairs = ak.unzip(ak.combinations(jets.idx, 2))
    else:
        pairs = pairs[:, :, 0], pairs[:, :, 1]

    j1, j2 = jets[pairs[0]], jets[pairs[1]]

    j1_id, j2_id = j1.signalId, j2.signalId
    j1_id = (j1_id+2)//2
    j2_id = (j2_id+2)//2
    h_id = ak.where(j1_id == j2_id, j1_id, 0) - 1

    j1_p4 = build_p4(j1, use_regressed=True)
    j2_p4 = build_p4(j2, use_regressed=True)

    dphi = calc_dphi(j1_p4.phi, j2_p4.phi)
    deta = calc_deta(j1_p4.eta, j2_p4.eta)
    dr = np.sqrt(deta**2 + dphi**2)

    dijet = j1_p4 + j2_p4

    dijet = dict(
        m=dijet.m,
        dm=np.abs(dijet.m-125),
        pt=dijet.pt,
        eta=dijet.eta,
        phi=dijet.phi,
        jet_deta=deta,
        jet_dphi=dphi,
        jet_dr=dr,
        btagsum=j1.btag+j2.btag,
        signalId=h_id,
        j1Idx=j1.idx,
        j2Idx=j2.idx,
        localId=ak.local_index(dijet.m, axis=-1)
    )

    if ordered and ordered in dijet:
        order = ak.argsort(-dijet[ordered], axis=-1)
        dijet = {
            key: value[order]
            for key, value in dijet.items()
        }

    tree.extend(
        **{
            f'{name}_{key}': value
            for key, value in dijet.items()
        }
    )


def X_m_hm_res_corrected(t, higgs='higgs', nhiggs=4):
    """Correct X_m by subtracting off the measured higgs mass and adding the nominal mass (125)

    Args:
        t (Tree): Tree to operate on
        higgs (str, list, optional): collection to use for higgs. Can be a list of higgs ["H1","H2",...], or just a string "higgs" refering to a list of higgs. Defaults to 'higgs'.
    """
    if isinstance(higgs, str):
        higgs_m = t[f'{higgs}_m']
    elif isinstance(higgs, list):
        higgs_m = ak_stack([t[f'{h}_m'] for h in higgs])

    higgs_m = higgs_m[:, :nhiggs]
    higgs_m_res = higgs_m - 125
    higgs_m_res_sum = ak.sum(higgs_m_res, axis=-1)
    t.extend(
        higgs_m_res_sum=higgs_m_res_sum,
        X_m_hm_res_corr=t.X_m - higgs_m_res_sum
    )


def X_m_hp4_res_corrected(t, higgs='higgs', nhiggs=4):
    """Correct X_m by scaling the measured p4 of the higgs to have the nominal mass(125)

    Args:
        t (Tree): Tree to operate on
        higgs (str, list, optional): collection to use for higgs. Can be a list of higgs ["H1","H2",...], or just a string "higgs" refering to a list of higgs. Defaults to 'higgs'.
    """
    if isinstance(higgs, str):
        higgs_p4 = build_p4(t, prefix=higgs)
    elif isinstance(higgs, list):
        higgs_p4 = ak_stack([build_p4(t, prefix=h) for h in higgs])

    higgs_p4 = higgs_p4[:, :nhiggs]

    higgs_p4_corr = (125/higgs_p4.m)*higgs_p4
    X_p4_corr = higgs_p4_corr[:, 0]
    for i in range(1, 4):
        X_p4_corr = X_p4_corr + higgs_p4_corr[:, i]

    t.extend(
        **{
            f'X_{var}_hp4_res_corr': getattr(X_p4_corr, var)
            for var in ('pt', 'm', 'eta', 'phi')
        }
    )

    if isinstance(higgs, str):
        t.extend(
            **{
                f'higgs_{var}_corr': getattr(higgs_p4_corr, var)
                for var in ('pt', 'm', 'eta', 'phi')
            }
        )
    elif isinstance(higgs, list):
        t.extend(
            **{
                f'{h}_{var}_corr': getattr(higgs_p4_corr, var)[:, i]
                for var in ('pt', 'm', 'eta', 'phi')
                for i, h in enumerate(higgs)
            }
        )
