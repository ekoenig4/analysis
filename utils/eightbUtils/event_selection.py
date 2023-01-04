from ..classUtils import EventFilter, CollectionFilter
from ..cutConfig import jet_btagWP 
from ..eightbUtils import quarklist

import awkward as ak
import numpy as np

def calc_4h_asym(higgs_m):
    higgs_m = ak.to_numpy(higgs_m)
    # higgs_m.sort(axis=-1)
    hm12_asym = 2*np.abs(higgs_m[:,3]-higgs_m[:,2])/(higgs_m[:,3]+higgs_m[:,2])
    hm13_asym = 2*np.abs(higgs_m[:,3]-higgs_m[:,1])/(higgs_m[:,3]+higgs_m[:,1])
    hm14_asym = 2*np.abs(higgs_m[:,3]-higgs_m[:,0])/(higgs_m[:,3]+higgs_m[:,0])
    
    hm23_asym = 2*np.abs(higgs_m[:,2]-higgs_m[:,1])/(higgs_m[:,2]+higgs_m[:,1])
    hm24_asym = 2*np.abs(higgs_m[:,2]-higgs_m[:,0])/(higgs_m[:,2]+higgs_m[:,0])
    
    hm34_asym = 2*np.abs(higgs_m[:,1]-higgs_m[:,0])/(higgs_m[:,1]+higgs_m[:,0])
    return dict(
        hm12_asym=hm12_asym,hm13_asym=hm13_asym,hm14_asym=hm14_asym,
        hm23_asym=hm23_asym,hm24_asym=hm24_asym,hm34_asym=hm34_asym,
        hm1234_asym=(hm12_asym+hm34_asym)/2, 
    )

def calc_2y_asym(y_m):
    y_m.sort(axis=-1)
    ym_asym = (y_m[:,1]-y_m[:,0])/(y_m[:,1]+y_m[:,0])
    return dict(ym_asym=ym_asym)


def calc_m_asym(tree):
    higgs_m = ak.concatenate([ array[:,None] for array in ak.unzip(tree[["H1Y1_m","H2Y1_m","H1Y2_m","H2Y2_m"]])],axis=-1).to_numpy()
    hm_asym = calc_4h_asym(higgs_m)

    y_m = ak.concatenate([ array[:,None] for array in ak.unzip(tree[["Y1_m","Y2_m"]])],axis=-1).to_numpy()
    ym_asym = calc_2y_asym(y_m)

    tree.extend(**hm_asym, **ym_asym)

def standardize_variable(array):
    mu, sigma = np.mean(array, axis=-1)[:,None], np.std(array, axis=-1)[:,None]
    return (array - mu)/sigma

def standardize_resonances(tree):
    higgs_m = ak.concatenate([ array[:,None] for array in ak.unzip(tree[["H1Y1_m","H2Y1_m","H1Y2_m","H2Y2_m"]])],axis=-1).to_numpy()
    higgs_std_m = standardize_variable(higgs_m)

    y_m = ak.concatenate([ array[:,None] for array in ak.unzip(tree[["Y1_m","Y2_m"]])],axis=-1).to_numpy()
    y_std_m = standardize_variable(y_m)

    tree.extend(higgs_std_m=higgs_std_m, y_std_m=y_std_m)
    
def set_asym(tree):
    tree.extend(
        asym1 = tree.hm13_asym,
        asym2 = tree.hm24_asym,
    )
    

ar = np.array([0.075,0.075])
sr = 0.1

cr = 2*sr
vr = ar + 1.2*(sr+cr) * ar/np.sqrt((ar**2).sum())

def hm_asym_diff(tree,ar=ar,vr=vr):
    arx,ary = ar
    vrx,vry = vr
    tree.extend(
        asym_diff= np.sqrt((tree.asym1 - arx)**2 + (tree.asym2 - ary)**2),
        asym_diff_vr= np.sqrt((tree.asym1 - vrx)**2 + (tree.asym2 - vry)**2)
    )
    
def n_selected_btag(tree, btagwp):
    jet_btag = ak.concatenate([ array[:,None] for array in ak.unzip(tree[[f"{bjet}_btag" for bjet in quarklist]])],axis=-1)
    return ak.sum(jet_btag > btagwp,axis=-1)

def selected_btagsum(tree):
    jet_btag = ak.concatenate([ array[:,None] for array in ak.unzip(tree[[f"{bjet}_btag" for bjet in quarklist]])],axis=-1)
    return ak.sum(jet_btag,axis=-1)

target_filter = EventFilter("medium_4btag",filter=lambda t : n_selected_btag(t,jet_btagWP[2]) > 3)
estimation_filter = EventFilter("medium_inv_3btag",filter=lambda t : n_selected_btag(t,jet_btagWP[2]) == 3)

target_filter_v2 = EventFilter("medium_5btag",filter=lambda t : n_selected_btag(t,jet_btagWP[2]) > 4)
estimation_filter_v2 = EventFilter("medium_inv_5btag",filter=lambda t : n_selected_btag(t,jet_btagWP[2]) < 5)

target_filter_v3 = EventFilter("high_btagsum",filter=lambda t : selected_btagsum(t) > 4)
estimation_filter_v3 = EventFilter("low_btagsum",filter=lambda t : selected_btagsum(t) <= 4)

asr_filter = EventFilter('asr',asym_diff_max=sr)
acr_filter = EventFilter('acr',asym_diff_min=sr,asym_diff_max=cr)

vsr_filter = EventFilter('vsr',asym_diff_vr_max=sr)
vcr_filter = EventFilter('vcr',asym_diff_vr_min=sr,asym_diff_vr_max=cr)

bdt_features = [
            'jet_ht','min_jet_deta','max_jet_deta','min_jet_dr','max_jet_dr'
        ] + [
            f'h{i+1}_{var}'
            for var in ('pt','jet_dr')
            for i in range(4)
        ] + [
            f'h{i+1}{j+1}_{var}'
            for var in ('dphi','deta')
            for i in range(4)
            for j in range(i+1, 4)
        ]