import numpy as np
import awkward as ak

from ..utils import *
from ..selectUtils import combinations, to_pair_combinations, calc_dphi, calc_dr_p4, build_all_dijets
from .pairing import load_weaver_output

def y_min_mass_asym(ys):
    return ak.argsort((ys.m[:,:,0] - ys.m[:,:,1])**2/(ys.m[:,:,0] + ys.m[:,:,1]),axis=-1)
def y_max_ht(ys):
    return ak.argsort(ys.pt[:,:,0] + ys.pt[:,:,1], axis=-1, ascending=False)
def y_max_dr(ys):
    dr = calc_dr_p4(ys)
    return ak.argsort(dr, axis=-1, ascending=False)
def y_max_dphi(ys):
    dphi = np.abs(calc_dphi(ys.phi[:,:,0], ys.phi[:,:,1]))
    return ak.argsort(dphi, axis=-1, ascending=False)
def y_min_higgs_dr(ys):
    dr2 = ys.dr[:,:,0]**2 + ys.dr[:,:,1]**2
    return ak.argsort(dr2, axis=-1)
def y1_min_higgs_dr(ys):
    return ak.argsort(ys.dr[:,:,0], axis=-1)

y_higgs_index = combinations(4, [2,2])
y_higgs_pair_index = to_pair_combinations(y_higgs_index[:,:,0], y_higgs_index[:,:,1])
def pair_y_from_higgs(t, higgs='higgs', operator=None, pairs=None):
    higgs = get_collection(t, higgs, False)
    higgs_p4 = build_p4(higgs)

    if operator is not None:
        h1, h2 = ak.unzip(ak.combinations(higgs_p4, n=2, axis=1))
        ys = h1 + h2
        ys_dr = calc_dr_p4(h1, h2)
        ys = ak.zip(dict(
            higgs_dr = ys_dr,
            **{
                var:getattr(ys,var)
                for var in ('pt','m','eta','phi')
            }
        ))
        y1 = ys[:,y_higgs_pair_index[:,0]]
        y2 = ys[:,y_higgs_pair_index[:,1]]
    elif pairs is not None:
        ...
    else:
        raise ValueError('need to provide ranking operator or pair index')
    
    ys = ak_stack((y1, y2), axis=2)
    y_pt_order = ak.argsort(ys.pt,axis=-1, ascending=False)
    ys = ys[y_pt_order]

    order = operator(ys)
    ys = ys[order][:,0]
    y_pt_order = y_pt_order[order][:,0]
    h1_idx = ak.from_regular(y_higgs_index[order[:,0]][:,:,0])[y_pt_order]
    h2_idx = ak.from_regular(y_higgs_index[order[:,0]][:,:,1])[y_pt_order]
    ys = join_fields(ys, h1Idx=h1_idx, h2Idx=h2_idx)
    h1 = higgs[ys.h1Idx]
    h2 = higgs[ys.h2Idx]
    h1y1 = h1[:,0]
    h2y1 = h2[:,0]
    h1y2 = h1[:,1]
    h2y2 = h2[:,1]

    t.extend(
        **{
            f'Y{i+1}_{var}':ys[var][:,i]
            for i in range(2)
            for var in ys.fields
        },
        **{
            f'{key}_{field}':h[field]
            for key, h in zip(['H1Y1','H2Y1','H1Y2','H2Y2'],[h1y1,h2y1,h1y2,h2y2])
            for field in h.fields
        },
    )

def load_yy_quadh_ranker(tree, model):
    ranker = load_weaver_output(tree, model, fields=['maxcomb','maxscore'])

    score, index = ranker['maxscore'], ranker['maxcomb']
    index = np.stack([index[:,::2],index[:,1::2]], axis=2)
    jet_index = ak.from_regular( index.astype(int) )

    build_all_dijets(tree, pairs=jet_index, name='higgs', ordered='pt')
    tree.extend(yy_quadh_score=score)

    higgs = get_collection(tree, 'higgs', named=False)
    hp4 = build_p4(higgs)
    h_idx = ak.argsort(higgs.localId//2,axis=-1)
    h1_idx = h_idx[:,::2]
    h2_idx = h_idx[:,1::2]
    h1, h2 = hp4[h1_idx], hp4[h2_idx]
    ys = h1 + h2
    higgs_dr = calc_dr_p4(h1, h2)
    ys = ak.zip(
        dict(
            higgs_dr = higgs_dr,
            **{
                var:getattr(ys,var)
                for var in ('pt','m','eta','phi')
            }
        )
    )
    ys = join_fields(ys, h1Idx=h1_idx, h2Idx=h2_idx)
    ys = ys[ak.argsort(-ys.pt,axis=-1)]
    h1 = higgs[ys.h1Idx]
    h2 = higgs[ys.h2Idx]
    h1y1 = h1[:,0]
    h2y1 = h2[:,0]
    h1y2 = h1[:,1]
    h2y2 = h2[:,1]


    tree.extend(
        **{
            f'Y{i+1}_{var}':ys[var][:,i]
            for i in range(2)
            for var in ys.fields
        },
        **{
            f'{key}_{field}':h[field]
            for key, h in zip(['H1Y1','H2Y1','H1Y2','H2Y2'],[h1y1,h2y1,h1y2,h2y2])
            for field in h.fields
        },
    )