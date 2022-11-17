import numpy as np
import awkward as ak

from ..utils import *
from ..selectUtils import combinations, to_pair_combinations, calc_dphi, calc_dr_p4

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
def pair_y_from_higgs(t, higgs='higgs', operator=y_min_mass_asym):
    higgs = get_collection(t, higgs, False)
    higgs_p4 = build_p4(higgs)
    h1, h2 = ak.unzip(ak.combinations(higgs_p4, n=2, axis=1))

    ys = h1 + h2
    ys_dr = calc_dr_p4(h1, h2)
    ys = ak.zip(dict(
        dr = ys_dr,
        **{
            var:getattr(ys,var)
            for var in ('pt','m','eta','phi')
        }
    ))

    y1 = ys[:,y_higgs_pair_index[:,0]]
    y2 = ys[:,y_higgs_pair_index[:,1]]
    
    ys = ak_stack((y1, y2), axis=2)
    ys = ys[ak.argsort(ys.pt,axis=-1, ascending=False)]

    order = operator(ys)
    ys = ys[order]

    t.extend(
        **{
            f'Y{i+1}_{var}':ys[var][:,i]
            for i in range(2)
            for var in ('pt','m','eta','phi','dr')
        },
    )