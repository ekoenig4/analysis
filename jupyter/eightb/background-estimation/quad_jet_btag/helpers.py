from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from utils.plotUtils.extension import draw_abcd
from utils.plotUtils.histogram2d import Histo2D
from utils.bdtUtils import ABCD
import awkward as ak

from utils import study


def region_score(A, B, C, D):
    k_estm = (C/D)
    k_targ = (A/B)

    error = k_estm/k_targ - 1

    print(
        f"(C/D) K factor: {k_estm:0.3f}\n"
        f"(A/B) K target: {k_targ:0.3f}\n"
        f"ABCD  % error:  {error:0.2%}\n"
        )
        
def score_region(h, x1, x2, x3, y1, y2, y3):
    x, y, w = h.x_array, h.y_array, h.weights
    a = np.sum(w[(x >= x2) & (x < x3) & (y >= y2) & (y < y3)])
    b = np.sum(w[(x >= x1) & (x < x2) & (y >= y2) & (y < y3)])
    c = np.sum(w[(x >= x2) & (x < x3) & (y >= y1) & (y < y2)])
    d = np.sum(w[(x >= x1) & (x < x2) & (y >= y1) & (y < y2)])
    return np.abs( (c/d)*(b/a)-1 )


def sort_regions(RR, z_r, mask=None):
    if mask is not None:
        RR = RR[mask]
        z_r = z_r[mask]
    return RR[z_r.argsort()]

R = ak.combinations(np.arange(6), n=3, axis=0).tolist()
RR = np.array([ (x_r,y_r) for x_r in R for y_r in R])

def grid_region_search(bkg, xvar, yvar, threshold=0.1):
    h = Histo2D( getattr(bkg, xvar).cat, getattr(bkg, yvar).cat, weights=bkg.scale.cat )
    z_r = np.array([score_region(h, *x_r, *y_r) for x_r, y_r in tqdm(RR)])
    print(f'N Regions in acceptance ({threshold}):', (z_r < threshold).sum())
    return z_r

def discrete_abcd(x_v, y_v, x_r, y_r):
    x1, x2, x3 = x_r
    y1, y2, y3 = y_r
    return dict(
        a = lambda t : (t[x_v] >= x2) & (t[x_v] < x3) & (t[y_v] >= y2) & (t[y_v] < y3),
        b = lambda t : (t[x_v] >= x1) & (t[x_v] < x2) & (t[y_v] >= y2) & (t[y_v] < y3),
        c = lambda t : (t[x_v] >= x2) & (t[x_v] < x3) & (t[y_v] >= y1) & (t[y_v] < y2),
        d = lambda t : (t[x_v] >= x1) & (t[x_v] < x2) & (t[y_v] >= y1) & (t[y_v] < y2)
    )

def train_and_evaluate(signal, bkg, features, x_v, y_v, x_r, y_r):
    bdt = ABCD(
        features=features,
        **discrete_abcd(x_v, y_v, x_r, y_r)
    )
    bdt.train(bkg)
    bdt.print_results(bkg)

    study.quick2d(
        signal+bkg,
        xvarlist=[x_v],
        yvarlist=[y_v],
        binlist=[np.arange(6)]*6,
        exe=draw_abcd(x_r=x_r, y_r=y_r),
    )
    plt.show()

    study.quick_region(
        bkg, bkg, bkg, label=['MC-C Region','MC-D BDT Reweighted','MC-D Normalized',], legend=True,
        h_color=None,
        masks=[bdt.c]*len(bkg) + [bdt.d]*len(bkg) + [bdt.d]*len(bkg),
        scale=[1]*len(bkg) + [bdt.reweight_tree]*len(bkg) + [bdt.scale_tree]*len(bkg),
        varlist=['X_m','X_pt','X_eta','X_phi'],
        dim=(-1, 4),
        ratio=True,

        empirical=True, e_show=False,
        e_difference=True, e_d_legend=True, e_d_legend_frameon=True,
    )

    plt.show()
    
    study.quick_region(
        bkg, bkg, label=['MC-A Target','MC-B Model'], legend=True,
        h_color=None,
        masks=[bdt.a]*len(bkg) + [bdt.b]*len(bkg),
        scale=[1]*len(bkg) + [bdt.reweight_tree]*len(bkg),
        varlist=['X_m','X_pt','X_eta','X_phi'],
        dim=(-1, 4),
        suptitle='BDT Reweighted',
        ratio=True,

        empirical=True, e_show=False,
        e_difference=True, e_d_legend=True, e_d_legend_frameon=True,
    )

    plt.show()

    fig, axs = study.get_figax(2)
    study.quick(
        signal+bkg, legend=True,
        stack_fill=True,
        masks=[bdt.a]*len(signal)+[bdt.a]*len(bkg),
        varlist=['X_m'],
        limits=True,
        title='No Modeling',
        figax=(fig, axs[0])
    )

    study.quick(
        signal+bkg, legend=True,
        stack_fill=True,
        masks=[bdt.a]*len(signal)+[bdt.b]*len(bkg),
        scale=[1]*len(signal)+[bdt.reweight_tree]*len(bkg),
        varlist=['X_m'],
        limits=True,
        title='With BDT Model',
        figax=(fig, axs[1])
    )
    plt.show()

    return bdt
