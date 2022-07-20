# %%
import os
os.environ['KMP_WARNINGS'] = 'off'
import sys
import git

import uproot as ut
import awkward as ak
import numpy as np
import math
import vector
import sympy as sp

import re
from tqdm import tqdm
import timeit
import re


sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils import *

# %%
varinfo.X_m = dict(bins=np.linspace(500,2000,30))
varinfo.Y1_m = dict(bins=np.linspace(0,1000,30))
varinfo.Y2_m = dict(bins=np.linspace(0,1000,30))
varinfo.H1Y1_m = dict(bins=np.linspace(0,250,30))
varinfo.H2Y1_m = dict(bins=np.linspace(0,250,30))
varinfo.H1Y2_m = dict(bins=np.linspace(0,250,30))
varinfo.H2Y2_m = dict(bins=np.linspace(0,250,30))

# %%

print("Getting Trees")

signal = ObjIter([Tree(fc.eightb.preselection_ranked_quadh.NMSSM_XYY_YToHH_8b_MX_1000_MY_450)])
# signal_asym = ObjIter([Tree(fc.eightb.preselection.NMSSM_XYY_YToHH_8b_MX_1000_MY_450)])
qcd = ObjIter([Tree(fc.eightb.preselection_ranked_quadh.QCD_B_List)])
# qcd_asym = ObjIter([Tree(fc.eightb.preselection.QCD_B_List)])
ttbar = ObjIter([Tree(fc.eightb.preselection_ranked_quadh.TTJets)])
# ttbar = ObjIter([])
# qcd = ObjIter([])
bkg = qcd + ttbar

# %%
class random_sample(ObjTransform):
  def __call__(self, tree):
    n = int(tree.raw_events*self.n)
    mask = np.zeros(tree.raw_events)
    mask[:n] = 1
    return np.random.permutation(mask) == 1

# %%
def get_abcd_masks(v1_r, v2_r):
  v1_sr = lambda t : (t.n_medium_btag >= v1_r[1]) & (t.n_medium_btag < v1_r[2])
  v1_cr = lambda t : (t.n_medium_btag >= v1_r[0]) & (t.n_medium_btag < v1_r[1])

  v2_sr = lambda t : (t.quadh_score >= v2_r[1]) & (t.quadh_score < v2_r[2])
  v2_cr = lambda t : (t.quadh_score >= v2_r[0]) & (t.quadh_score < v2_r[1])

  r_a = lambda t : v1_sr(t) & v2_sr(t)
  r_b = lambda t : v1_cr(t) & v2_sr(t)

  r_c = lambda t : v1_sr(t) & v2_cr(t)
  r_d = lambda t : v1_cr(t) & v2_cr(t)
  return r_a, r_b, r_c, r_d

def get_abcd_scale(r_a, r_b, r_c, r_d):
  n_d = bkg.apply(lambda t:t.scale[r_d(t)]).apply(np.sum).npy.sum()
  n_c = bkg.apply(lambda t:t.scale[r_c(t)]).apply(np.sum).npy.sum()
  n_b = bkg.apply(lambda t:t.scale[r_b(t)]).apply(np.sum).npy.sum()
  n_a = bkg.apply(lambda t:t.scale[r_a(t)]).apply(np.sum).npy.sum()
  k_factor = n_c/n_d
  k_target = n_a/n_b
  return k_target, k_factor

def apply_abcd(v1_r, v2_r, tag=""):
  print(f'Processing ABCD region - {tag}')

  r_a, r_b, r_c, r_d = get_abcd_masks(v1_r, v2_r)
  k_target, k_factor = get_abcd_scale(r_a, r_b, r_c, r_d)

  print(f'K Factor: {k_factor:0.3}')
  print(f'K Target: {k_target:0.3}')
  print(f'K Ratio:  {k_target/k_factor:0.3}')

  fig, axs = study.get_figax(2)
  study.quick2d_region(
    bkg, label=['MC-Bkg'],
    varlist=['n_medium_btag','quadh_score'],
    binlist=[np.array(v1_r), np.array(v2_r)],
    efficiency=True,
    show_counts=True,
    figax=(fig,axs[0])
  )

  study.quick2d(
    signal,
    varlist=['n_medium_btag','quadh_score'],
    binlist=[np.array(v1_r), np.array(v2_r)],
    efficiency=True,
    show_counts=True,
    title=tag,
    figax=(fig,axs[1])
  )
  fig.tight_layout()
  study.save_fig(fig, '', f'{tag}/abcd_regions')


  varlist = [f'{obj}_{var}' for obj in ['X']+eightb.ylist+eightb.higgslist for var in ('m','pt','eta','phi')]
  fig, axs = study.quick_region(
    bkg, bkg, 
    varlist=varlist,
    h_color=None, label=['target','model'], legend=True,
    masks=[r_a]*len(bkg) + [r_b]*len(bkg),
    scale=[None]*len(bkg) + [k_factor]*len(bkg),
    h_label_stat=lambda h:f'{np.sum(h.weights):0.2e}',
    legend_loc='upper left',
    dim=(-1,4),

    ratio=True,
    r_size='50%',
    r_fill_error=True,
    r_ylabel=r'$\frac{target}{model}$',
    r_label_stat='y_mean_std',
    r_legend=True,
    r_legend_loc='upper left',

    empirical=True,
    # e_ylim=(-0.15,1.15),
    e_show=False,

    e_difference=True,
    e_d_size='75%',
    e_d_ylabel='$\Delta$ ECDF',
    e_d_legend_loc='upper left',
    saveas=f'{tag}/model_comparison',

    title=tag,
    return_figax=True,
  )


  ks = np.array([ax.store.empiricals.differences[0].stats.ks for ax in axs.flatten()])
  avgks = np.mean(ks)
  ypos = np.arange(ks.shape[0])

  fig,ax = plt.subplots(figsize=(8,10))
  ax.barh(ypos, ks, label=f'Avg KS: {avgks:0.2}')
  ax.set_yticks(ypos, labels=varlist)
  ax.invert_yaxis()  # labels read top-to-bottom
  ax.set(xlabel="KS Test Statstic", title=tag)
  ax.legend(loc='upper left')
  fig.tight_layout()

  study.save_fig(fig, '', f'{tag}/kstest')


# %%

abcd_regions = {
  'nominal':                    [(0,5,9),(0,0.2 ,1)],
  'validation/btag_hi':         [(3,5,9),(0,0.2 ,1)],
  'validation/btag_lo':         [(0,3,5),(0,0.2 ,1)],
  'validation/score_lo':        [(0,5,9),(0,0.15,0.2)],
  'validation/btag_hi_score_lo':[(3,5,9),(0,0.15,0.2)],
  'validation/btag_lo_score_lo':[(0,3,5),(0,0.15,0.2)]
}

for tag, (v1_r, v2_r) in abcd_regions.items():
  apply_abcd(v1_r, v2_r, tag=tag)
