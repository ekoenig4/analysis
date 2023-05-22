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

from utils.notebookUtils.driver.run_analysis import RunAnalysis

def main():
    notebook = Notebook.from_parser()
    notebook.hello()
    notebook.run()

varinfo.feynnet_maxscore = dict(bins=(-0.05,1.05,30))
varinfo.feynnet_minscore = dict(bins=(-0.05,1.05,30))
    
@cache_variable(bins=(0,300,30))
def hm_chi(t):
    return np.sqrt( ak.sum((t.h_m-125)**2, axis=1) )

@cache_variable(bins=(0,1,30))
def flatten_mxmy(tree):
    mx_bins = np.linspace(0,2000,30)
    my_bins = np.linspace(0,1000,30)

    mx = np.digitize(tree.X_m, mx_bins)
    my1 = np.digitize(tree.Y1_m, my_bins)

    mxmy = len(my_bins)*mx + my1
    mxmy = mxmy/np.max(mxmy)
    return mxmy

# %%
import hilbert 
@cache_variable(bins=(0,1,30))
def hilbert_mxmy(tree):
    mx_bins = np.linspace(0,2000,30)
    my_bins = np.linspace(0,1000,30)

    mx = np.digitize(tree.X_m, mx_bins)
    my1 = np.digitize(tree.Y1_m, my_bins)

    mxmy = hilbert.encode(np.stack([mx, my1], axis=1).to_numpy(), 2, 32)
    mxmy = mxmy/(np.max(mxmy)+1)
    return mxmy

@cache_variable(bins=(0,1,30))
def hilbert_mxmy2(tree):
    mx_bins = np.linspace(0,2000,30)
    my_bins = np.linspace(0,1000,30)

    mx = np.digitize(tree.X_m, mx_bins)
    my1 = np.digitize(tree.Y1_m, my_bins)
    my2 = np.digitize(tree.Y2_m, my_bins)

    mxmy2 = np.stack([mx,my1, my2], axis=1).to_numpy()
    mxmy2 = hilbert.encode(mxmy2, 3, 20)
    mxmy2 = mxmy2/(np.max(mxmy2)+1)
    return mxmy2

class Notebook(RunAnalysis):
    @staticmethod
    def add_parser(parser):
        parser.add_argument("--model", default='feynnet_trgkin_mx_my_reweight', help="specify the feynnet model to use for analysis")
        parser.set_defaults(
            module='fc.eightb.preselection.t8btag_minmass',
            use_signal='feynnet_signal_list',
        )

    @required
    def init(self, signal):

        self.use_signal = [ i for i, mass in enumerate(signal.mass) if mass in ( '(800, 350)', '(1200, 450)', '(1200, 250)' ) ]
        self.dout = f'feynnet/{self.model}'
        self.model = getattr(eightb.models, self.model)
        self.load_feynnet_assignment = eightb.f_load_feynnet_assignment(self.model.analysis)

    @required
    def load_feynnet(self, signal, bkg, data):
        import multiprocess as mp

        # with mp.Pool(5) as pool:
        #     (signal+bkg+data).parallel_apply( self.load_feynnet_assignment, report=True, pool=pool )

        (signal+bkg+data).apply( self.load_feynnet_assignment, report=True )
        (signal+bkg+data).apply(eightb.assign, report=True)

    def plot_signal_mcbkg(self, signal, bkg):
        study.quick(
            signal[self.use_signal]+bkg,
            legend=True,
            varlist=['H1Y1_m', 'Y1_m'],
            efficiency=True,
            saveas=f'{self.dout}/signal_mcbkg',
        )

    @required
    def blind_data(self, signal, bkg, data):
        ( signal+bkg+data ).apply(hm_chi, report=True)
        blinded = EventFilter('blinded', filter=lambda t :  ~ ((t.n_medium_btag > 4) & ( hm_chi(t) < 50 )) )
        self.blinded_data = data.apply(blinded)

    def plot_blinded_data(self, bkg, blinded_data):
        study.quick(
            blinded_data+bkg,
            masks=lambda t: t.n_medium_btag == 3,
            varlist=['X_m', 'Y1_m', 'Y2_m', None]+[f'{H}_m' for H in eightb.higgslist],
            efficiency=True,
            legend=True,
            h_rebin=15,
            ratio=True,
            saveas=f'{self.dout}/blinded_data_3btag',
            # log=True,
        )

    def plot_abcd_region(self, blinded_data):
        study.quick2d(
            blinded_data,
            varlist=['H1Y1_m','H2Y1_m'],
            binlist=[(0,500,30),(0,500,30)],
            interp=True,
            h_cmap='jet',
            # size=(5,10),
            colorbar=True,

            exe=[
                draw_circle(x=125,y=125,r=50, text=None, linewidth=2),
                lambda ax, **kwargs : ax.text(125, 125, 'AR', horizontalalignment='center', verticalalignment='center', fontsize=20),

                draw_circle(x=125,y=125,r=75, text=None, linewidth=2, color='red'),
                lambda ax, **kwargs : ax.text(200, 200, 'VR', horizontalalignment='center', verticalalignment='center', fontsize=20, color='red'),
            ],
            saveas=f'{self.dout}/abcd_region',
        )

    @required
    def build_ar_bdt(self, blinded_data):
        self.bkg_model = blinded_data.asmodel()

        self.bdt = ABCD(
            features=['feynnet_minscore']+[f'{res}_m' for res in eightb.scalarlist],
            a=lambda t : (t.n_medium_btag > 4) & ( hm_chi(t) < 50 ),
            b=lambda t : ((t.n_medium_btag == 4) | (t.n_medium_btag == 3)) & ( hm_chi(t) < 50 ),
            c=lambda t : (t.n_medium_btag > 4) & ( hm_chi(t) > 50 ),
            d=lambda t : ((t.n_medium_btag == 4) | (t.n_medium_btag == 3)) & ( hm_chi(t) > 50 ),
        )

    def print_ar_bdt(self, bkg_model):
        self.bdt.print_yields(bkg_model)

    @required
    def train_ar_bdt(self, bkg_model):
        self.bdt.train(bkg_model)
        self.bdt.print_results(bkg_model)
    
    def plot_ar_bdt(self, signal, bkg_model):
        study.quick(
            signal[self.use_signal]+bkg_model,
            masks=[self.bdt.a]*len(self.use_signal) + [self.bdt.b]*len(bkg_model),
            scale=[1]*len(self.use_signal) + [self.bdt.reweight_tree]*len(bkg_model),
            varlist=['X_m', 'Y1_m', 'Y2_m', None]+[f'{H}_m' for H in eightb.higgslist],
            h_rebin=50,
            legend=True,
            saveas=f'{self.dout}/ar_bdt',
        )

    def plot_extraction_variables(self, signal, bkg_model):
        study.quick(
            signal[self.use_signal]+bkg_model,
            masks=[self.bdt.a]*len(self.use_signal) + [self.bdt.b]*len(bkg_model),
            scale=[1]*len(self.use_signal) + [self.bdt.reweight_tree]*len(bkg_model),
            varlist=['X_m',flatten_mxmy, hilbert_mxmy, hilbert_mxmy2],
            h_rebin=50,
            legend=True,
            dim=-1,
            saveas=f'{self.dout}/extraction_variables',
        )   
    
    @required
    def get_limits(self, signal, bkg_model):
        self.sig_mx_models = study.limits(
            signal+bkg_model,
            masks=[self.bdt.a]*len(signal) + [self.bdt.b]*len(bkg_model),
            scale=[1]*len(signal) + [self.bdt.reweight_tree]*len(bkg_model),
            varlist=['X_m'],
            h_rebin=50,
            poi=np.linspace(0,10,31),
        )   

        self.sig_hilbert_mxmy_models = study.limits(
            signal+bkg_model,
            masks=[self.bdt.a]*len(signal) + [self.bdt.b]*len(bkg_model),
            scale=[1]*len(signal) + [self.bdt.reweight_tree]*len(bkg_model),
            varlist=[hilbert_mxmy],
            binlist=[(0,1,50)],
            poi=np.linspace(0,10,31),
        )   

        self.sig_flatten_mxmy_models = study.limits(
            signal+bkg_model,
            masks=[self.bdt.a]*len(signal) + [self.bdt.b]*len(bkg_model),
            scale=[1]*len(signal) + [self.bdt.reweight_tree]*len(bkg_model),
            varlist=[flatten_mxmy],
            binlist=[(0,1,50)],
            poi=np.linspace(0,10,31),
        )   

    def plot_limits(self, sig_mx_models, sig_flatten_mxmy_models, sig_hilbert_mxmy_models):
        study.brazil_limits(
            sig_mx_models,
            xlabel='my',
            saveas=f'{self.dout}/mx_limits',
        )

        study.brazil_limits(
            sig_flatten_mxmy_models,
            xlabel='my',
            saveas=f'{self.dout}/flatten_mxmy_limits',
        )

        study.brazil_limits(
            sig_hilbert_mxmy_models,
            xlabel='my',
            saveas=f'{self.dout}/hilbert_mxmy_limits',
        )

    def plot_2d_limits(self, sig_mx_models, sig_flatten_mxmy_models, sig_hilbert_mxmy_models):
        study.brazil2d_limits(
            sig_mx_models,
            zlim=np.linspace(0,100,11),
            g_cmap='jet',
            saveas=f'{self.dout}/mx_2d_limits',
        )

        study.brazil2d_limits(
            sig_flatten_mxmy_models,
            zlim=np.linspace(0,100,11),
            g_cmap='jet',
            saveas=f'{self.dout}/flatten_mxmy_2d_limits',
        )

        study.brazil2d_limits(
            sig_hilbert_mxmy_models,
            zlim=np.linspace(0,100,11),
            g_cmap='jet',
            saveas=f'{self.dout}/hilbert_mxmy_2d_limits',
        )

    def build_vr_bdt(self, bkg_model):
        self.vr_bdt = ABCD(
            features=['feynnet_minscore']+[f'{res}_m' for res in eightb.scalarlist],
            a=lambda t : (t.n_medium_btag > 4) & ( hm_chi(t) > 50 ) & ( hm_chi(t) < 75 ),
            b=lambda t : ((t.n_medium_btag == 4) | (t.n_medium_btag == 3)) & ( hm_chi(t) > 50 ) & ( hm_chi(t) < 75 ),
            c=lambda t : (t.n_medium_btag > 4) & ( hm_chi(t) > 75 ),
            d=lambda t : ((t.n_medium_btag == 4) | (t.n_medium_btag == 3)) & ( hm_chi(t) > 75 ),
        )

        self.vr_bdt.print_yields(bkg_model)

    def train_vr_bdt(self, bkg_model):
        self.vr_bdt.train(bkg_model)
        self.vr_bdt.print_results(bkg_model)

    def plot_vr_bdt(self, blinded_data, bkg_model):
        study.quick(
            blinded_data+bkg_model,
            masks=[self.vr_bdt.a]*len(blinded_data) + [self.vr_bdt.b]*len(bkg_model),
            scale=[1]*len(blinded_data) + [self.vr_bdt.reweight_tree]*len(bkg_model),
            varlist=['X_m', 'Y1_m', 'Y2_m', None]+[f'{H}_m' for H in eightb.higgslist],
            h_rebin=15,
            legend=True,

            ratio=True,
            **study.kstest,

            saveas=f'{self.dout}/vr_bdt',
        )



if __name__ == '__main__': main()