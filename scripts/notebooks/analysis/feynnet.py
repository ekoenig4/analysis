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
from utils.notebookUtils import required, dependency

def main():
    notebook = Notebook.from_parser()
    notebook.hello()
    notebook.run()

varinfo.feynnet_maxscore = dict(bins=(-0.05,1.05,30))
varinfo.feynnet_minscore = dict(bins=(-0.05,1.05,30))

org_features = [
    jet_ht, min_jet_deta, max_jet_deta, min_jet_dr, max_jet_dr, 
    'h_pt', 'h_j_dr', h_deta, h_dphi
]

new_features = [
    'X_m','Y1_m','Y2_m','H1Y1_m','H2Y1_m','H1Y2_m','H2Y2_m','feynnet_minscore','feynnet_maxscore'
]

@cache_variable(bins=(0,500,30))
def Y1_hm_chi(t):
    return np.sqrt( ak.sum( (t.h_m[:,:2]-125)**2, axis=1 ) )

@cache_variable(bins=(0,500,30))
def Y2_hm_chi(t):
    return np.sqrt( ak.sum( (t.h_m[:,2:]-125)**2, axis=1 ) )

bdtVersions = {
    'org': {
        'ar': ABCD(
            features=org_features,
            a=lambda t : (t.n_medium_btag >  4) & ( Y1_hm_chi(t) < 125/2 ) & ( Y2_hm_chi(t) < 125 ),
            b=lambda t : (t.n_medium_btag == 4) & ( Y1_hm_chi(t) < 125/2 ) & ( Y2_hm_chi(t) < 125 ),
            c=lambda t : (t.n_medium_btag >  4) & ( Y1_hm_chi(t) > 125/2 ) & ( Y2_hm_chi(t) < 125 ),
            d=lambda t : (t.n_medium_btag == 4) & ( Y1_hm_chi(t) > 125/2 ) & ( Y2_hm_chi(t) < 125 ),
        ),
        'vr': ABCD(
            features=org_features,
            a=lambda t : (t.n_medium_btag >  4) & ( Y1_hm_chi(t) < 125/2 ) & ( Y2_hm_chi(t) > 125 ),
            b=lambda t : (t.n_medium_btag == 4) & ( Y1_hm_chi(t) < 125/2 ) & ( Y2_hm_chi(t) > 125 ),
            c=lambda t : (t.n_medium_btag >  4) & ( Y1_hm_chi(t) > 125/2 ) & ( Y2_hm_chi(t) > 125 ),
            d=lambda t : (t.n_medium_btag == 4) & ( Y1_hm_chi(t) > 125/2 ) & ( Y2_hm_chi(t) > 125 ),
        )
    },
    "new":{
        'ar': ABCD(
            features=new_features,
            a=lambda t : (t.n_medium_btag > 4) & ( hm_chi(t) < 50 ),
            b=lambda t : ((t.n_medium_btag == 4) | (t.n_medium_btag == 3)) & ( hm_chi(t) < 50 ),
            c=lambda t : (t.n_medium_btag > 4) & ( hm_chi(t) > 50 ),
            d=lambda t : ((t.n_medium_btag == 4) | (t.n_medium_btag == 3)) & ( hm_chi(t) > 50 ),
        ),
        'vr': ABCD(
            features=new_features,
            a=lambda t : (t.n_medium_btag > 4) & ( hm_chi(t) > 50 ) & ( hm_chi(t) < 75 ),
            b=lambda t : ((t.n_medium_btag == 4) | (t.n_medium_btag == 3)) & ( hm_chi(t) > 50 ) & ( hm_chi(t) < 75 ),
            c=lambda t : (t.n_medium_btag > 4) & ( hm_chi(t) > 75 ),
            d=lambda t : ((t.n_medium_btag == 4) | (t.n_medium_btag == 3)) & ( hm_chi(t) > 75 ),
        )

    }
}

class Notebook(RunAnalysis):
    @staticmethod
    def add_parser(parser):
        parser.add_argument("--model", default='feynnet_trgkin_mx_my_reweight', help="specify the feynnet model to use for analysis")
        parser.add_argument("--bdt", default='new', help="specify the bdt version to use for analysis", choices=bdtVersions.keys()) 
        parser.add_argument("--serial", action='store_true', help="run in serial mode")
        parser.set_defaults(
            module='fc.eightb.preselection.t8btag_minmass',
            use_signal='feynnet_signal_list',
        )

    @required
    def init(self, signal):
        
        self.use_signal = [ i for i, mass in enumerate(signal.apply(lambda t : t.mass)) if mass in ( '(800, 350)', '(1200, 450)', '(1200, 250)' ) ]
        self.dout = f'feynnet/{self.model}/{self.bdt}'
        self.model = getattr(eightb.models, self.model)
        self.load_feynnet_assignment = eightb.f_load_feynnet_assignment(self.model.analysis)

        ar_bdt = bdtVersions[self.bdt]['ar']
        vr_bdt = bdtVersions[self.bdt]['vr']

        self.bdt = ar_bdt
        self.vr_bdt = vr_bdt

    @required
    def load_feynnet(self, signal, bkg, data):

        if self.serial:
            (signal+bkg+data).apply( self.load_feynnet_assignment, report=True )
        else:
            # import multiprocessing as mp
            # with mp.Pool(5) as pool:
            import concurrent.futures as cf
            with cf.ProcessPoolExecutor(5) as pool:
                (signal+bkg+data).parallel_apply( self.load_feynnet_assignment, report=True, pool=pool )

        (signal+bkg+data).apply( add_h_j_dr, report=True )
        (signal+bkg+data).apply( eightb.assign, report=True )

    def plot_signal_mcbkg(self, signal, bkg):
        study.quick(
            signal[self.use_signal]+bkg,
            legend=True,
            varlist=['H1Y1_m', 'Y1_m'],
            efficiency=True,
            saveas=f'{self.dout}/signal_mcbkg',
        )

        study.quick(
            signal[self.use_signal]+bkg,
            varlist=new_features+org_features,
            h_rebin=15,
            legend=True,
            efficiency=True,
            saveas=f'{self.dout}/signal_mcbkg_bdt_features',
        )


    @required
    def blind_data(self, signal, bkg, data):
        # ( signal+bkg+data ).apply(hm_chi, report=True)
        blinded = EventFilter('blinded', filter=lambda t :  ~self.bdt.a(t) )
        self.blinded_data = data.apply(blinded)
        self.bkg_model = self.blinded_data.asmodel()

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

    def plot_abcd_region(self, blinded_data, bkg):
        study.quick(
            blinded_data+bkg,
            varlist=[hm_chi],
            binlist=[(0,300,30)],
            h_rebin=15,
            legend=True,
            ratio=True,
        )

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

    def build_ar_bdt(self, bkg_model):
        self.bdt.print_yields(bkg_model)

    @dependency(build_ar_bdt)
    def train_ar_bdt(self, bkg_model):
        self.bdt.train(bkg_model)
        self.bdt.print_results(bkg_model)
    
    @dependency(train_ar_bdt)
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

    @dependency(train_ar_bdt)
    def plot_extraction_variables(self, signal, bkg_model):
        study.quick(
            signal[self.use_signal]+bkg_model,
            masks=[self.bdt.a]*len(self.use_signal) + [self.bdt.b]*len(bkg_model),
            scale=[1]*len(self.use_signal) + [self.bdt.reweight_tree]*len(bkg_model),
            varlist=['X_m',flatten_mxmy, hilbert_mxmy],
            h_rebin=50,
            legend=True,
            dim=-1,
            saveas=f'{self.dout}/extraction_variables',
        )   

    def _plot_limits(self, sig_models, tag):
        study.brazil_limits(
            sig_models,
            xlabel='my',
            saveas=f'{self.dout}/{tag}_limits_my',
        )

        study.brazil_limits(
            sig_models,
            xlabel='mx',
            saveas=f'{self.dout}/{tag}_limits_mx',
        )

        study.brazil2d_limits(
            sig_models,
            zlim=np.linspace(0,100,11),
            g_cmap='jet',
            saveas=f'{self.dout}/{tag}_2d_limits',
        )


    @dependency(train_ar_bdt)
    def get_mx_limits(self, signal, bkg_model):
        sig_mx_models = study.limits(
            signal+bkg_model,
            masks=[self.bdt.a]*len(signal) + [self.bdt.b]*len(bkg_model),
            scale=[1]*len(signal) + [self.bdt.reweight_tree]*len(bkg_model),
            varlist=['X_m'],
            h_rebin=50,
            poi=np.linspace(0,10,31),
        )   

        self._plot_limits(sig_mx_models, 'mx')

    @dependency(train_ar_bdt)
    def get_flatten_mxmy_limits(self, signal, bkg_model):
        sig_flatten_mxmy_models = study.limits(
            signal+bkg_model,
            masks=[self.bdt.a]*len(signal) + [self.bdt.b]*len(bkg_model),
            scale=[1]*len(signal) + [self.bdt.reweight_tree]*len(bkg_model),
            varlist=[flatten_mxmy],
            binlist=[(0,1,50)],
            poi=np.linspace(0,10,31),
        )   

        self._plot_limits(sig_flatten_mxmy_models, 'flatten_mxmy')

    @dependency(train_ar_bdt)
    def get_hilbert_mxmy_limits(self, signal, bkg_model):
        sig_hilbert_mxmy_models = study.limits(
            signal+bkg_model,
            masks=[self.bdt.a]*len(signal) + [self.bdt.b]*len(bkg_model),
            scale=[1]*len(signal) + [self.bdt.reweight_tree]*len(bkg_model),
            varlist=[hilbert_mxmy],
            binlist=[(0,1,50)],
            poi=np.linspace(0,10,31),
        )   

        self._plot_limits(sig_hilbert_mxmy_models, 'hilbert_mxmy')

    def build_vr_bdt(self, bkg_model):
        self.vr_bdt.print_yields(bkg_model)

    @dependency(build_vr_bdt)
    def train_vr_bdt(self, bkg_model):
        self.vr_bdt.train(bkg_model)
        self.vr_bdt.print_results(bkg_model)

    @dependency(train_vr_bdt)
    def plot_vr_features(self, blinded_data, bkg_model):
        study.quick(
            blinded_data+bkg_model,
            masks=[self.vr_bdt.a]*len(blinded_data) + [self.vr_bdt.b]*len(bkg_model),
            scale=[1]*len(blinded_data) + [self.vr_bdt.scale_tree]*len(bkg_model),
            varlist=new_features+org_features,
            h_rebin=15,
            suptitle='VR BDT Pre-Fit',
            legend=True,

            ratio=True,
            **study.kstest,
            saveas=f'{self.dout}/vr_features_prefit',
        )

        study.quick(
            blinded_data+bkg_model,
            masks=[self.vr_bdt.a]*len(blinded_data) + [self.vr_bdt.b]*len(bkg_model),
            scale=[1]*len(blinded_data) + [self.vr_bdt.reweight_tree]*len(bkg_model),
            varlist=new_features+org_features,
            h_rebin=15,
            suptitle='VR BDT Post-Fit',
            legend=True,

            ratio=True,
            **study.kstest,
            saveas=f'{self.dout}/vr_features_postfit',
        )

    @dependency(train_vr_bdt)
    def plot_vr_bdt(self, blinded_data, bkg_model):
        study.quick(
            blinded_data+bkg_model,
            masks=[self.vr_bdt.a]*len(blinded_data) + [self.vr_bdt.b]*len(bkg_model),
            scale=[1]*len(blinded_data) + [self.vr_bdt.scale_tree]*len(bkg_model),
            varlist=['X_m', 'Y1_m', 'Y2_m', None]+[f'{H}_m' for H in eightb.higgslist],
            h_rebin=15,
            suptitle='VR BDT Pre-Fit',
            legend=True,

            ratio=True,
            **study.kstest,
            saveas=f'{self.dout}/vr_bdt_prefit',
        )

        study.quick(
            blinded_data+bkg_model,
            masks=[self.vr_bdt.a]*len(blinded_data) + [self.vr_bdt.b]*len(bkg_model),
            scale=[1]*len(blinded_data) + [self.vr_bdt.reweight_tree]*len(bkg_model),
            varlist=['X_m', 'Y1_m', 'Y2_m', None]+[f'{H}_m' for H in eightb.higgslist],
            h_rebin=15,
            suptitle='VR BDT Post-Fit',
            legend=True,

            ratio=True,
            **study.kstest,
            saveas=f'{self.dout}/vr_bdt_postfit',
        )



if __name__ == '__main__': main()