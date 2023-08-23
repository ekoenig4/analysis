from functools import partial
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

from utils.notebookUtils import Notebook, required, dependency

def main():
    notebook = Analysis.from_parser()
    notebook.hello()
    notebook.run()


@cache_variable
def n_loose_btag(t):
    nL = t.ak4_h1b1_btag_L + t.ak4_h1b2_btag_L + t.ak4_h2b1_btag_L + t.ak4_h2b2_btag_L
    return ak.values_astype(nL, np.int32)

@cache_variable
def n_medium_btag(t):
    nM = t.ak4_h1b1_btag_M + t.ak4_h1b2_btag_M + t.ak4_h2b1_btag_M + t.ak4_h2b2_btag_M
    return ak.values_astype(nM, np.int32)

@cache_variable(bins=(0,100,30))
def h_dm(t):
    return np.sqrt( (t.dHH_H1_mass - 125)**2 + (t.dHH_H2_mass - 125)**2 )

@cache_variable(bins=(0,100,30))
def vr_h_dm(t):
    return np.sqrt( (t.dHH_H1_mass - 179)**2 + (t.dHH_H2_mass - 172)**2 )

bdt_features = [
    'ak4_h1b1_regpt', 'ak4_h1b2_regpt', 'ak4_h2b1_regpt', 'ak4_h2b2_regpt',
    'dHH_H1_mass', 'dHH_H2_mass', 'dHH_H1_pt', 'dHH_H2_pt', 
    'dHH_HH_mass', 'dHH_HH_pt','dHH_SumRegPtb', 'dHH_SumRegResb',
    'dHH_H1b1_H1b2_deltaR', 'dHH_H2b1_H2b2_deltaR', 'dHH_H1_H2_deltaEta','dHH_mindRbb', 
    'dHH_maxdEtabb','dHH_absCosTheta_H1_inHHcm', 'dHH_absCosTheta_H1b1_inH1cm', 'dHH_NbtagT',
]



varinfo.dHH_HH_mass = dict(bins=(200,1800,30))



def get_local_alt(f):
    to_local = lambda f : f.replace('/eos/user/e/ekoenig/','/store/user/ekoenig/')
    alt_pattern = to_local(f)

    alt_glob = fc.fs.eos.glob(alt_pattern)
    if any(alt_glob):
        return alt_glob
    
    remote_glob = fc.fs.cernbox.glob(f)
    if any(remote_glob):
        alt_glob = [ to_local(f) for f in remote_glob ]
        remote_glob = [ fc.fs.cernbox.fullpath(f) for f in remote_glob ]
        fc.fs.eos.batch_copy_to(remote_glob, alt_glob)

    alt_glob = fc.fs.eos.glob(alt_pattern)
    return alt_glob

class Analysis(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--dout', type=str, default='')
        parser.add_argument('--pairing', type=str, default='mindiag', )
        parser.add_argument('--use-old', action='store_true')
        parser.add_argument('--btagwp', type=str, default='medium')

    def _init_old(self):
        treekwargs = dict(  
            use_gen=False,
            treename='Events',
            normalization='Runs:genEventCount',
        )

        f_pattern = '/eos/user/e/ekoenig/4BAnalysis/CMSSW_12_5_0/src/PhysicsTools/NanoHH4b/run/jobs_sig_{pairing}_2018_0L/mc/GluGluToHHTo4B_node_cHHH1_TuneCP5_13TeV-powheg-pythia8_1_tree.root'
        f_sig = f_pattern.format(pairing=self.pairing)
        self.signal = ObjIter([Tree( get_local_alt(f_sig), **treekwargs)])

        # %%
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/bkg_{pairing}_2018_0L/mc/QCD*.root'
        f_bkg = f_pattern.format(pairing=self.pairing)
        self.bkg = ObjIter([Tree( get_local_alt(f_bkg), **treekwargs)])
        # self.bkg = ObjIter([])

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/data_{pairing}_2018_0L/data/JetHT*.root'
        f_data = f_pattern.format(pairing=self.pairing)
        self.data = ObjIter([Tree( get_local_alt(f_data), **dict(treekwargs, normalization=None, color='black'))])

    def _init_new(self):
        treekwargs = dict(
            use_gen=False,
            treename='Events',
            normalization=None,
        )
        
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_sig_2018_0L/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(pairing=self.pairing)

        self.signal = ObjIter([Tree( get_local_alt(f_sig), **treekwargs)])

        # %%
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_bkg_2018_0L/mc/qcd-mg_tree.root'
        f_bkg = f_pattern.format(pairing=self.pairing)
        self.bkg = ObjIter([Tree( get_local_alt(f_bkg), **treekwargs)])
        # self.bkg = ObjIter([])

        # signal xsec is set to 0.010517 pb -> 31.05 fb * (0.58)^2 
        (self.signal).apply(lambda t : t.reweight(t.genWeight * t.xsecWeight / 1000))
        (self.bkg).apply(lambda t : t.reweight(t.genWeight * t.xsecWeight / 1000))

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_data_2018_0L/data/jetht_tree.root'
        f_data = f_pattern.format(pairing=self.pairing)
        self.data = ObjIter([Tree( get_local_alt(f_data), **dict(treekwargs, normalization=None, color='black'))])

    @required
    def init(self):
        self.dout = os.path.join("nanoHH4b",self.dout,self.pairing,self.btagwp)

        if self.use_old:
            self._init_old()
        else:
            self._init_new()

        if self.btagwp == 'loose':
            self.n_btag = n_loose_btag
        else:
            self.n_btag = n_medium_btag
        
        self.bdt = ABCD(
            features=bdt_features,
            a=lambda t : (h_dm(t) <  25) & (self.n_btag(t) == 4),
            b=lambda t : (h_dm(t) <  25) & (self.n_btag(t) == 3),
            c=lambda t : (h_dm(t) >= 25) & (h_dm(t) < 50) & (self.n_btag(t) == 4),
            d=lambda t : (h_dm(t) >= 25) & (h_dm(t) < 50) & (self.n_btag(t) == 3),
        )

        self.vr_bdt = ABCD(
            features=bdt_features,
            a=lambda t : (vr_h_dm(t) <  25) & (self.n_btag(t) == 4),
            b=lambda t : (vr_h_dm(t) <  25) & (self.n_btag(t) == 3),
            c=lambda t : (vr_h_dm(t) >= 25) & (vr_h_dm(t) < 50) & (self.n_btag(t) == 4),
            d=lambda t : (vr_h_dm(t) >= 25) & (vr_h_dm(t) < 50) & (self.n_btag(t) == 3),
        )

    @required
    def apply_trigger(self):
        trigger = EventFilter('trigger', filter=lambda t : t.passTrig0L)
        self.signal = self.signal.apply(trigger)
        self.bkg = self.bkg.apply(trigger)
        self.data = self.data.apply(trigger)

    def plot_higgs(self, signal, bkg):
        study.quick(
            signal+bkg,
            varlist=['dHH_H1_mass','dHH_H2_mass'],
            binlist=[(0,300,30)]*2,
            efficiency=True,
            legend=True,
            saveas=f'{self.dout}/higgs_mass',
        )

        study.quick2d(
            signal+bkg,
            varlist=['dHH_H1_mass','dHH_H2_mass'],
            binlist=[(0,300,30)]*2,
            legend=True,
            saveas=f'{self.dout}/higgs_mass2d',
        )

    def plot_region_vars(self, signal, bkg):
        study.quick(
            signal+bkg,
            masks=self.bdt.mask,
            varlist=[h_dm, n_medium_btag],
            efficiency=True,
            legend=True,
            saveas=f'{self.dout}/bdt_region_vars',
        )

        study.quick2d(
            signal+bkg,
            varlist=[n_medium_btag, h_dm],
            binlist=[np.array((3,4,5)),None],
            efficiency=True,
            legend=True,
            exe=draw_abcd(
            x_r=(3,4,5), y_r=(0,25,50), regions=['C','D','A','B']
            ),
            saveas=f'{self.dout}/bdt_region_vars_2d',
        )

    def print_soverb(self, signal, bkg):
        def get_yields(t, f_mask):
            weights = t.scale
            mask = f_mask(t)
            return ak.sum(weights[mask])

        bkg_yields = bkg.apply(partial(get_yields, f_mask=self.bdt.a)).npy.sum()
        sig_yields = signal.apply(partial(get_yields, f_mask=self.bdt.a)).npy.sum()
        soverb = sig_yields / bkg_yields
        print(f'S/B = {soverb:.2f}')

    def data_ratio(self, data):

        n_3btag = data.apply(lambda t : np.sum( n_loose_btag(t) == 3 )).npy.sum()
        n_4btag = data.apply(lambda t : np.sum( n_loose_btag(t) == 4 )).npy.sum()

        print('Data 4b:    ', n_4btag)
        print('Data 3b:    ', n_3btag)
        print('Data 3b/4b: ', n_3btag/n_4btag)

    @required
    def blind_data(self, data):
        blind_data = EventFilter('blinded', filter=lambda t : ~self.bdt.a(t))
        self.data = data.apply(blind_data)

    def plot_3btag_datamc(self, data, bkg):
        study.quick(
            data+bkg,
            masks=lambda t : n_loose_btag(t) == 3,
            varlist=['dHH_H1_mass','dHH_H2_mass'],
            binlist=[(0,300,30)]*2,
            log=True,
            ratio=True, r_ylim=None,
            legend=True,

            saveas=f'{self.dout}/higgs_mass_3btag_datamc',
        )

        study.quick2d(
            data+bkg,
            masks=lambda t : n_loose_btag(t) == 3,
            varlist=['dHH_H1_mass','dHH_H2_mass'],
            binlist=[(0,300,30)]*2,
            legend=True,

            saveas=f'{self.dout}/higgs_mass2d_3btag_datamc',
        )

    def train_bdt(self, data):
        self.bdt.print_yields(data)
        self.bdt.train(data)
        self.bdt.print_results(data)


    @dependency(train_bdt)
    def build_bkg_model(self, data):
        bkg_model = EventFilter('bkg_model', filter=self.bdt.b)
        self.bkg_model = data.asmodel('bkg model').apply(bkg_model)
        self.bkg_model.apply(lambda t : t.reweight(self.bdt.reweight_tree))

    @dependency(build_bkg_model)
    def print_yields(self, signal, bkg_model):
        def get_yields(t, f_mask=None):
            weights = t.scale

            if f_mask is None:
                return ak.sum(weights)
            
            mask = f_mask(t)
            return ak.sum(weights[mask])

        bkg_yields = bkg_model.apply(partial(get_yields)).npy.sum()

        lumi = lumiMap[2018][0]
        sig_yields = lumi*signal.apply(partial(get_yields, f_mask=self.bdt.a)).npy.sum()

        print(f'Sig = {sig_yields:.2f}')
        print(f'Bkg = {bkg_yields:.2f}')
        print(f'S/B = {sig_yields/bkg_yields:.3f}')

    @dependency(build_bkg_model)
    def limits(self, signal, bkg_model):
        study.quick(
            signal+bkg_model,
            varlist=['dHH_HH_mass'],
            plot_scale=[100]*len(signal),
            limits=True,
            legend=True,
            saveas=f'{self.dout}/HH_mass_limits',
        )

    def train_vr_bdt(self, data):
        self.vr_bdt.print_yields(data)
        self.vr_bdt.train(data)
        self.vr_bdt.print_results(data)

    @dependency(train_vr_bdt)
    def build_vr_bkg_model(self, data):
        vr_bkg_model = EventFilter('vr_bkg_model', filter=self.vr_bdt.b)
        self.vr_bkg_model = data.asmodel('vr bkg model', color='salmon').apply(vr_bkg_model)
        self.vr_bkg_model.apply(lambda t : t.reweight(self.vr_bdt.reweight_tree))

    @dependency(build_vr_bkg_model)
    def plot_vr_model(self, data, vr_bkg_model):
        study.quick(
            data+vr_bkg_model,
            masks=[self.vr_bdt.a]*len(data),
            varlist=['dHH_HH_mass'],
            binlist=[(200,1800,30)],
            ratio=True, r_ylim=(0.7,1.3),
            legend=True,
            efficiency=True,
            saveas=f'{self.dout}/HH_mass_vr_model',
        )

        study.quick(
            data + vr_bkg_model,
            masks=[self.vr_bdt.a]*len(data),
            varlist=bdt_features,
            ratio=True, r_ylim=(0.7,1.3),
            legend=True,
            efficiency=True,
            saveas=f'{self.dout}/vr_model_bdt_features',
        )

if __name__ == '__main__':
    main()