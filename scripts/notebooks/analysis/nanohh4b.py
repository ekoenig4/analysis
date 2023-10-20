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

import utils.compat.tabulate as tabulate

from utils.notebookUtils import Notebook, required, dependency

def main():
    notebook = Analysis.from_parser()
    notebook.hello()
    notebook.run()


@cache_variable
def n_loose_btag(t):
    nL = 1*t.ak4_h1b1_btag_L + 1*t.ak4_h1b2_btag_L + 1*t.ak4_h2b1_btag_L + 1*t.ak4_h2b2_btag_L
    return ak.values_astype(nL, np.int32)

@cache_variable
def n_loose_leading_btag(t):
    bdisc = t.ak4_bdisc
    btag_L = t.ak4_btag_L
    nL = ak.sum( btag_L[ak.argsort(-bdisc, axis=1)][:,:4], axis=1 )
    return ak.values_astype(nL, np.int32)

@cache_variable
def n_medium_btag(t):
    nM = 1*t.ak4_h1b1_btag_M + 1*t.ak4_h1b2_btag_M + 1*t.ak4_h2b1_btag_M + 1*t.ak4_h2b2_btag_M
    return ak.values_astype(nM, np.int32)

@cache_variable
def n_medium_leading_btag(t):
    bdisc = t.ak4_bdisc
    btag_M = t.ak4_btag_M
    nM = ak.sum( btag_M[ak.argsort(-bdisc, axis=1)][:,:4], axis=1 )
    return ak.values_astype(nM, np.int32)

@cache_variable
def n_tight_btag(t):
    nT = 1*t.ak4_h1b1_btag_T + 1*t.ak4_h1b2_btag_T + 1*t.ak4_h2b1_btag_T + 1*t.ak4_h2b2_btag_T
    return ak.values_astype(nT, np.int32)

@cache_variable
def n_tight_leading_btag(t):
    bdisc = t.ak4_bdisc
    btag_T = t.ak4_btag_T
    nT = ak.sum( btag_T[ak.argsort(-bdisc, axis=1)][:,:4], axis=1 )
    return ak.values_astype(nT, np.int32)

@cache_variable(bins=(0,100,30))
def h_dm(t):
    return np.sqrt( (t.dHH_H1_regmass - 125)**2 + (t.dHH_H2_regmass - 120)**2 )

@cache_variable(bins=(0,100,30))
def vr_h_dm(t):
    return np.sqrt( (t.dHH_H1_regmass - 179)**2 + (t.dHH_H2_regmass - 172)**2 )

bdt_features = [
    'ak4_h1b1_regpt', 'ak4_h1b2_regpt', 'ak4_h2b1_regpt', 'ak4_h2b2_regpt',
    'dHH_H1_regmass', 'dHH_H2_regmass', 'dHH_H1_pt', 'dHH_H2_pt', 
    'dHH_HH_mass', 'dHH_HH_pt','dHH_SumRegPtb', 'dHH_SumRegResb',
    'dHH_H1b1_H1b2_deltaR', 'dHH_H2b1_H2b2_deltaR', 'dHH_H1_H2_deltaEta','dHH_mindRbb', 
    'dHH_maxdEtabb','dHH_absCosTheta_H1_inHHcm', 'dHH_absCosTheta_H1b1_inH1cm', 'dHH_NbtagT',
]

varinfo.dHH_HH_mass = dict(bins=(200,1800,30))
varinfo.dHH_HH_regmass = dict(bins=(200,1800,30))
varinfo.bdt_score = dict(bins=np.concatenate([np.arange(0,0.9,0.02), np.array([0.9,1.0])]))

def to_local(f):
    if f.startswith('/eos/user/e/ekoenig/'):
        return f.replace('/eos/user/e/ekoenig/','/store/user/ekoenig/')
    
    if f.startswith('/eos/user/m/mkolosov/'):
        return f.replace('/eos/user/m/mkolosov/','/store/user/ekoenig/')

class Analysis(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--dout', type=str, default='')
        parser.add_argument('--pairing', type=str, default='mindiag', )
        parser.add_argument('--load-init', type=int, default=7)

        parser.add_argument('--weights', nargs='+', default=[])

        parser.add_argument('--btagwp', type=str, default='medium')
        parser.add_argument('--leading-btag', action='store_true')
        parser.add_argument('--hh-mass-cut', type=str, default=None)

        parser.add_argument('--no-bkg', action='store_true')
        parser.add_argument('--no-data', action='store_true')
        parser.add_argument('--model', type=str, default=None)

        parser.add_argument('--bdisc-wp', type=float, default=None)

        parser.add_argument('--load-reweighter', type=str, default=None)
        parser.add_argument('--load-classifier', type=str, default=None)

        parser.add_argument('--n-classifier-estimators', type=int, default=200)

    @required
    def init(self):
        if self.load_init == 5: 
            self.dout = os.path.join(self.dout,'central')
        else:
            self.dout = os.path.join(self.dout,'private')

        init_version = f'_init_v{self.load_init}'
        init = getattr(self, init_version)
        init()

        if self.model is not None:
            if not os.path.exists(self.model):
                raise ValueError(f'Invalid model path: {self.model}')
            if 'feynnet' in self.model:
                self.pairing = 'feynnet'
            elif 'spanet' in self.model:
                self.pairing = 'spanet'
        self.dout = os.path.join("nanoHH4b",self.dout,self.pairing)


        def ttbar_subset(tree : Tree):
            if 'ttbar' not in tree.sample: return tree
            print(f'Taking 10% of ttbar: {len(tree)}')
            org_sumw = np.sum(tree.scale)
            tree = tree.subset(fraction=0.1, randomize=False)
            print(f'  => {len(tree)}')
            new_sumw = np.sum(tree.scale)
            tree.reweight(org_sumw / new_sumw)
            return tree
        # self.bkg = self.bkg.apply(ttbar_subset)

        btagwp = f'n_{self.btagwp}_btag'
        if self.leading_btag:
            btagwp = f'n_{self.btagwp}_leading_btag'
        self.n_btag = globals()[btagwp]

        btagwp = self.btagwp
        if self.bdisc_wp is not None:
            wptag = str(self.bdisc_wp).replace('.','p')
            btagwp = f'{self.btagwp}_bdisc{wptag}'
        if self.leading_btag:
            btagwp = f'leading_{btagwp}'
        self.dout = os.path.join(self.dout, btagwp)


        hparams = dict(
            n_estimators=70,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=300,
            gb_args=dict(subsample=0.6),
            n_folds=2,
        )
        
        self.bdt = ABCD(
            features=bdt_features,
            a=lambda t : (h_dm(t) <  25) & (self.n_btag(t) == 4),
            b=lambda t : (h_dm(t) <  25) & (self.n_btag(t) == 3),
            c=lambda t : (h_dm(t) >= 25) & (h_dm(t) < 50) & (self.n_btag(t) == 4),
            d=lambda t : (h_dm(t) >= 25) & (h_dm(t) < 50) & (self.n_btag(t) == 3),
            **hparams
        )

        self.vr_bdt = ABCD(
            features=bdt_features,
            a=lambda t : (vr_h_dm(t) <  25) & (self.n_btag(t) == 4),
            b=lambda t : (vr_h_dm(t) <  25) & (self.n_btag(t) == 3),
            c=lambda t : (vr_h_dm(t) >= 25) & (vr_h_dm(t) < 50) & (self.n_btag(t) == 4),
            d=lambda t : (vr_h_dm(t) >= 25) & (vr_h_dm(t) < 50) & (self.n_btag(t) == 3),
            **hparams
        )

        self.vr2_bdt = ABCD(
            features=bdt_features,
            a=lambda t : (h_dm(t) >= 25) & (h_dm(t) < 50) & (self.n_btag(t) == 4),
            b=lambda t : (h_dm(t) >= 25) & (h_dm(t) < 50) & (self.n_btag(t) == 3),
            c=lambda t : (h_dm(t) >= 50) & (h_dm(t) < 75) & (self.n_btag(t) == 4),
            d=lambda t : (h_dm(t) >= 50) & (h_dm(t) < 75) & (self.n_btag(t) == 3),
            **hparams
        )

    def _init_v0(self):
        treekwargs = dict(  
            weights=['xsecWeight/1000','genWeight','puWeight'],
            treename='Events',
            normalization='Runs:genEventCount',
        )

        f_pattern = '/eos/user/e/ekoenig/4BAnalysis/CMSSW_12_5_0/src/PhysicsTools/NanoHH4b/run/jobs_sig_{pairing}_2018_0L/mc/GluGluToHHTo4B_node_cHHH1_TuneCP5_13TeV-powheg-pythia8_1_tree.root'
        f_sig = f_pattern.format(pairing=self.pairing)
        self.signal = ObjIter([Tree( fc.fs.cernbox.fullpath(f_sig), **treekwargs)])

        # %%
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/bkg_{pairing}_2018_0L/mc/QCD*.root'
        f_bkg = f_pattern.format(pairing=self.pairing)
        self.bkg = ObjIter([Tree( fc.fs.cernbox.fullpath(f_bkg), **treekwargs)])
        # self.bkg = ObjIter([])

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/data_{pairing}_2018_0L/data/JetHT*.root'
        f_data = f_pattern.format(pairing=self.pairing)
        self.data = ObjIter([Tree( fc.fs.cernbox.fullpath(f_data), **dict(treekwargs, weights=None, color='black'))])

    def _init_v1(self):
        treekwargs = dict(
            weights=['xsecWeight/1000','genWeight','puWeight'],
            treename='Events',
            normalization=None,
        )
        
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/v1/{pairing}_sig_2018_0L/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(pairing=self.pairing)

        self.signal = ObjIter([Tree( fc.fs.cernbox.fullpath(f_sig), **treekwargs)])

        # %%
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/v1/{pairing}_bkg_2018_0L/mc/qcd-mg_tree.root'
        f_bkg = f_pattern.format(pairing=self.pairing)
        # self.bkg = ObjIter([Tree( fc.fs.cernbox.fullpath(f_bkg), **treekwargs)])
        self.bkg = ObjIter([])

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/v1/{pairing}_data_2018_0L/data/jetht_tree.root'
        f_data = f_pattern.format(pairing=self.pairing)
        self.data = ObjIter([Tree( fc.fs.cernbox.fullpath(f_data), **dict(treekwargs, weights=None, color='black'))])


    def _init_v2(self):
        treekwargs = dict(
            weights=['xsecWeight/1000','genWeight','puWeight'],
            treename='Events',
            normalization=None,
        )
        
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_sig_2018_0L/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(pairing=self.pairing)

        self.signal = ObjIter([Tree( fc.fs.cernbox.fullpath(f_sig), **treekwargs)])

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_2018_0L/mc/qcd-mg_tree.root'
        f_qcd = f_pattern.format(pairing=self.pairing)

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_2018_0L/mc/ttbar-powheg_tree.root'
        f_ttbar = f_pattern.format(pairing=self.pairing)
        
        if self.no_bkg:
            self.bkg = ObjIter([])
        else:
            self.bkg = ObjIter([Tree( fc.fs.cernbox.fullpath(f_qcd), **treekwargs), Tree( fc.fs.cernbox.fullpath(f_ttbar), **treekwargs)])

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_2018_0L/data/jetht_tree.root'
        f_data = f_pattern.format(pairing=self.pairing)
        
        if self.no_data:
            self.data = ObjIter([])
        else:
            self.data = ObjIter([Tree( fc.fs.cernbox.fullpath(f_data), **dict(treekwargs, weights=None, color='black'))])

    def _init_v3(self):
        treekwargs = dict(
            weights=['xsecWeight/1000','genWeight','puWeight'],
            treename='Events',
            normalization=None,
        )
        
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/trg/{pairing}_sig_2018_0L/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(pairing=self.pairing)

        self.signal = ObjIter([Tree( fc.fs.cernbox.fullpath(f_sig), **treekwargs)])

        # %%
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/trg/{pairing}_2018_0L/mc/qcd-mg_tree.root'
        f_qcd = f_pattern.format(pairing=self.pairing)

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/trg/{pairing}_2018_0L/mc/ttbar-powheg_tree.root'
        f_ttbar = f_pattern.format(pairing=self.pairing)
        
        if self.no_bkg:
            self.bkg = ObjIter([])
        else:
            self.bkg = ObjIter([Tree( fc.fs.cernbox.fullpath(f_qcd), **treekwargs), Tree( fc.fs.cernbox.fullpath(f_ttbar), **treekwargs)])

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/trg/{pairing}_2018_0L/data/jetht_tree.root'
        f_data = f_pattern.format(pairing=self.pairing)

        if self.no_data:
            self.data = ObjIter([])
        else:
            self.data = ObjIter([Tree( fc.fs.cernbox.fullpath(f_data), **dict(treekwargs, weights=None, color='black'))])

    def _init_v4(self):
        treekwargs = dict(
            weights=['xsecWeight/1000','genWeight','puWeight'],
            treename='Events',
            normalization=None,
        )

        base = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/trg/btagfix/skim_20231001'
        
        f_pattern = '{base}/{pairing}_sig_2018_0L/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(base=base, pairing=self.pairing)

        self.signal = ObjIter([Tree( fc.fs.cernbox.fullpath(f_sig), **treekwargs)])

        # %%
        f_pattern = '{base}/{pairing}_2018_0L/mc/qcd-mg_tree.root'
        f_qcd = f_pattern.format(base=base, pairing=self.pairing)

        f_pattern = '{base}/{pairing}_2018_0L/mc/ttbar-powheg_tree.root'
        f_ttbar = f_pattern.format(base=base, pairing=self.pairing)
        
        if self.no_bkg:
            self.bkg = ObjIter([])
        else:
            self.bkg = ObjIter([Tree( fc.fs.cernbox.fullpath(f_qcd), **treekwargs), Tree( fc.fs.cernbox.fullpath(f_ttbar), **treekwargs)])

        f_pattern = '{base}/{pairing}_2018_0L/data/jetht_tree.root'
        f_data = f_pattern.format(base=base, pairing=self.pairing)

        if self.no_data:
            self.data = ObjIter([])
        else:
            self.data = ObjIter([Tree( fc.fs.cernbox.fullpath(f_data), **dict(treekwargs, weights=None, color='black'))])


    def _init_v5(self):
        """
        DeepJet with MinDiag produced by Marina
        """

        treekwargs = dict(
            weights=['xsecWeight/1000','genWeight','puWeight'],
            treename='Events',
            normalization=None,
        )

        self.pairing = 'mindiag'

        base = '/eos/user/m/mkolosov/Run2_HHTo4B_NTuples/UL2018/BugFix_UL2018_2018_0L'
        
        f_pattern = '{base}/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(base=base)

        self.signal = ObjIter([Tree( fc.fs.cernbox.fullpath(f_sig), **treekwargs)])

        # %%
        f_pattern = '{base}/mc/qcd-mg_tree.root'
        f_qcd = f_pattern.format(base=base)

        f_pattern = '{base}/mc/ttbar-powheg_tree.root'
        f_ttbar = f_pattern.format(base=base)
        
        if self.no_bkg:
            self.bkg = ObjIter([])
        else:
            self.bkg = ObjIter([Tree( fc.fs.cernbox.fullpath(f_qcd), **treekwargs), Tree( fc.fs.cernbox.fullpath(f_ttbar), **treekwargs)])

        f_pattern = '{base}/data/jetht_tree.root'
        f_data = f_pattern.format(base=base)

        if self.no_data:
            self.data = ObjIter([])
        else:
            self.data = ObjIter([Tree( fc.fs.cernbox.fullpath(f_data), **dict(treekwargs, weights=None, color='black'))])
    
    def _init_v6(self):
        treekwargs = dict(
            weights=['xsecWeight/1000','genWeight','puWeight'],
            treename='Events',
            normalization=None,
        )

        base = f'/eos/user/e/ekoenig/Ntuples/NanoHH4b/run2/{self.pairing}_2018_0L'
        
        f_pattern = '{base}/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(base=base)

        self.signal = ObjIter([Tree( fc.fs.cernbox.fullpath(f_sig), **treekwargs)])

        # %%
        f_pattern = '{base}/mc/qcd-mg_tree.root'
        f_qcd = f_pattern.format(base=base)

        f_pattern = '{base}/mc/ttbar-powheg_tree.root'
        f_ttbar = f_pattern.format(base=base)
        
        if self.no_bkg:
            self.bkg = ObjIter([])
        else:
            self.bkg = ObjIter([Tree( fc.fs.cernbox.fullpath(f_qcd), **treekwargs), Tree( fc.fs.cernbox.fullpath(f_ttbar), **treekwargs)])

        f_pattern = '{base}/data/jetht_tree.root'
        f_data = f_pattern.format(base=base)

        if self.no_data:
            self.data = ObjIter([])
        else:
            self.data = ObjIter([Tree( fc.fs.cernbox.fullpath(f_data), **dict(treekwargs, weights=None, color='black'))])

    def _init_v7(self):

        weights = self.weights or ['btagSF_central','trgSF_central']
        weights.sort()
        if any(self.weights):
            self.dout = os.path.join(self.dout , '_'.join(weights))

        if self.pairing == 'mindiag':
            base = f'/eos/user/m/mkolosov/Run2_HHTo4B_NTuples/UL2018/DeepJet_woTrgMatching_17Oct2023_2018_0L/'
        elif self.pairing == 'pn_mindiag':
            base = f'/eos/user/m/mkolosov/Run2_HHTo4B_NTuples/UL2018/PNet_woTrgMatching_17Oct2023_2018_0L/'

        treekwargs = dict(
            weights=['xsecWeight/1000','genWeight','puWeight',] + weights,
            treename='Events',
            normalization=None,
        )


        f_pattern = '{base}/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(base=base)

        self.signal = ObjIter([Tree( fc.fs.cernbox.fullpath(f_sig), **treekwargs)])

        # %%
        f_pattern = '{base}/mc/qcd-mg_tree.root'
        f_qcd = f_pattern.format(base=base)

        f_pattern = '{base}/mc/ttbar-powheg_tree.root'
        f_ttbar = f_pattern.format(base=base)
        
        if self.no_bkg:
            self.bkg = ObjIter([])
        else:
            self.bkg = ObjIter([
                Tree( fc.fs.cernbox.fullpath(f_qcd), **treekwargs), 
                # Tree( fc.fs.cernbox.fullpath(f_ttbar), **treekwargs)
                ])

        f_pattern = '{base}/data/jetht_tree.root'
        f_data = f_pattern.format(base=base)

        if self.no_data:
            self.data = ObjIter([])
        else:
            self.data = ObjIter([Tree( fc.fs.cernbox.fullpath(f_data), **dict(treekwargs, weights=None, color='black'))])

    def _init_v8(self):
        self.dout = os.path.join(self.dout, 'daniel')
        self.pairing = 'mindiag'

        treekwargs = dict(
            weights=['genWeight','PUWeight'],
            treename='bbbbTree',
            normalization='eff_histo',
        )

        base = f'root://cmseos.fnal.gov//store/user/mkolosov/MultiHiggs/DiHiggs/RunII/NTuples/NTuple_UL2018_05Oct2023_withSFs/'
        
        f_pattern = '{base}/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/signal_merged_fnal_withSFs.root'
        f_sig = f_pattern.format(base=base)

        self.signal = ObjIter([Tree( f_sig, **treekwargs, sample='ggHH4b')])

        # %%
        f_pattern = '{base}/mc/qcd-mg_tree.root'
        f_qcd = f_pattern.format(base=base)

        f_pattern = '{base}/mc/ttbar-powheg_tree.root'
        f_ttbar = f_pattern.format(base=base)
        
        self.no_bkg = True
        if self.no_bkg:
            self.bkg = ObjIter([])
        else:
            self.bkg = ObjIter([Tree( fc.fs.cernbox.fullpath(f_qcd), **treekwargs), Tree( fc.fs.cernbox.fullpath(f_ttbar), **treekwargs)])

        f_pattern = 'root://cmseos.fnal.gov//store/user/mkolosov/MultiHiggs/DiHiggs/RunII/NTuples/NTuple_UL2018_05Oct2023_v2/Data/ntuple.root'
        f_data = f_pattern.format(base=base)

        if self.no_data:
            self.data = ObjIter([])
        else:
            self.data = ObjIter([Tree( f_data, **dict(treekwargs, weights=None, normalization=None, color='black', sample='jetht', xsec=1, is_data=True))])

        from utils.fourbUtils.bbbbUtils import map_to_nano
        (self.signal + self.data).apply(map_to_nano)

        bbbb_lepton_veto = EventFilter('lepton_Veto', filter=lambda t : (t.IsolatedMuon_Multiplicity == 0) & (t.IsolatedElectron_Multiplicity == 0), verbose=True)
        self.signal = self.signal.apply(bbbb_lepton_veto)
        self.bkg = self.bkg.apply(bbbb_lepton_veto)
        self.data = self.data.apply(bbbb_lepton_veto)

    @required
    def apply_trigger(self):
        if self.load_init >= 3:
            return

        trigger = EventFilter('trigger', filter=lambda t : t.passTrig0L, verbose=True)
        self.signal = self.signal.apply(trigger)
        self.bkg = self.bkg.apply(trigger)
        self.data = self.data.apply(trigger)

    ################################
    # B Discriminator Optimization #
    ################################
    @required
    def set_bdisc_threshold(self, signal, bkg, data):
        if self.bdisc_wp is None:
            return
        
        def new_threshold(tree):
            tree.extend(
                ak4_h1b1_btag_M = ak.where(tree.ak4_h1b1_bdisc > self.bdisc_wp, 1, 0),
                ak4_h1b2_btag_M = ak.where(tree.ak4_h1b2_bdisc > self.bdisc_wp, 1, 0),
                ak4_h2b1_btag_M = ak.where(tree.ak4_h2b1_bdisc > self.bdisc_wp, 1, 0),
                ak4_h2b2_btag_M = ak.where(tree.ak4_h2b2_bdisc > self.bdisc_wp, 1, 0),
            )

        (signal+bkg+data).apply(new_threshold)

        event_filter = EventFilter(f'exactly_3_btag{self.bdisc_wp}', filter=lambda t : self.n_btag(t) >= 3, verbose=True)
        self.signal = signal.apply(event_filter)
        self.bkg = bkg.apply(event_filter)
        self.data = data.apply(event_filter)

    @required
    def load_feynnet(self, signal, bkg, data):
        if self.model is None:
            return

        if 'feynnet' not in self.model:
            return

        if self.model.endswith('/'): self.model=self.model[:-1]

        import utils.resources as rsc
        accelerator = 'cpu' if rsc.ngpus == 0 else 'cuda'

        if self.model.endswith('onnx') or self.model.endswith('onnx/'):
            load_feynnet = fourb.nanohh4b.f_evaluate_feynnet( os.path.dirname(self.model), 'onnx', accelerator=accelerator )
        elif self.model.endswith('predict') or self.model.endswith('predict/'):
            load_feynnet = fourb.nanohh4b.f_evaluate_feynnet( os.path.dirname(self.model), 'predict', accelerator=accelerator )
        else:
            load_feynnet = fourb.nanohh4b.f_evaluate_feynnet(self.model, accelerator=accelerator)


        import multiprocess as mp
        if accelerator == 'cuda': 
            (signal+bkg+data).apply(load_feynnet, report=True)
        else:
            nprocs = min(rsc.ncpus, len(signal+bkg+data))
            with mp.Pool(nprocs) as pool:
                (signal+bkg+data).parallel_apply(load_feynnet, pool=pool, report=True)

        import json
        study.save_file( json.dumps(load_feynnet.metadata, indent=2), saveas=f'{self.dout}/feynnet_metadata', fmt=['txt'])

    @required
    def load_spanet(self, signal, bkg, data):
        if self.model is None:
            return
        
        if 'spanet' not in self.model:
            return

        load_spanet = fourb.nanohh4b.f_evaluate_spanet(self.model)
        import utils.resources as rsc
        nprocs = min(rsc.ncpus, len(signal+bkg+data))
        with mp.Pool(nprocs) as pool:
            (signal+bkg+data).parallel_apply(load_spanet, pool=pool, report=True)

    @required
    def hh_mass_cut(self, signal, bkg, data):
        if self.hh_mass_cut is None:
            return

        operators = {
            ">=" : "ge",
            "<=" : "le",
            ">" : "gt",
            "<" : "lt",
            None : None,
        }

        for operator, tag in operators.items():
            if self.hh_mass_cut.startswith(operator):
                break

        tag = self.hh_mass_cut.replace(operator, tag).replace('.','p')
        self.dout = os.path.join(self.dout, f'hh_mass_{tag}')
        
        hh_mass_cut = eval(f"lambda t : t.dHH_HH_regmass{self.hh_mass_cut}")
        hh_mass_cut = EventFilter(f'hh_mass_cut_{tag}', filter=hh_mass_cut, verbose=True)
        self.signal = signal.apply(hh_mass_cut)
        self.bkg = bkg.apply(hh_mass_cut)
        self.data = data.apply(hh_mass_cut)

    def plot_jet_multiplicity(self, signal, bkg, data):
        study.quick(
            signal + bkg + data, legend=True,
            varlist=['ak.num(ak4_pt, axis=1)'],
            xlabels=['Jet Multiplicity'],
            log=True, ylim=(5e-1, 5e9),
            **study.datamc, r_ylim=(0.5, 1.5),
            saveas=f'{self.dout}/jet_multiplicity',
        ) 
        study.quick(
            signal + bkg + data, legend=True,
            efficiency=True,
            varlist=['ak.num(ak4_pt, axis=1)'],
            xlabels=['Jet Multiplicity'],
            **study.datamc, r_ylim=(0.5, 1.5),
            saveas=f'{self.dout}/jet_multiplicity_shape',
        ) 

    def plot_reco_eff(self, signal):
        signal.apply(fourb.nanohh4b.match_ak4_gen)

        study.quick(
            signal,
            varlist=['nfound_select', 'nfound_paired'],
            legend=True,
            efficiency=True,
            saveas=f'{self.dout}/gen_match_eff',
        )

        study.quick(
            signal,
            masks=lambda t : t.nfound_select==4,
            varlist=['nfound_select', 'nfound_paired'],
            h_label_stat=lambda h : f'{h.histo[-1]:0.2%}',
            legend=True,
            efficiency=True,
            saveas=f'{self.dout}/reco_eff',
        )

        study.quick(
            signal,
            masks=lambda t : t.nfound_select==4,
            varlist=['dHH_H1_regmass','dHH_H2_regmass'],
            legend=True,
            efficiency=True,
            saveas=f'{self.dout}/reco_eff_higgs_mass',
        )

        study.compare_masks(
            signal,
            masks=[None,'nfound_paired >= 1','nfound_paired == 2'],
            varlist=['dHH_H1_regmass','dHH_H2_regmass'],
            binlist=[(0,300,30)]*2,
            legend=True,
            saveas=f'{self.dout}/gen_matched_eff_stacked_higgs_mass',
        )
        study.compare_masks(
            signal,
            masks=['nfound_paired == 0','nfound_paired == 1','nfound_paired == 2'],
            varlist=['dHH_H1_regmass','dHH_H2_regmass'],
            binlist=[(0,300,30)]*2,
            legend=True,
            saveas=f'{self.dout}/gen_matched_eff_higgs_mass',
        )

    def plot_higgs(self, signal, bkg):
        study.quick(
            signal+bkg,
            varlist=['dHH_H1_regmass','dHH_H2_regmass'],
            binlist=[(0,300,30)]*2,
            efficiency=True,
            legend=True,
            saveas=f'{self.dout}/higgs_mass',
        )

        study.quick2d(
            signal+bkg,
            varlist=['dHH_H1_regmass','dHH_H2_regmass'],
            binlist=[(0,300,30)]*2,
            legend=True,
            saveas=f'{self.dout}/higgs_mass2d',
        )

    def plot_region_vars(self, signal, bkg):
        study.quick(
            signal+bkg,
            varlist=[n_loose_btag, n_medium_btag, n_tight_btag],
            efficiency=True,
            legend=True,
            saveas=f'{self.dout}/n_btag',
        )

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

    def print_4btag_yields(self, signal, bkg, data):
        # NOTE: yields from AN2019_250_v6:Table 16
        an_yields = {
            'ggHH4b' : 1338.9 + 527.1,
            'qcd-mg' : 86381.0 + 39741.9,
            'ttbar-powheg' : 5369.2 + 8648.7,
            'jetht': 164307.0 + 96274.0,
        }

        def get_yield(tree):
            label = tree.sample
            an = an_yields.get(label, -1)

            mask = self.n_btag(tree) == 4
            scale = tree.scale[mask]

            if not tree.is_data:
                lumi = lumiMap[2018][0]
                scale = lumi * scale

            if tree.is_signal:
                label = f'{label} (x100)'
                scale = 100 * scale

            events = np.sum(scale)

            return (label, events, an, events/an)

        table = (signal+bkg+data).apply(get_yield).list
        table = tabulate.tabulate(table, headers=['sample','yield','an yield', 'this/an'], tablefmt='simple', numalign='right', floatfmt='.2f')
        print(table)
        study.save_file(table, os.path.join(self.dout, 'presel_4btag_yields'), fmt=['txt'])

    def plot_4b_control(self, signal, bkg, data):

        bins = np.concatenate([np.arange(0, 600, 25), np.arange(600, 850, 50)])
        study.quick(
            signal + bkg + data, legend=True,
            suptitle='4 b-tag Control Plots',
            masks=lambda t : self.n_btag(t) == 4,
            varlist=['dHH_H1_regpt','dHH_H2_regpt','dHH_H1_pt','dHH_H2_pt'],
            binlist=[bins]*4,
            log=True, ylim=(5e-1, 5e8),
            **study.datamc, r_ylim=(0.5, 1.5),
            saveas=f'{self.dout}/control_plots_4b',
        ) 

    def print_3btag_yields(self, signal, bkg, data):
        # NOTE: yields from AN2019_250_v6:Table 19
        an_yields = {
            'ggHH4b' : 2404.3 + 78.7,
            'qcd-mg' : 1058750.0 + 43059.3,
            'ttbar-powheg' : 111774.6 + 6518.1,
            'jetht': 2273811.0 + 107918.0,
        }
        def get_yield(tree):
            label = tree.sample
            an = an_yields.get(label, -1)

            mask = self.n_btag(tree) == 3
            scale = tree.scale[mask]

            if not tree.is_data:
                lumi = lumiMap[2018][0]
                scale = lumi * scale

            if tree.is_signal:
                label = f'{label} (x100)'
                scale = 100 * scale

            events = np.sum(scale)

            return (label, events, an, events/an)

        table = (signal+bkg+data).apply(get_yield).list
        table = tabulate.tabulate(table, headers=['sample','yield','an yield', 'this/an'], tablefmt='simple', numalign='right', floatfmt='.2f')
        print(table)
        study.save_file(table, os.path.join( self.dout, 'presel_3btag_yields.txt'), fmt=['txt'])
    
    def plot_3b_control(self, signal, bkg, data):
        bins = np.concatenate([np.arange(0, 600, 25), np.arange(600, 850, 50)])
        study.quick(
            signal + bkg + data, legend=True,
            suptitle='3 b-tag Control Plots',
            masks=lambda t : self.n_btag(t) == 3,
            varlist=['dHH_H1_regpt','dHH_H2_regpt','dHH_H1_pt','dHH_H2_pt'],
            binlist=[bins]*4,
            log=True, ylim=(5e-1, 5e8),
            **study.datamc, r_ylim=(0.5, 1.5),
            saveas=f'{self.dout}/control_plots_3b',
        ) 

    @required
    def blind_data(self, data):
        blind_data = EventFilter('blinded', filter=lambda t : ~self.bdt.a(t))
        self.data = data.apply(blind_data)

    def print_abcd_yields(self, signal, bkg, data):
        # NOTE: yields from AN2019_250_v6:Table 23 - 26
        an_yields = dict(
            a = {
                'ggHH4b' : 1189.0 + 44.8,
                'qcd-mg' : 4391.3 + 289.8,
                'ttbar-powheg' : 1379.3 + 85.5,
                'jetht': 0,
            },
            b = {
                'ggHH4b' : 788.5 + 26.4,
                'qcd-mg' : 32113.6 + 1345.0,
                'ttbar-powheg' : 14435.2 + 642.3,
                'jetht': 75628.0 + 3538.0,
            },
            c = {
                'ggHH4b' : 364.2 + 14.1,
                'qcd-mg' : 14384.3 + 582.8,
                'ttbar-powheg' : 2589.0 + 154.8,
                'jetht': 28150.0 + 1616.0,
            },
            d = {
                'ggHH4b' : 434.1 + 12.9,
                'qcd-mg' : 87038.1 + 3393.5,
                'ttbar-powheg' : 26304.3 + 1157.0,
                'jetht': 208145.0 + 9144.0,
            }
        )

        def get_yield(tree, region):
            label = tree.sample
            an = an_yields[region].get(label, -1)

            mask = getattr(self.bdt, region)(tree)
            scale = tree.scale[mask]

            if not tree.is_data:
                lumi = lumiMap[2018][0]
                scale = lumi * scale

            if tree.is_signal:
                label = f'{label} (x100)'
                scale = 100 * scale

            events = np.sum(scale)

            return (label, events, an, events/an)

        tables = []
        for region in ['a','b','c','d']:
            table = (signal+bkg+data).apply(partial(get_yield, region=region)).list
            table = tabulate.tabulate(table, headers=['sample','yield','an yield', 'this/an'], tablefmt='simple', numalign='right', floatfmt='.2f')

            name = dict(
                a='A_SR(4b)',
                b='A_SR(3b)',
                c='A_CR(4b)',
                d='A_CR(3b)',
            ).get(region)
            print('Region:', name)
            print(table)
            tables.append(f'Region: {name}\n' + str(table) + '\n\n')
        tables = '\n'.join(tables)
        study.save_file(tables, os.path.join( self.dout, 'abcd_yields'), fmt=['txt'])

    def plot_3btag_datamc(self, data, bkg):
        study.quick(
            data+bkg,
            masks=lambda t : self.n_btag(t) == 3,
            varlist=['dHH_H1_regmass','dHH_H2_regmass'],
            binlist=[(0,300,30)]*2,
            efficiency=True,
            log=True,
            ratio=True, 
            legend=True,

            saveas=f'{self.dout}/higgs_mass_3btag_datamc',
        )

        study.quick2d(
            data+bkg,
            masks=lambda t : self.n_btag(t) == 3,
            varlist=['dHH_H1_regmass','dHH_H2_regmass'],
            binlist=[(0,300,30)]*2,
            legend=True,

            saveas=f'{self.dout}/higgs_mass2d_3btag_datamc',
        )

    def train_bdt(self, data):

        import utils.config as cfg
        default_file = os.path.join(cfg.GIT_WD, 'models', self.dout, 'bdt_reweighter.pkl')
        if not self.load_reweighter:
            if os.path.exists(default_file):
                self.load_reweighter = default_file

        if self.load_reweighter:
            print('Loading reweighter from', self.load_reweighter)
            self.bdt = self.bdt.load(self.load_reweighter)
            self.bdt.print_yields(data)
        else:
            self.bdt.print_yields(data)
            self.bdt.train(data)
            self.bdt.save( os.path.join(self.dout, 'bdt_reweighter') )

        self.bdt.print_results(data)

    @dependency(train_bdt)
    def build_bkg_model(self, signal, bkg, data):
        a_sr_filter = EventFilter('A_SR(4b)', filter=self.bdt.a)
        self.signal = signal.apply(a_sr_filter)
        self.bkg = bkg.apply(a_sr_filter)

        bkg_model = EventFilter('bkg_model', filter=self.bdt.b)
        self.bkg_model = data.asmodel('bkg model').apply(bkg_model)
        self.bkg_model.apply(lambda t : t.reweight(self.bdt.reweight_tree))

    @dependency(build_bkg_model)
    def print_sr_yields(self, signal, bkg_model):

        def get_yields(t, f_mask=None):
            weights = t.scale

            if f_mask is not None:
                weights = weights[f_mask(t)]

            if not t.is_data:
                lumi = lumiMap[2018][0]
                weights = lumi * weights

            sumw = np.sum(weights)
            error = np.sqrt(np.sum(weights**2))
            return (t.sample, sumw, error)
        rows = (signal + bkg_model).apply(get_yields).list
        rows = tabulate.tabulate(rows, headers=['sample','yield','error'], tablefmt='simple', numalign='right', floatfmt='.2f')
        print(rows)
        study.save_file(rows, os.path.join(self.dout, 'sr_yields'), fmt=['txt'])

    @dependency(build_bkg_model, print_abcd_yields)
    def plot_limits(self, signal, bkg_model):
        model = []
        study.quick(
            signal+bkg_model,
            varlist=['dHH_HH_regmass'],
            plot_scale=[100]*len(signal),
            limits=True,
            l_store=model,
            legend=True,
            ylim=(0, 3000),
            saveas=f'{self.dout}/HH_mass_limits',
        )

        model = model[0][0]

        info = dict(
            sig_yield = np.sum(model.h_sig.histo),
            bkg_yield = np.sum(model.h_bkg.histo),
            exp_lim = model.h_sig.stats.exp_limits[2],
        )
        import pickle

        if not os.path.exists(self.dout):
            os.makedirs(self.dout)
        with open(f'{self.dout}/limit_values.pkl', 'wb') as f:
            pickle.dump(info, f)

    @dependency(build_bkg_model)
    def write_features(self, signal, bkg_model):
        signal.apply(lambda t : t.extend(label=np.ones(len(t))))
        bkg_model.apply(lambda t : t.extend(label=np.zeros(len(t))))

        signal.write('bdt_features_{base}', include=bdt_features+['scale','label'])
        bkg_model.write('bdt_features_{base}', include=bdt_features+['scale','label'])

    @dependency(build_bkg_model)
    def train_bdt_classifier(self, signal, bkg_model):
        import utils.config as cfg
        default_file = os.path.join(cfg.GIT_WD, 'models', self.dout, 'bdt_classifier.pkl')
        if not self.load_classifier:
            if os.path.exists(default_file):
                self.load_classifier = default_file

        if self.load_classifier:
            print('Loading classifier from', self.load_classifier)
            self.bdt_classifier = KFoldBDTClassifier.load(self.load_classifier)
        else:
            self.bdt_classifier = KFoldBDTClassifier(
                features=bdt_features,
                kfold=3,
                n_estimators=self.n_classifier_estimators,
            )

            self.bdt_classifier.train(bkg_model, signal)
            self.bdt_classifier.save( os.path.join(self.dout, 'bdt_classifier') )

        self.bdt_classifier.print_results(bkg_model, signal)
        (signal + bkg_model).apply(lambda t : t.extend(bdt_score=self.bdt_classifier.predict_tree(t)))

    @dependency(train_bdt_classifier)
    def plot_bdt_classifier_validation(self, signal, bkg_model):
        sig_weights = signal.scale.cat * lumiMap[2018][0]
        sig_index = signal.apply(lambda t : np.arange(len(t))).cat % self.bdt_classifier.kfold
        sig_bdt_scores = [ signal.apply(bdt.predict_tree).cat for bdt in self.bdt_classifier.bdts ]

        bkg_weights = bkg_model.scale.cat
        bkg_index = bkg_model.apply(lambda t : np.arange(len(t))).cat % self.bdt_classifier.kfold
        bkg_bdt_scores = [ bkg_model.apply(bdt.predict_tree).cat for bdt in self.bdt_classifier.bdts ]

        for i in range(self.bdt_classifier.kfold):
        # trained discriminant in train/test samples
            fig, _, _ = hist_multi(
                [sig_bdt_scores[i][sig_index != i], bkg_bdt_scores[i][bkg_index != i], sig_bdt_scores[i][sig_index == i], bkg_bdt_scores[i][bkg_index == i]],
                weights = [sig_weights[sig_index != i], bkg_weights[bkg_index != i], sig_weights[sig_index == i], bkg_weights[bkg_index == i]],
                bins=(0, 1, 21), stacked=False,
                title=f'BDT Classifier (Fold {i})',
                is_data = [False, False, True, True],
                h_label=['S (Train)', 'B (Train)', 'S (Test)', 'B (Test)'],
                h_color=['blue', 'red', 'blue', 'red'],
                h_alpha=[0.5, 0.5, 1.0, 1.0],
                h_histtype=['stepfilled', 'stepfilled', None, None],
                efficiency=True, legend=True,

                ratio=True, r_group=((0, 2), (1, 3)), r_ylabel='Test/Train',
                
                empirical=True, e_show=False,
                e_correlation=True, e_c_method='roc', e_c_group=[(2,3), (0,1)],
                e_c_o_label=['Train','Test'],
                e_c_o_linestyle=['-','--'],
                e_c_o_color='black',
                e_c_o_alpha=[0.8, 1.0],
                e_c_label_stat='area',
            )
            study.save_fig(fig, f'{self.dout}/bdt_classifier_{i}_train_test')

            for j in range(i + 1, self.bdt_classifier.kfold):
                fig, _, _ = hist_multi(
                    [sig_bdt_scores[j][sig_index == j], bkg_bdt_scores[j][bkg_index == j], sig_bdt_scores[i][sig_index == i], bkg_bdt_scores[i][bkg_index == i]],
                    weights = [sig_weights[sig_index == j], bkg_weights[bkg_index == j], sig_weights[sig_index == i], bkg_weights[bkg_index == i]],
                    bins=(0, 1, 21), stacked=False,
                    is_data = [False, False, True, True],
                    title=f'BDT Classifier (Fold {i} vs {j})',
                    h_label=[f'S (C{j})', f'B (C{j})', f'S (C{i})', f'B (C{i})'],
                    h_color=['blue', 'red', 'blue', 'red'],
                    h_alpha=[0.5, 0.5, 1.0, 1.0],
                    h_histtype=['stepfilled', 'stepfilled', None, None],
                    efficiency=True, legend=True,

                    ratio=True, r_group=((0, 2), (1, 3)), r_ylabel=f'C{j}/C{i}',
                    
                    empirical=True, e_show=False,
                    e_correlation=True, e_c_method='roc', e_c_group=((0,1), (2, 3)),
                    e_c_o_label=['C2','C1'],
                    e_c_o_linestyle=['-','--'],
                    e_c_o_color='black',
                    e_c_o_alpha=[0.8, 1.0],
                    e_c_label_stat='area',
                )
                study.save_fig(fig, f'{self.dout}/bdt_classifier_{i}_vs_{j}')

    @dependency(train_bdt_classifier)
    def plot_bdt_classifier_limits(self, signal, bkg_model):
        model = []

        study.quick(
            signal+bkg_model,
            varlist=['bdt_score'],
            plot_scale=[100]*len(signal),
            binlist=[(0,1,30)],
            limits=True,
            l_store=model,
            legend=True,
            log=True, ylim=(0.5e-2, 2e6),
            saveas=f'{self.dout}/bdt_score_limits',
        )
        
        # study.quick(
        #     signal+bkg_model,
        #     varlist=['bdt_score'],
        #     plot_scale=[100]*len(signal),
        #     binlist=[(0,1,30)],
        #     limits=True,
        #     legend=True,
        #     ylim=(0, 750),
        #     saveas=f'{self.dout}/bdt_score_limits_zoomed',
        # )
        
        study.quick(
            signal+bkg_model,
            varlist=['bdt_score'],
            limits=True,
            legend=True,
            log=True, ylim=(0.5e-2, 2e6),
            saveas=f'{self.dout}/bdt_score_limits_run2',
        )

        model = model[0][0]

        info = dict(
            sig_yield = np.sum(model.h_sig.histo),
            bkg_yield = np.sum(model.h_bkg.histo),
            exp_lim = model.h_sig.stats.exp_limits[2],
        )
        import pickle

        if not os.path.exists(self.dout):
            os.makedirs(self.dout)
        with open(f'{self.dout}/bdt_limit_values.pkl', 'wb') as f:
            pickle.dump(info, f)


    def train_vr_bdt(self, data):
        self.vr_bdt.print_yields(data)
        self.vr_bdt.train(data)
        self.vr_bdt.print_results(data)

    @dependency(train_vr_bdt)
    def build_vr_bkg_model(self, data):
        # vr_bkg_model = EventFilter('vr_bkg_model', filter=self.vr_bdt.b)
        self.vr_bkg_model = data.asmodel('vr bkg model', color='salmon')#.apply(vr_bkg_model)
        # self.vr_bkg_model.apply(lambda t : t.reweight(self.vr_bdt.reweight_tree))

    @dependency(build_vr_bkg_model)
    def plot_vr_model(self, data, vr_bkg_model):
        study.quick(
            data+vr_bkg_model,
            masks=[self.vr_bdt.a]*len(data)+[self.vr_bdt.b]*len(vr_bkg_model),
            varlist=['dHH_HH_mass','dHH_HH_pt','ak4_h1b1_regpt'],
            binlist=[(200,1800,30),(0,400,30),(0,350,30)],
            ratio=True, r_ylim=(0.7,1.3),
            legend=True,
            efficiency=True,
            **study.kstest,
            saveas=f'{self.dout}/vr_norm_model',
        )

        study.quick(
            data+vr_bkg_model,
            masks=[self.vr_bdt.a]*len(data)+[self.vr_bdt.b]*len(vr_bkg_model),
            scale=[None]*len(data)+[self.vr_bdt.reweight_tree]*len(vr_bkg_model),
            varlist=['dHH_HH_mass','dHH_HH_pt','ak4_h1b1_regpt'],
            binlist=[(200,1800,30),(0,400,30),(0,350,30)],
            ratio=True, r_ylim=(0.7,1.3),
            legend=True,
            efficiency=True,
            **study.kstest,
            saveas=f'{self.dout}/vr_bdt_model',
        )
        

        study.quick(
            data + vr_bkg_model,
            masks=[self.vr_bdt.a]*len(data)+[self.vr_bdt.b]*len(vr_bkg_model),
            scale=[None]*len(data)+[self.vr_bdt.reweight_tree]*len(vr_bkg_model),
            varlist=bdt_features,
            ratio=True, r_ylim=(0.7,1.3),
            legend=True,
            efficiency=True,
            saveas=f'{self.dout}/vr_model_bdt_features',
        )

    def train_vr2_bdt(self, data):
        self.vr2_bdt.print_yields(data)
        self.vr2_bdt.train(data)
        self.vr2_bdt.print_results(data)

    @dependency(train_vr2_bdt)
    def build_vr2_bkg_model(self, data):
        # vr_bkg_model = EventFilter('vr_bkg_model', filter=self.vr_bdt.b)
        self.vr2_bkg_model = data.asmodel('vr2 bkg model', color='salmon')#.apply(vr_bkg_model)
        # self.vr_bkg_model.apply(lambda t : t.reweight(self.vr_bdt.reweight_tree))

    @dependency(build_vr2_bkg_model)
    def plot_vr2_model(self, data, vr2_bkg_model):
        study.quick(
            data+vr2_bkg_model,
            masks=[self.vr_bdt.a]*len(data)+[self.vr2_bdt.b]*len(vr2_bkg_model),
            varlist=['dHH_HH_mass','dHH_HH_pt','ak4_h1b1_regpt'],
            binlist=[(200,1800,30),(0,400,30),(0,350,30)],
            ratio=True, r_ylim=(0.7,1.3),
            legend=True,
            efficiency=True,
            **study.kstest,
            saveas=f'{self.dout}/vr2_norm_model',
        )

        study.quick(
            data+vr2_bkg_model,
            masks=[self.vr2_bdt.a]*len(data)+[self.vr2_bdt.b]*len(vr2_bkg_model),
            scale=[None]*len(data)+[self.vr2_bdt.reweight_tree]*len(vr2_bkg_model),
            varlist=['dHH_HH_mass','dHH_HH_pt','ak4_h1b1_regpt'],
            binlist=[(200,1800,30),(0,400,30),(0,350,30)],
            ratio=True, r_ylim=(0.7,1.3),
            legend=True,
            efficiency=True,
            **study.kstest,
            saveas=f'{self.dout}/vr2_bdt_model',
        )
        

        study.quick(
            data + vr2_bkg_model,
            masks=[self.vr2_bdt.a]*len(data)+[self.vr2_bdt.b]*len(vr2_bkg_model),
            scale=[None]*len(data)+[self.vr2_bdt.reweight_tree]*len(vr2_bkg_model),
            varlist=bdt_features,
            ratio=True, r_ylim=(0.7,1.3),
            legend=True,
            efficiency=True,
            saveas=f'{self.dout}/vr2_model_bdt_features',
        )
    


if __name__ == '__main__':
    main()