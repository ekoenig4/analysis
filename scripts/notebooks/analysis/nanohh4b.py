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
import tabulate


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

@cache_variable
def n_tight_btag(t):
    nT = t.ak4_h1b1_btag_T + t.ak4_h1b2_btag_T + t.ak4_h2b1_btag_T + t.ak4_h2b2_btag_T
    return ak.values_astype(nT, np.int32)

@cache_variable(bins=(0,100,30))
def h_dm(t):
    return np.sqrt( (t.dHH_H1_mass - 125)**2 + (t.dHH_H2_mass - 120)**2 )

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
        parser.add_argument('--load-init', type=int, default=3)
        parser.add_argument('--btagwp', type=str, default='medium')
        parser.add_argument('--no-bkg', action='store_true')
        parser.add_argument('--no-data', action='store_true')
        parser.add_argument('--model', type=str, default=None)

        parser.add_argument('--bdisc-wp', type=float, default=None)

    @required
    def init(self):
        self.dout = os.path.join("nanoHH4b",self.dout,self.pairing)

        if self.model is not None:
            if not os.path.exists(self.model):
                raise ValueError(f'Invalid model path: {self.model}')
        
        if self.bdisc_wp is not None:
            wptag = str(self.bdisc_wp).replace('.','p')
            self.dout = os.path.join(self.dout, f'{self.btagwp}_bdisc{wptag}/')
        else:
            self.dout = os.path.join(self.dout, self.btagwp)

        init_version = f'_init_v{self.load_init}'
        init = getattr(self, init_version)
        init()

        if self.btagwp == 'loose':
            self.n_btag = n_loose_btag
        else:
            self.n_btag = n_medium_btag

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

    def _init_v0(self):
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

    def _init_v1(self):
        treekwargs = dict(
            use_gen=False,
            treename='Events',
            normalization=None,
        )
        
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/v1/{pairing}_sig_2018_0L/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(pairing=self.pairing)

        self.signal = ObjIter([Tree( get_local_alt(f_sig), **treekwargs)])

        # %%
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/v1/{pairing}_bkg_2018_0L/mc/qcd-mg_tree.root'
        f_bkg = f_pattern.format(pairing=self.pairing)
        # self.bkg = ObjIter([Tree( get_local_alt(f_bkg), **treekwargs)])
        self.bkg = ObjIter([])

        # signal xsec is set to 0.010517 pb -> 31.05 fb * (0.58)^2 
        (self.signal).apply(lambda t : t.reweight(t.genWeight * t.xsecWeight / 1000))
        (self.bkg).apply(lambda t : t.reweight(t.genWeight * t.xsecWeight / 1000))

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/v1/{pairing}_data_2018_0L/data/jetht_tree.root'
        f_data = f_pattern.format(pairing=self.pairing)
        self.data = ObjIter([Tree( get_local_alt(f_data), **dict(treekwargs, normalization=None, color='black'))])


    def _init_v2(self):
        treekwargs = dict(
            use_gen=False,
            treename='Events',
            normalization=None,
        )
        
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_sig_2018_0L/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(pairing=self.pairing)

        self.signal = ObjIter([Tree( get_local_alt(f_sig), **treekwargs)])

        # %%
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_2018_0L/mc/qcd-mg_tree.root'
        f_qcd = f_pattern.format(pairing=self.pairing)

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_2018_0L/mc/ttbar-powheg_tree.root'
        f_ttbar = f_pattern.format(pairing=self.pairing)
        
        if self.no_bkg:
            self.bkg = ObjIter([])
        else:
            self.bkg = ObjIter([Tree( get_local_alt(f_qcd), **treekwargs), Tree( get_local_alt(f_ttbar), **treekwargs)])

        # signal xsec is set to 0.010517 pb -> 31.05 fb * (0.58)^2 
        (self.signal).apply(lambda t : t.reweight(t.genWeight * t.xsecWeight / 1000))
        (self.bkg).apply(lambda t : t.reweight(t.genWeight * t.xsecWeight / 1000))

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/{pairing}_2018_0L/data/jetht_tree.root'
        f_data = f_pattern.format(pairing=self.pairing)
        
        if self.no_data:
            self.data = ObjIter([])
        else:
            self.data = ObjIter([Tree( get_local_alt(f_data), **dict(treekwargs, normalization=None, color='black'))])

    def _init_v3(self):
        treekwargs = dict(
            use_gen=False,
            treename='Events',
            normalization=None,
        )
        
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/trg/{pairing}_sig_2018_0L/mc/ggHH4b_tree.root'
        f_sig = f_pattern.format(pairing=self.pairing)

        self.signal = ObjIter([Tree( get_local_alt(f_sig), **treekwargs)])

        # %%
        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/trg/{pairing}_2018_0L/mc/qcd-mg_tree.root'
        f_qcd = f_pattern.format(pairing=self.pairing)

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/trg/{pairing}_2018_0L/mc/ttbar-powheg_tree.root'
        f_ttbar = f_pattern.format(pairing=self.pairing)
        
        if self.no_bkg:
            self.bkg = ObjIter([])
        else:
            self.bkg = ObjIter([Tree( get_local_alt(f_qcd), **treekwargs), Tree( get_local_alt(f_ttbar), **treekwargs)])

        # signal xsec is set to 0.010517 pb -> 31.05 fb * (0.58)^2 
        (self.signal).apply(lambda t : t.reweight(t.genWeight * t.xsecWeight / 1000))
        (self.bkg).apply(lambda t : t.reweight(t.genWeight * t.xsecWeight / 1000))

        f_pattern = '/eos/user/e/ekoenig/Ntuples/NanoHH4b/trg/{pairing}_2018_0L/data/jetht_tree.root'
        f_data = f_pattern.format(pairing=self.pairing)

        if self.no_data:
            self.data = ObjIter([])
        else:
            self.data = ObjIter([Tree( get_local_alt(f_data), **dict(treekwargs, normalization=None, color='black'))])

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
        self.signal = self.signal.apply(event_filter)
        self.bkg = self.bkg.apply(event_filter)
        self.data = self.data.apply(event_filter)

    @required
    def load_feynnet(self, signal, bkg, data):
        if self.model is None:
            return
        
        load_feynnet = fourb.nanohh4b.f_evaluate_feynnet(self.model)
        import multiprocess as mp
        nprocs = min(4, len(signal+bkg+data))
        with mp.Pool(nprocs) as pool:
            (signal+bkg+data).parallel_apply(load_feynnet, pool=pool, report=True)


    def plot_reco_eff(self, signal):
        signal.apply(fourb.nanohh4b.match_ak4_gen)

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
            varlist=['dHH_H1_mass','dHH_H2_mass'],
            legend=True,
            efficiency=True,
            saveas=f'{self.dout}/reco_eff_higgs_mass',
        )

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
        f_yields = study.make_path( os.path.join( self.dout, 'presel_4btag_yields.txt' ) )

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

        with open(f_yields, 'w') as f:
            table = tabulate.tabulate(table, headers=['sample','yield','an yield', 'this/an'], tablefmt='simple', numalign='right', floatfmt='.2f')
            print(table)
            f.write(table)

    def print_3btag_yields(self, signal, bkg, data):
        # NOTE: yields from AN2019_250_v6:Table 19
        an_yields = {
            'ggHH4b' : 2404.3 + 78.7,
            'qcd-mg' : 1058750.0 + 43059.3,
            'ttbar-powheg' : 111774.6 + 6518.1,
            'jetht': 2273811.0 + 107918.0,
        }
        f_yields = study.make_path( os.path.join( self.dout, 'presel_3btag_yields.txt' ) )

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

        with open(f_yields, 'w') as f:
            table = tabulate.tabulate(table, headers=['sample','yield','an yield', 'this/an'], tablefmt='simple', numalign='right', floatfmt='.2f')
            print(table)
            f.write(table)

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
                'ggHH4b' : 364.2 + 14.1,
                'qcd-mg' : 14384.3 + 582.8,
                'ttbar-powheg' : 2589.0 + 154.8,
                'jetht': 28150.0 + 1616.0,
            },
            c = {
                'ggHH4b' : 788.5 + 26.4,
                'qcd-mg' : 32113.6 + 1345.0,
                'ttbar-powheg' : 14435.2 + 642.3,
                'jetht': 75628.0 + 3538.0,
            },
            d = {
                'ggHH4b' : 434.1 + 12.9,
                'qcd-mg' : 87038.1 + 3393.5,
                'ttbar-powheg' : 26304.3 + 1157.0,
                'jetht': 208145.0 + 9144.0,
            }
        )

        f_yields = study.make_path( os.path.join( self.dout, 'abcd_yields.txt' ) )
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

        with open(f_yields, 'w') as f:
            for region in ['a','b','c','d']:
                table = (signal+bkg+data).apply(partial(get_yield, region=region)).list
                table = tabulate.tabulate(table, headers=['sample','yield','an yield', 'this/an'], tablefmt='simple', numalign='right', floatfmt='.2f')

                name = dict(
                    a='A_SR(4b)',
                    b='A_CR(4b)',
                    c='A_SR(3b)',
                    d='A_CR(3b)',
                ).get(region)
                print('Region:', name)
                print(table)
                print()
                f.write(f'Region: {name}\n' + table + '\n\n')

    def plot_3btag_datamc(self, data, bkg):
        study.quick(
            data+bkg,
            masks=lambda t : self.n_btag(t) == 3,
            varlist=['dHH_H1_mass','dHH_H2_mass'],
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

    @dependency(build_bkg_model, print_abcd_yields)
    def limits(self, signal, bkg_model):
        model = []
        study.quick(
            signal+bkg_model,
            masks=[self.bdt.a]*len(signal),
            varlist=['dHH_HH_mass'],
            plot_scale=[1000]*len(signal),
            limits=True,
            l_store=model,
            legend=True,
            ylim=(0, 7000),
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

    


if __name__ == '__main__':
    main()