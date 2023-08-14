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
    notebook = FeynNet.from_parser()
    notebook.hello()
    notebook.run()


@tree_variable(xlabel='N GEN-B FeynNet')
def nfound_feynnet(t):
    return ak.sum(t.j_signalId>-1,axis=1)

@tree_variable(xlabel='N FeynNet Paired Higgs')
def nfound_feynnet_h(t):
    return ak.sum(t.h_signalId>-1,axis=1)

# %%
@tree_variable(xlabel='N FeynNet Loose BTag')
def n_feynnet_loose_btag(t):
    return ak.sum(t.j_btag>jet_btagWP[1],axis=1)

@tree_variable(xlabel='N FeynNet Medium BTag')
def n_feynnet_medium_btag(t):
    return ak.sum(t.j_btag>jet_btagWP[2],axis=1)
def n_all_medium_btag(t):
    return ak.sum(t.jet_btag>jet_btagWP[2],axis=1)
@tree_variable(xlabel='N FeynNet Tight BTag')
def n_feynnet_tight_btag(t):
    return ak.sum(t.j_btag>jet_btagWP[3],axis=1)


def n_gen_medium_btag(t):
    btag = t.jet_btag
    signalId = t.jet_signalId
    return ak.sum(btag[signalId>-1]>jet_btagWP[2],axis=1)

varinfo.H1_m = dict(bins=(0,500,30))
varinfo.H2_m = dict(bins=(0,500,30))
class FeynNet(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--dout', default='fourb_feynnet')
        # parser.add_argument('--model', default='bkg0.25_hm10')
        parser.add_argument('--model', default='ranker')
        parser.add_argument('--serial', action='store_true')
        parser.add_argument('--njet', default=6, type=int)
        return parser
 
    @required
    def init(self):
        self.dout = f'{self.dout}/{self.model}-{self.njet}jet'

        modelpath =  f"/eos/uscms/store/user/ekoenig/weaver/models/exp_fourb/feynnet_4b/20230524_*_ranger_lr0.0047_batch1024_{self.model}_withbkg"
        # self.modelpath = f"/eos/uscms/store/user/ekoenig/weaver/models/exp_fourb/feynnet_ranker_4b/20230713_8ec79fb06e8ec977414bd16de9bc7323_ranger_lr0.0047_batch500"
        # self.modelpath = f"/eos/uscms/store/user/ekoenig/weaver/models/exp_fourb/feynnet_ranker_4b/20230713_f25b895e24bda8af40934a0c7d86fd06_ranger_lr0.0047_batch500"
        # self.modelpath = f"/eos/uscms/store/user/ekoenig/weaver/models/exp_bkg_ranker/feynnet_ranker_4b/20230714_f9c66830463a1206e87162025651fb59_ranger_lr0.0047_batch500"
        # self.modelpath = f"/eos/uscms/store/user/ekoenig/weaver/models/exp_fourb/feynnet_ranker_4b/20230719_88aaa1ecd77546f56d1b64cc3376c992_ranger_lr0.0047_batch2000"
        # self.modelpath = "/eos/uscms/store/user/ekoenig/weaver/models/exp_fourb/feynnet_ranker_4b/20230720_ed3f925338d4336f79e225dc1927ab01_ranger_lr0.0047_batch2000"

        # self.modelpath = {
        #     6 : '/eos/uscms/store/user/ekoenig/weaver/models/exp_fourb_njet/feynnet_ranker_4b/20230721_b1411d182851339f5a3d27dcf4f4e949_ranger_lr0.0047_batch2000',
        #     4 : '/eos/uscms/store/user/ekoenig/weaver/models/exp_fourb_njet/feynnet_ranker_4b/20230721_1b987f5c4ddc36b11b2b48bb2b0159b7_ranger_lr0.0047_batch2000'
        # }.get(self.njet)

        self.modelpath = {
            4 : '/eos/uscms/store/user/ekoenig/weaver/models/exp_fourb_sig_weighted_4jet/feynnet_ranker_4b/20230725_009c01444a2d07d1a420b37a2aa26d6c_ranger_lr0.0047_batch2000',
            6 : '/eos/uscms/store/user/ekoenig/weaver/models/exp_fourb_sig_weighted_6jet/feynnet_ranker_4b/20230725_2c6fa70f10b322fdf1b39a9b08a0bea9_ranger_lr0.0047_batch2000'
        }.get(self.njet)

        basepath = "/store/user/ekoenig/MultiHiggs/DiHiggs/RunII/FeynNetTraining_27June2023/ForFeynNet_UL18_SignalPlusBackground_27June2023/"
        f_signal = [f'{basepath}/GluGlu*/test_ntuple.root']
        f_qcd = [f'{basepath}/QCD*bEn*/test_ntuple.root',f'{basepath}/QCD*BGen*/test_ntuple.root']
        f_tt = [f'{basepath}/TT*/test_ntuple.root']

        # %%
        self.signal = ObjIter([Tree(f_signal)])
        self.bkg = ObjIter([Tree(f_qcd), Tree(f_tt)])

    def _plot_n_all_jets(self, signal, bkg):
        (signal+bkg).apply(lambda t : t.extend(n_all_jet=t.n_jet))

        study.quick(
            signal+bkg,
            legend=True,
            varlist=['n_all_jet'],
            xlabels=['Number of Jets'],
            binlist=[np.arange(15)],
            # lumi=None,
            efficiency=True,
            saveas=f'{self.dout}/n_all_jet',
        )

    @required
    def select_jets(self, signal, bkg):
        t4btag = CollectionFilter('jet', filter=lambda t : ak_rank(-t.jet_btag) < self.njet)
        signal = signal.apply(t4btag, report=True)
        bkg = bkg.apply(t4btag, report=True)
        signal.apply(lambda t : t.extend(nfound_select=ak.sum(t.jet_signalId>-1,axis=1)))

        self.signal = signal
        self.bkg = bkg

    def _plot_nfound_select(self, signal):
        study.quick(
            signal,
            varlist=['nfound_select'],
            xlabels=['N GEN Higgs Jets in Selection'],
            efficiency=True,
            saveas=f'{self.dout}/nfound_select',
        )

    @required
    def load_feynnet(self, signal, bkg):
        print(self.modelpath)
        load_feynnet = fourb.f_load_feynnet_assignment( self.modelpath )

        if not self.serial:
            import multiprocessing as mp
            with mp.Pool(8) as pool:
                (signal+bkg).parallel_apply(load_feynnet, report=True, pool=pool)
        else:
            (signal+bkg).apply(load_feynnet, report=True)

        (signal+bkg).apply(fourb.assign)

    def plot_higgs_m(self, signal, bkg):

        study.quick(
            signal+bkg,
            varlist=['H1_m'],
            efficiency=True,
            legend=True,
            saveas=f'{self.dout}/h1_m',
        )
        
        study.quick(
            signal+bkg,
            varlist=['H2_m'],
            efficiency=True,
            legend=True,
            saveas=f'{self.dout}/h2_m',
        )

        # %%
        study.quick2d(
            signal+bkg,
            varlist=['H1_m','H2_m'],
            binlist=[(0,250,30),(0,250,30)],
            size=(4,5),
            # scatter=True,
            legend=True,
            saveas=f'{self.dout}/higgs_m_2d',
        )

    def plot_higgs_sr(self, signal, bkg):
        study.quick(
            signal+bkg,
            masks=lambda t : abs(t.H1_m-125)<30,
            varlist=['H1_m','H2_m'],
            binlist=[(0,250,30),(0,250,30)],
            size=(4,5),
            # scatter=True,
            efficiency=True,
            legend=True,
            suptitle='|H1_m - 125| < 30 GeV',
            saveas=f'{self.dout}/higgs_m_sr',
        )

    def plot_nfound_feynnet(self, signal, bkg):
        study.quick(
            signal,
            varlist=[nfound_feynnet_h],
            efficiency=True,
            h_label_stat=lambda h: f'Correct = {h.histo[-1]:0.2%}',
            legend=True,
            saveas=f'{self.dout}/nfound_feynnet_h',
        )

    def plot_feynnet_btag(self, signal, bkg):
        study.quick(
            signal+bkg,
            varlist=[n_feynnet_loose_btag, n_feynnet_medium_btag, n_feynnet_tight_btag],
            efficiency=True,
            legend=True,
            saveas=f'{self.dout}/feynnet_btag',
        )

    def plot_feynnet_btag_eff(self, signal):
        study.quick(
            signal,
            masks=lambda t : n_feynnet_loose_btag(t)==4,
            varlist=[nfound_feynnet_h],
            efficiency=True,
            h_label_stat=lambda h: f'Correct = {h.histo[-1]:0.2%}',
            suptitle='N FeynNet Loose BTag = 4',
            legend=True,
            saveas=f'{self.dout}/nfound_feynnet_h_loose4',
        )

        study.quick(
            signal,
            masks=lambda t : n_feynnet_medium_btag(t)==4,
            varlist=[nfound_feynnet_h],
            efficiency=True,
            h_label_stat=lambda h: f'Correct = {h.histo[-1]:0.2%}',
            suptitle='N FeynNet Medium BTag = 4',
            legend=True,
            saveas=f'{self.dout}/nfound_feynnet_h_medium4',
        )

        study.quick(
            signal,
            masks=lambda t : n_feynnet_tight_btag(t)==4,
            varlist=[nfound_feynnet_h],
            efficiency=True,
            h_label_stat=lambda h: f'Correct = {h.histo[-1]:0.2%}',
            suptitle='N FeynNet Tight BTag = 4',
            legend=True,
            saveas=f'{self.dout}/nfound_feynnet_h_tight4',
        )


    def plot_feynnet_btag_hm(self, signal, bkg):
        study.quick(
            signal+bkg,
            masks=lambda t : n_feynnet_loose_btag(t)==4,
            varlist=['H1_m','H2_m'],
            efficiency=True,
            suptitle='N FeynNet Loose BTag = 4',
            legend=True,
            saveas=f'{self.dout}/higgs_m_loose4',
        )
        study.quick(
            signal+bkg,
            masks=lambda t : n_feynnet_loose_btag(t)==4,
            varlist=['H1_m','H2_m'],
            efficiency=True,
            suptitle='N FeynNet Loose BTag = 4',
            legend=True, log=True,
            saveas=f'{self.dout}/higgs_m_loose4_log',
        )

        study.quick(
            signal+bkg,
            masks=lambda t : n_feynnet_medium_btag(t)==4,
            varlist=['H1_m','H2_m'],
            efficiency=True,
            suptitle='N FeynNet Medium BTag = 4',
            legend=True,
            saveas=f'{self.dout}/higgs_m_medium4',
        )

        study.quick(
            signal+bkg,
            masks=lambda t : n_feynnet_medium_btag(t)==4,
            varlist=['H1_m','H2_m'],
            efficiency=True,
            suptitle='N FeynNet Medium BTag = 4',
            legend=True, log=True,
            saveas=f'{self.dout}/higgs_m_medium4_log',
        )

        study.quick(
            signal+bkg,
            masks=lambda t : n_feynnet_tight_btag(t)==4,
            varlist=['H1_m','H2_m'],
            efficiency=True,
            suptitle='N FeynNet Tight BTag = 4',
            legend=True,
            saveas=f'{self.dout}/higgs_m_tight4',
        )

        study.quick(
            signal+bkg,
            masks=lambda t : n_feynnet_tight_btag(t)==4,
            varlist=['H1_m','H2_m'],
            efficiency=True,
            suptitle='N FeynNet Tight BTag = 4',
            legend=True, log=True,
            saveas=f'{self.dout}/higgs_m_tight4_log',
        )

if __name__ == '__main__':
    main()
