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
def n_feynnet_loose_btag(t):
    return ak.sum(t.j_btag>jet_btagWP[1],axis=1)

@tree_variable(xlabel='N FeynNet Medium BTag')
def n_feynnet_medium_btag(t):
    return ak.sum(t.j_btag>jet_btagWP[2],axis=1)
def n_all_medium_btag(t):
    return ak.sum(t.jet_btag>jet_btagWP[2],axis=1)
def n_feynnet_tight_btag(t):
    return ak.sum(t.j_btag>jet_btagWP[3],axis=1)


def n_gen_medium_btag(t):
    btag = t.jet_btag
    signalId = t.jet_signalId
    return ak.sum(btag[signalId>-1]>jet_btagWP[2],axis=1)

class FeynNet(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--dout', default='fourb_feynnet')
        parser.add_argument('--model', default='bkg0.25_hm10')
        return parser

    @required
    def init(self):
        self.dout = f'{self.dout}/{self.model}'

        modelpath =  f"/eos/uscms/store/user/ekoenig/weaver/models/exp_fourb/feynnet_4b/20230524_*_ranger_lr0.0047_batch1024_{self.model}_withbkg"
        print(modelpath)
        self.modelpath = fc.glob(modelpath)[0]

        f_signal = ['/store/user/ekoenig/4BAnalysis/NTuples/feynnet/GluGluToHHTo4B/test_ntuple.root']
        f_qcd = ['/store/user/ekoenig/4BAnalysis/NTuples/feynnet/QCD*bEn*/test_ntuple.root','/store/user/ekoenig/4BAnalysis/NTuples/feynnet/QCD*BGen*/test_ntuple.root']
        f_tt = ['/store/user/ekoenig/4BAnalysis/NTuples/feynnet/TT*/test_ntuple.root']

        # %%
        self.signal = ObjIter([Tree(f_signal)])
        self.bkg = ObjIter([Tree(f_qcd), Tree(f_tt)])

    def plot_n_all_jets(self, signal, bkg):
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
        t4btag = CollectionFilter('jet', filter=lambda t : ak_rank(-t.jet_btag) < 6)
        signal = signal.apply(t4btag, report=True)
        bkg = bkg.apply(t4btag, report=True)
        signal.apply(lambda t : t.extend(nfound_select=ak.sum(t.jet_signalId>-1,axis=1)))

        self.signal = signal
        self.bkg = bkg

    def plot_nfound_select(self, signal):
        study.quick(
            signal,
            varlist=['nfound_select'],
            xlabels=['N GEN Higgs Jets in Selection'],
            efficiency=True,
            saveas=f'{self.dout}/nfound_select',
        )

    @required
    def load_feynnet(self, signal, bkg):
        load_feynnet = fourb.f_load_feynnet_assignment( self.modelpath )

        (signal+bkg).apply(load_feynnet, report=True)
        (signal+bkg).apply(fourb.assign)

    def plot_higgs_m(self, signal, bkg):
        varinfo.H1_m = dict(bins=(0,500,30))
        varinfo.H2_m = dict(bins=(0,500,30))

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
            size=(4,5),
            scatter=True,
            legend=True,
            saveas=f'{self.dout}/higgs_m_2d',
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
            masks=lambda t : n_feynnet_medium_btag(t)==4,
            varlist=['H1_m','H2_m'],
            efficiency=True,
            suptitle='N FeynNet Medium BTag = 4',
            legend=True,
            saveas=f'{self.dout}/higgs_m_medium4',
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

if __name__ == '__main__':
    main()
