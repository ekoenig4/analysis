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

varinfo.HX_m = dict(bins=(0,300,30))
varinfo.H1_m = dict(bins=(0,300,30))
varinfo.H2_m = dict(bins=(0,300,30))

varinfo.Y_m = dict(bins=(0,1500,30))
varinfo.X_m = dict(bins=(375,1500,31))

class FeynNet(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--dout', default='sixb_feynnet')
        # parser.add_argument('--model', default='bkg0.25_hm10')
        parser.add_argument('--model', default='ranker')
        parser.add_argument('--serial', action='store_true')
        return parser
 
    @required
    def init(self):
        self.dout = f'{self.dout}/{self.model}'

        if not os.path.exists(self.dout):
            os.makedirs(self.dout)

        # self.modelpath = '/eos/uscms/store/user/ekoenig/weaver/analysis/models/exp_sixb_diff_aggr/feynnet_ranker_6b/20230728_59b53a4bde5e7da6eb8e6aa522b30859_ranger_lr0.0047_batch2000_withbkg'
        self.modelpath= '/eos/uscms/store/user/ekoenig/weaver/models/exp_sixb_official/feynnet_ranker_6b/20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg/'

        # self.modelpath= '/eos/uscms/store/user/srosenzw/weaver/models/exp_sixb_official/feynnet_ranker_6b/20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg/'

        # self.modelpath= '/eos/uscms/store/user/ekoenig/weaver/models/exp_sixb_megamind/feynnet_ranker_6b/20230803_495f81bce0c466c3345918572c4e0906_ranger_lr0.0047_batch2000_withbkg'

        fc.sixb = fc.FileCollection('/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/')
        f_data = fc.sixb.JetHT_Data_UL.get('ntuple.root')

        self.trees = ObjIter([Tree(f_data)])
        self.ar_bdt = sixb.bdt.get_ar_bdt()

    @required
    def load_feynnet(self, trees):
        print(self.modelpath)
        load_feynnet = sixb.f_load_feynnet_assignment( self.modelpath, onnx=False, order='random' )

        if not self.serial:
            import multiprocess as mp
            with mp.Pool(4) as pool:
                trees.parallel_apply(load_feynnet, report=True, pool=pool)
        else:
            trees.apply(load_feynnet, report=True)

        trees.apply(sixb.assign)

    def data_ratio(self, trees):
        n_lowscore = trees.apply(lambda t : np.sum( sixb.bdt.btag6bavg(t) < sixb.bdt.btag_cfg )).npy.sum()
        n_highscore = trees.apply(lambda t : np.sum( sixb.bdt.btag6bavg(t) >= sixb.bdt.btag_cfg )).npy.sum()

        print('Data high score:    ', n_highscore)
        print('Data low score:    ', n_lowscore)
        print('Data low/high: ', n_lowscore/n_highscore)

        
        n_out_of_asr = trees.apply(lambda t : np.sum( sixb.bdt.h_dm(t) >  sixb.bdt.hm_cfg['SRedge'] )).npy.sum()
        print('Data out of ASR:    ', n_out_of_asr)

    def blind_data(self, trees):
        blinded = EventFilter('blinded', filter=lambda t : ~self.ar_bdt.a(t))
        self.trees = trees.apply(blinded).asmodel()

    def train_bdt(self, trees):
        self.ar_bdt.print_yields(trees)
        self.ar_bdt.train(trees)
        self.ar_bdt.print_results(trees)

    def plot_bdt(self, trees):
        study.quick(
            trees,
            masks=self.ar_bdt.b,
            scale=self.ar_bdt.reweight_tree,
            varlist=['X_m','Y_m',None,'HX_m','H1_m','H2_m'],
            legend=True,
            saveas=f'{self.dout}/bdt_model',
        )

    def save_result(self, trees):

        histos = obj_store()
        study.quick(
            trees,
            masks=self.ar_bdt.b,
            scale=self.ar_bdt.reweight_tree,
            varlist=['X_m','Y_m', sixb.bdt.btag6bavg,'HX_m','H1_m','H2_m'],
            store=histos,
        )

        import pickle as pkl

        bkg_model_histos = {
            key : histo[0] if not isinstance(histo[0], Stack) else histo[0][0]
            for key, histo in zip(('X_m','Y_m','btag6bavg','HX_m','H1_m','H2_m'), histos)
        }

        bkg_model_X_m = bkg_model_histos['X_m']
         
        # dump the bins for the bkg model
        print(f'data = {repr(bkg_model_X_m.histo)}')

        with open(f'{self.dout}/bkg_model_histos.pkl', 'wb') as f:
            pkl.dump(bkg_model_histos, f)


    


if __name__ == '__main__':
    main()
