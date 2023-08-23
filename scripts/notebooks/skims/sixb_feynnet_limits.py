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

re_mxmy = re.compile(r'.*MX-(\d+)_MY-(\d+).*')
def get_mxmy(fname):
    m = re_mxmy.match(fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

class FeynNet(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--dout', default='sixb_feynnet')
        # parser.add_argument('--model', default='bkg0.25_hm10')
        parser.add_argument('--model', default='ranker')
        parser.add_argument('--serial', action='store_true')
        parser.add_argument('--input', nargs='+', type=int)
        return parser
 
    @required
    def init(self):
        self.dout = f'{self.dout}/{self.model}'

        if not os.path.exists(f'{self.dout}/limits/'):
            os.makedirs(f'{self.dout}/limits/')

        # self.modelpath = '/eos/uscms/store/user/ekoenig/weaver/analysis/models/exp_sixb_diff_aggr/feynnet_ranker_6b/20230728_59b53a4bde5e7da6eb8e6aa522b30859_ranger_lr0.0047_batch2000_withbkg'
        self.modelpath= '/eos/uscms/store/user/ekoenig/weaver/models/exp_sixb_official/feynnet_ranker_6b/20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg/'
        # self.modelpath= '/eos/uscms/store/user/ekoenig/weaver/models/exp_sixb_megamind/feynnet_ranker_6b/20230803_495f81bce0c466c3345918572c4e0906_ranger_lr0.0047_batch2000_withbkg'

        fc.sixb = fc.FileCollection('/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/')
        f_signal = np.array([ f'{d}/ntuple.root' for d in fc.sixb.Official_NMSSM.ls if d.endswith('pythia8') ])

        # 210 GeV mass points
        masses = np.array(list(map(get_mxmy, f_signal)))

        # sort by MX, then MY
        order = np.lexsort((masses[:,1],masses[:,0]))
        f_signal = f_signal[order]
        masses = masses[order]

        mask = np.ones(len(f_signal), dtype=bool)
        # remove all 50 GeV mass points
        # mask &= np.all(( (masses / 100) - (masses // 100) ) == 0, axis=1)
        # remove all points above 1400 GeV
        mask &= masses[:,0] > 1400
        f_signal = f_signal[mask]

        print('Total number of signal files:', len(f_signal))
        print(f'Running on {len(self.input)} signal file(s)')

        self.signal = ObjIter([Tree([f_signal[idx]], report=False) for idx in tqdm(self.input)])
        
        self.ar_bdt = sixb.bdt.get_ar_bdt()
        import pickle as pkl
        with open(f'{self.dout}/bkg_model_histos.pkl', 'rb') as f:
            self.bkg_model_X_m = pkl.load(f)['X_m']


    @required
    def load_feynnet(self, signal):
        print(self.modelpath)
        load_feynnet = sixb.f_load_feynnet_assignment( self.modelpath, onnx=True, order='random' )

        if len(signal) > 1:
            import multiprocess as mp
            with mp.Pool(4) as pool:
                signal.parallel_apply(load_feynnet, report=True, pool=pool)
        else:
            signal.apply(load_feynnet, report=True)
        signal.apply(sixb.assign)

    
    def get_limits(self, signal):
        histos = obj_store()

        study.quick(
            signal,
            masks=self.ar_bdt.a,
            varlist=['X_m'],
            store=histos
        )

        self.models = ObjIter([
            Model(h_sig, self.bkg_model_X_m)
            for h_sig in histos[0]
        ])

        get_upperlimits = f_upperlimit(poi=np.linspace(0.1,5,50))

        if len(self.models) > 1:
            import multiprocess as mp
            with mp.Pool(4) as pool:
                self.models.parallel_apply(get_upperlimits, report=True, pool=pool)
        else:
            self.models.apply(get_upperlimits, report=True)

    def save_limits(self, models):
        def write(model):
            newtree = dict(
                mx = [model.mx],
                my = [model.my],
                exp_limits = [0.3 * np.array(model.h_sig.stats.exp_limits) ],
            )
            
            with ut.recreate(f'{self.dout}/limits/MX_{model.mx}_MY_{model.my}.root') as f:
                f['tree'] = {
                    k : [v] for k, v in newtree.items()
                }

        models.apply(write, report=True)




    


if __name__ == '__main__':
    main()
