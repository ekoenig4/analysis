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

varinfo.HX_m = dict(bins=(0,300,30))
varinfo.H1_m = dict(bins=(0,300,30))
varinfo.H2_m = dict(bins=(0,300,30))

varinfo.Y_m = dict(bins=(0,1500,30))
varinfo.X_m = dict(bins=(375,1500,31))

class FeynNet(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--dout', default='sixb_feynnet')
        parser.add_argument('--model', default=None)
        return parser
 
    @required
    def init(self):
        self.dout = f'{self.dout}'

        if not os.path.exists(self.dout):
            os.makedirs(self.dout)

        fc.sixb = fc.FileCollection('/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/')
        f_data = fc.fs.default.fullpath(fc.sixb.JetHT_Data_UL.get('ntuple.root'))

        self.data = ObjIter([Tree(f_data)])
        self.ar_bdt = sixb.bdt.get_ar_bdt()

    @required
    def load_feynnet(self, data):
        if self.model is None:
            return

        if 'feynnet' not in self.model:
            return

        self.dout = os.path.join(self.dout, 'feynnet')
        
        if self.model.endswith('/'): self.model=self.model[:-1]
        
        import utils.resources as rsc
        accelerator = 'cpu' if rsc.ngpus == 0 else 'cuda'

        import utils.sixbUtils as sixb
        load_feynnet = sixb.f_load_x3h_feynnet(self.model, accelerator=accelerator)
        data.apply(load_feynnet)

    @required
    def blind_data(self, data):
        blinded = EventFilter('blinded', filter=lambda t : ~self.ar_bdt.a(t))
        self.data = data.apply(blinded)

    def train_ar_bdt(self, data):
        self.ar_bdt.print_yields(data)
        self.ar_bdt.train(data)

    @dependency(train_ar_bdt)
    def build_bkg_model(self, data):
        bkg_model = EventFilter('bkg_model', filter=self.ar_bdt.b)
        self.bkg_model = data.asmodel('bkg model').apply(bkg_model)
        self.bkg_model.apply(lambda t : t.reweight(self.ar_bdt.reweight_tree))

    @dependency(build_bkg_model)
    def save_bkg_model(self, bkg_model):
        bkg_model.write('bkg_model_{base}')
    


    





    


if __name__ == '__main__':
    FeynNet.main()
