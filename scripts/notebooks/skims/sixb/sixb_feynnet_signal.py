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
        parser.add_argument('files', nargs='*')
        return parser
 
    @required
    def init(self):
        self.dout = f'{self.dout}'

        if not os.path.exists(self.dout):
            os.makedirs(self.dout)

        self.trees = ObjIter([Tree(f) for f in self.files])
        self.ar_bdt = sixb.bdt.get_ar_bdt()

    @required
    def load_feynnet(self, trees):
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
        trees.apply(load_feynnet)

    def get_sr_region(self, trees):
        asr = EventFilter('asr', filter=self.ar_bdt.a)
        self.trees = self.trees.apply(asr)

    def save(self, trees):
        trees.write('asr_{base}')
    


    





    


if __name__ == '__main__':
    FeynNet.main()
