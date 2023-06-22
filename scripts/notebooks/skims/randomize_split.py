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
from utils.notebookUtils.driver.run_skim import RunSkim

import json

# %%
def main():
    notebook = Notebook.from_parser()
    notebook.hello()
    notebook.run()

class Notebook(RunSkim):
    @staticmethod
    def add_parser(parser):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--nsplit", type=int)
        group.add_argument("--frac", type=float, nargs='+')


        return parser
    
    def init(self):
        if self.nsplit:
            self.frac = [1/self.nsplit]*self.nsplit

        self.nsplit = len(self.frac)
        self.frac = np.array(self.frac)/np.sum(self.frac)
        self.bins = np.cumsum(np.insert(self.frac,0,0))

        print(f'nsplit: {self.nsplit}')
        print(f'frac: {self.frac}')

    def randomize_split(self, trees):
        (trees).apply(
            lambda t : t.extend(
                _random_split= ak.from_numpy( np.digitize(np.random.uniform(size=len(t)), self.bins)-1 )
            )
        )

    def write_split(self, trees):

        for i in range(self.nsplit):
            split = trees.copy()
            split = split.apply(EventFilter(f'split_{i}', filter=lambda t : t._random_split==i))

            def rescale_cutflow(t):
                frac = ak.mean(t._random_split==i)
                t.cutflow = [ Histo(frac*cutflow.histo, cutflow.bins, frac*cutflow.error) for cutflow in t.cutflow ]
            split.apply(rescale_cutflow)

            study.quick( 
                split,
                varlist=['jet_pt[:,0]']
            )
            split.write(f'split_{i}_{{base}}')


if __name__ == '__main__':
    main()