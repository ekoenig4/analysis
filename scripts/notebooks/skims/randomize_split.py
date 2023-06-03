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
        parser.add_argument("--nsplit", default=5, type=int)
        return parser

    def randomize_split(self, trees):
        (trees).apply(
            lambda t : t.extend(
                _random_split= ak.from_numpy( np.random.randint(self.nsplit, size=len(t)) )
            )
        )

    def write_split(self, trees):
        for i in range(self.nsplit):
            split = trees.copy()
            split = split.apply(EventFilter(f'split_{i}', filter=lambda t : t._random_split==i))

            study.quick( 
                split,
                varlist=['jet_pt[:,0]']
            )
            split.write(f'split_{i}_{{base}}')


if __name__ == '__main__':
    main()