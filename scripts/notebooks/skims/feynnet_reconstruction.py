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

# %%
def main():
    notebook = Notebook.from_parser()
    notebook.hello()
    notebook.run()

class Notebook(RunSkim):
    
    @staticmethod
    def add_parser(parser):
        parser.add_argument("--model", default='feynnet_bkg_33sig', help="model to use for loading feynnet")
        return parser
    
    @required
    def init_model(self, model):
        self.outfile = f'{model}_{{base}}'
        self.model = eightb.models.get_model(model)
    
    @required
    def load_feynnet(self, signal, bkg, data):
        load = lambda t : eightb.load_feynnet_assignment(t, self.model.analysis)
        (signal+bkg+data).apply(load, report=True)

    @required
    def assign_resonances(self, signal, bkg, data):
        def assign(tree):
            j = get_collection(tree, 'j', named=False)
            h = get_collection(tree, 'h', named=False)
            y = get_collection(tree, 'y', named=False)
            x = get_collection(tree, 'x', named=False)

            tree.extend(
                **{
                    f'{J}_{field}': j[field][:,i]
                    for field in j.fields
                    for i, J in enumerate(eightb.quarklist)
                },
                **{
                    f'{H}_{field}': h[field][:,i]
                    for field in h.fields
                    for i, H in enumerate(eightb.higgslist)
                },
                **{
                    f'{Y}_{field}': y[field][:,i]
                    for field in y.fields
                    for i, Y in enumerate(eightb.ylist)
                },
                **{
                    f'X_{field}': x[field]
                    for field in x.fields
                }
            )
        (signal+bkg+data).apply(assign, report=True)

    @required
    def write(self, signal, bkg, data):
        (signal+bkg+data).apply(
                lambda t : t.write(self.outfile), report=True
            )


if __name__ == '__main__':
    main()