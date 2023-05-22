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
    notebook.print_namespace()
    print(notebook)
    notebook.run()

class Notebook(RunSkim):
    
    @staticmethod
    def add_parser(parser):
        parser.set_defaults(
            module='fc.eightb.feynnet'
        )


        return parser
    
    def init_feynman(self):
        from utils.FeynNet.Feynman import Feynman

        diagram = Feynman('x').decays(
            Feynman('y').decays(
                Feynman('h').decays('b','b'),
                Feynman('h').decays('b','b')
            ),
            Feynman('y').decays(
                Feynman('h').decays('b','b'),
                Feynman('h').decays('b','b')
            )
        ).build_diagram()

        internalstates = diagram.get_internalstate_types()

        jet_perms = diagram.get_finalstate_permutations(b=8)['b']
        self.h_b_perms = internalstates['h'][0].get_product_permutations(b=8)['b']
        self.y_h_perms = internalstates['y'][0].get_product_permutations(b=8)['h']
        self.x_y_perms = internalstates['x'][0].get_product_permutations(b=8)['y']

    def _reco_perm(self, jet_p4):
        b_perms = jet_p4[:, self.h_b_perms]
        b_hid = b_perms.signalId//2

        h_p4 = b_perms[:,:,0] + b_perms[:,:,1]
        h_id = ak.where( b_hid[:,:,0]==b_hid[:,:,1], b_hid[:,:,0], -1)

        h_perms = h_p4[:, self.y_h_perms]
        h_yid = h_id[:, self.y_h_perms]//2

        y_p4 = h_perms[:,:,0] + h_perms[:,:,1]
        y_id = ak.where( h_yid[:,:,0]==h_yid[:,:,1], h_yid[:,:,0], -1)

        y_perms = y_p4[:, self.x_y_perms]
        y_xid = y_id[:, self.x_y_perms]//2

        x_p4 = y_perms[:,:,0] + y_perms[:,:,1]
        # x_id = ak.where( y_xid[:,:,0] == y_xid[:,:,1], y_xid[:,:,0], -1)

        x_4h_m = ak.flatten(h_perms.m[:, self.x_y_perms], axis=3)
        x_2y_m = y_perms.m
        # x_m = x_p4.m

        chi2 = ( (x_2y_m[:,:,0] - x_2y_m[:,:,1])/(x_2y_m[:,:,0] + x_2y_m[:,:,1]) )**2 + ak.sum( ( (x_4h_m - 125)/25 )**2, axis=2)
        return chi2

if __name__ == '__main__':
    main()