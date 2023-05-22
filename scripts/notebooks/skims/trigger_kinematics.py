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
    
    def trigger_kinematics(self, signal, bkg, data):
        def _trigger_kinematics(t):

            jet_pt = t.jet_pt 
            pfht_pt30=ak.sum( jet_pt[jet_pt>30], axis=1 )

            pfht330_pt30 = pfht_pt30 > 330
            quadpf_jet = (t.jet_pt[:,0] > 75) & (t.jet_pt[:,1] > 60) & (t.jet_pt[:,2] > 45) & (t.jet_pt[:,3] > 40)

            pfht330_pt30_quadpf_jet = pfht330_pt30 & quadpf_jet

            t.extend(
                trigger_kinematics=pfht330_pt30_quadpf_jet
            )
        (signal+bkg+data).apply(_trigger_kinematics, report=True)

    def apply_trigger_kinematics(self, signal, bkg, data):
        self.signal = signal.apply(EventFilter('trigger_kinematics', filter=lambda t : t.trigger_kinematics))
        self.bkg = bkg.apply(EventFilter('trigger_kinematics', filter=lambda t : t.trigger_kinematics))
        self.data = data.apply(EventFilter('trigger_kinematics', filter=lambda t : t.trigger_kinematics))

    def write(self, signal, bkg, data):
        (signal + bkg + data).write(
            'trgkin_{base}',
        )
    

if __name__ == '__main__':
    main()