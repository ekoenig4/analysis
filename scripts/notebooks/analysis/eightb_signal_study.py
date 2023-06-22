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

from utils.notebookUtils.driver.run_analysis import RunAnalysis
from utils.notebookUtils import required, dependency, optional


class Notebook(RunAnalysis):
    @staticmethod
    def add_parser(parser):
        parser.set_defaults(
            module='fc.eightb.preselection.t8btag_minmass',
            use_signal='feynnet_plus_list',
            no_bkg=True, no_data=True,
        )

    @required
    def init(self, signal):
        self.dout = 'signal_study'
        self.use_signal = [ i for i, mass in enumerate(signal.apply(lambda t : t.mass)) if mass in ( '(700, 300)', '(1000, 450)', '(1200, 500)' ) ]

    @required
    def trigger_kinematics(self, signal):
        pt_filter = eightb.selected_jet_pt('trigger')

        def pfht(t):
            return ak.sum(t.jet_pt[(t.jet_pt > 30)], axis=-1)
        pfht_filter = EventFilter('pfht330', filter=lambda t : pfht(t) > 330)

        event_filter = FilterSequence(pfht_filter, pt_filter)
        self.signal = signal.apply(event_filter)

    def fully_resolved_efficiency(self, signal):
        fig, ax = study.get_figax(size=(10,8))

        study.mxmy_phase(
            signal,
            label=signal.mass.list,
            zlabel='Fraction Fully Resolved After Selection',
            efficiency=True,

            f_var=lambda t: ak.mean(t.nfound_select==8),
            g_cmap='jet',

            xlabel='$M_X$ (GeV)',
            ylabel='$M_Y$ (GeV)',

            # xlim=(550,1250),
            # ylim=(200,650),
            zlim=np.linspace(0,1,11),

            figax=(fig,ax),
            saveas=f'{self.dout}/fully_resolved_efficiency.png'
        )

if __name__ == '__main__': 
    Notebook.main()