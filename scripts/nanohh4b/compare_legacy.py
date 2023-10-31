from functools import partial
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
import tabulate


import re
from tqdm import tqdm
import timeit
import re

sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils import *
from utils.notebookUtils import Notebook, required, dependency

class Notebook(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.set_defaults(
            config='configs/nanohh4b/compare_legacy.yaml',
        )
        ...

    def init_out(self, signal):
        import uproot as ut

        signal = signal['ours']

        ttree = ut.lazy(f'{signal}:Events')
        ttree = Tree.from_ak(ttree, is_signal=True, sample='Our ggHH4b')

        scale = np.ones(len(ttree))
        for w in self.weights['ours']:
            scale = scale * ttree[w]

        print('Our SumW:', self.lumi * ak.sum(scale))
        ttree.extend(scale=scale)

        self.our_signal = ObjIter([ttree])

    def init_legacy(self, signal):
        import uproot as ut

        signal = signal['legacy']

        ttree = ut.lazy(f'{signal}:bbbbTree')
        ttree = Tree.from_ak(ttree, is_signal=True, sample='Legacy ggHH4b')

        import utils.fourbUtils.bbbbUtils as legacy
        legacy.map_to_nano(ttree)

        scale = np.ones(len(ttree))
        for w in self.weights['legacy']:
            scale = scale * ttree[w]

        eff_histo = ut.open(f'{signal}:eff_histo')
        eff_labels = eff_histo.axis().labels()
        sumw = eff_histo.counts()[eff_labels.index('Ntot_w')]
        scale = scale / sumw

        print('Legacy SumW:', self.lumi * ak.sum(scale))
        ttree.extend(scale=scale)

        self.legacy_signal = ObjIter([ttree])


    def legacy_selection(self, legacy_signal):
        lepton_veto = EventFilter('lepton_veto', filter=lambda t : (t.IsolatedMuon_Multiplicity == 0) & (t.IsolatedElectron_Multiplicity == 0), verbose=True)

        self.legacy_signal = legacy_signal.apply(lepton_veto)

        if not self.trigger: return

        trigger_matching = EventFilter('trigger_matching', filter=lambda t : t.HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5_ObjectMatched == 1, verbose=True)
        
        self.legacy_signal = self.legacy_signal.apply(trigger_matching)

    def our_selection(self, our_signal):

        if not self.trigger: return
        
        trigger = EventFilter("trigger", filter=lambda t : t.passTrig0L_TrgObjMatching == 1, verbose=True)
        self.our_signal = our_signal.apply(trigger)

    def print_sum2(self, our_signal, legacy_signal):
        print('Our SumW:', self.lumi * ak.sum(our_signal.scale))
        print('Legacy SumW:', self.lumi * ak.sum(legacy_signal.scale))
        print('Our/Legacy:', ak.sum(our_signal.scale) / ak.sum(legacy_signal.scale))



if __name__ == "__main__":
    Notebook.main()