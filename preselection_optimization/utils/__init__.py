import uproot as ut
import awkward as ak
import numpy as np
import vector
from uproot3_methods import TLorentzVectorArray
from tqdm import tqdm


from .selectUtils import *
from .plotUtils import *
from .studyUtils import *
from .cutConfig import *


class Branches:
    def __init__(self,ttree):
        self.ttree = ttree
        self.nevents = ak.size( self["Run"] )

        self.sixb_found_mask = self["nfound_all"] == 6
        self.nsignal = ak.sum( self.sixb_found_mask )
        self.sixb_jet_mask = get_jet_index_mask(self,self["signal_bjet_index"])
    def __getitem__(self,key): return self.ttree[key].array()
