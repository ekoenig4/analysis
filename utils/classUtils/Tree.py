from ..selectUtils import *
from ..xsecUtils import *
from ..utils import *

import uproot as ut
import awkward as ak
import numpy as np


class SixBFile:
    def __init__(self, fname):
        self.fname = fname
        self.tfile = ut.open(fname)

        cutflow = self.tfile.get('h_cutflow')
        self.cutflow_labels = cutflow.axis().labels()
        self.cutflow = cutflow.to_numpy()[0]
        self.total_events = self.cutflow[0]

        self.sample, self.xsec = next(
            ((key, value) for key, value in xsecMap.items() if key in fname), ("unk", 1))
        self.scale = self.xsec / \
            self.total_events if type(self.xsec) == float else 1

        self.ttree = self.tfile.get('sixBtree')
        self.raw_events = self.ttree.num_entries


def init_sample(self):  # Helper Method For Tree Class
    self.is_data = any("Data" in fn.fname for fn in self.filelist)
    self.is_signal = all("NMSSM" in fn.fname for fn in self.filelist)
    sample_tag = [next((tag for key, tag in tagMap.items(
    ) if key in fn.sample), None) for fn in self.filelist]

    if (sample_tag.count(sample_tag[0]) == len(sample_tag)):
        self.sample = sample_tag[0]
    else:
        self.sample = "Bkg"

    if self.is_data:
        self.sample = "Data"
    self.color = colorMap.get(self.sample, None)


def init_tree(self):
    self.ttree = ut.lazy([fn.ttree for fn in self.filelist])
    self.fields = self.ttree.fields
    self.extend(
        sample_id=ak.concatenate([ak.Array([i]*fn.raw_events)
                                 for i, fn in enumerate(self.filelist)]),
        scale=ak.concatenate([np.full(
            shape=fn.raw_events, fill_value=fn.scale, dtype=np.float) for fn in self.filelist])
    )

    self.raw_events = sum(fn.raw_events for fn in self.filelist)
    self.cutflow_labels = max(
        (fn.cutflow_labels for fn in self.filelist if fn.cutflow_labels), key=lambda l: len(l))
    ncutflow = len(self.cutflow_labels)
    self.cutflow = [ak.fill_none(ak.pad_none(
        fn.cutflow, ncutflow, axis=0, clip=True), 0).to_numpy() for fn in self.filelist]


def init_selection(self):
    self.all_events_mask = ak.Array([True]*self.raw_events)
    njet = self["n_jet"]
    self.all_jets_mask = ak.unflatten(
        np.repeat(np.array(self.all_events_mask, dtype=bool), njet), njet)

    self.mask = self.all_events_mask
    self.jets_selected = self.all_jets_mask

    self.sixb_jet_mask = self["jet_signalId"] != -1
    self.bkgs_jet_mask = self.sixb_jet_mask == False

    self.sixb_found_mask = self["nfound_presel"] == 6


class Tree:
    def __init__(self, filelist):
        if type(filelist) != list:
            filelist = [filelist]
        self.filelist = [SixBFile(fn) for fn in filelist]

        init_sample(self)
        init_tree(self)
        init_selection(self)

        # self.reco_XY()
    def __str__(self):
        sample_string = [
            f"=== File Info ===",
            f"File: {[fn.fname for fn in self.filelist]}",
            f"Total Events:    {[fn.total_events for fn in self.filelist]}",
            f"Raw Events:      {[fn.raw_events for fn in self.filelist]}",
        ]
        return "\n".join(sample_string)

    def __getitem__(self, key):
        item = self.ttree[key]
        return item

    def __getattr__(self,key): return self[key]
    def get(self, key): return self[key]

    def expected_events(self, lumikey=2018):
        lumi, _ = lumiMap[lumikey]
        return ak.sum(self["scale"])*lumi

    def extend(self,*args, **kwargs): 
        self.ttree = join_fields(self.ttree,*args, **kwargs)
        self.fields = self.ttree.fields

    def copy(self):
        new_tree = CopyTree(self)
        return new_tree

    def subset(self,nentries,randomize=True): 
        tree = self.copy()
        mask = [True]*nentries + [False]*(len(self.ttree)-nentries)
        if randomize: np.random.shuffle(mask)
        tree.ttree = tree.ttree[mask]
        return tree


class CopyTree(Tree):
    def __init__(self, tree):
        copy_fields(tree, self)

def reco_XY(self):
    def bjet_p4(key): return vector.obj(pt=self[f"gen_{key}_recojet_pt"], eta=self[f"gen_{key}_recojet_eta"],
                                        phi=self[f"gen_{key}_recojet_phi"], mass=self[f"gen_{key}_recojet_m"])
    hx_b1 = bjet_p4("HX_b1")
    hx_b2 = bjet_p4("HX_b2")
    hy1_b1 = bjet_p4("HY1_b1")
    hy1_b2 = bjet_p4("HY1_b2")
    hy2_b1 = bjet_p4("HY2_b1")
    hy2_b2 = bjet_p4("HY2_b2")

    Y = hy1_b1 + hy1_b2 + hy2_b1 + hy2_b2
    X = hx_b1 + hx_b2 + Y

    self.extend(**{"X_pt": X.pt, "X_m": X.mass, "X_eta": X.eta, "X_phi": X.phi,
                    "Y_pt": Y.pt, "Y_m": Y.mass, "Y_eta": Y.eta, "Y_phi": Y.phi})

def calc_jet_dr(self, compare=None, tag="jet"):
    select_eta = self.get("jet_eta")
    select_phi = self["jet_phi"]

    if compare is None:
        compare = self.jets_selected

    compare_eta = self["jet_eta"][compare]
    compare_phi = self["jet_phi"][compare]

    dr = calc_dr(select_eta, select_phi, compare_eta, compare_phi)
    dr_index = ak.local_index(dr, axis=-1)

    remove_self = dr != 0
    dr = dr[remove_self]
    dr_index = dr_index[remove_self]

    imin_dr = ak.argmin(dr, axis=-1, keepdims=True)
    imax_dr = ak.argmax(dr, axis=-1, keepdims=True)

    min_dr = ak.flatten(dr[imin_dr], axis=-1)
    imin_dr = ak.flatten(dr_index[imin_dr], axis=-1)

    max_dr = ak.flatten(dr[imax_dr], axis=-1)
    imax_dr = ak.flatten(dr_index[imax_dr], axis=-1)

    self.extend(**{f"{tag}_min_dr": min_dr, f"{tag}_imin_dr": imin_dr,
                    f"{tag}_max_dr": max_dr, f"{tag}_imax_dr": imax_dr})

def calc_event_shapes(self):
    jet_pt, jet_eta, jet_phi, jet_m = self.get("jet_pt"), self.get(
        "jet_eta"), self.get("jet_phi"), self.get("jet_m")

    self.extend(
        **calc_y23(jet_pt),
        **calc_sphericity(jet_pt, jet_eta, jet_phi, jet_m),
        **calc_thrust(jet_pt, jet_eta, jet_phi, jet_m),
        **calc_asymmetry(jet_pt, jet_eta, jet_phi, jet_m),
    )

def calc_btagsum(self):
    for nj in (5, 6):
        self.extend(
            **{f"jet{nj}_btagsum": ak.sum(self.get("jet_btag")[:, :nj], axis=-1)})