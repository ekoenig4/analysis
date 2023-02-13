from ..selectUtils import *
from ..xsecUtils import *
from ..utils import *
from ..fileUtils.eos import *

import uproot as ut
import awkward as ak
import numpy as np
import re, glob, os
import functools

from tqdm import tqdm
import subprocess
from collections import defaultdict

def _check_file(fname):
    if os.path.isfile(fname): return fname
    if eos.exists(fname): return eos.url+'/'+fname
    return None

def _glob_files(pattern):
    files = glob.glob(pattern)
    if any(files): return files
    files = eos.ls(pattern, with_path=True)
    if any(files): return files
    return []

class SixBFile:
    def __init__(self, fname):
        self.fname = fname
        self.tfile = ut.open(fname)

        cutflow = self.tfile.get('h_cutflow')
        self.cutflow_labels = cutflow.axis().labels()
        if self.cutflow_labels is None: self.cutflow_labels = []
        self.cutflow = cutflow.to_numpy()[0]
        self.total_events = self.cutflow[0]

        self.sample, self.xsec = next(
            ((key, value) for key, value in xsecMap.items() if key in fname), ("unk", 1))

        self.scale = self.xsec / \
            self.total_events if type(self.xsec) == float else 1

        self.ttree = self.tfile.get('sixBtree')
        self.raw_events = self.ttree.num_entries

class LazySixBFile:
    def __init__(self, fname, use_cutflow=True):
        self.fname = _check_file(fname)
        self.raw_events = 0
        self.total_events = 0

        if self.fname is None: 
            print(f'[WARNING] skipping {self.fname}, was not found.')
            return
        
        if use_cutflow:
            try:
                with ut.open(f'{self.fname}:h_cutflow') as cutflow:
                    self.cutflow_labels = cutflow.axis().labels()
                    self.cutflow = cutflow
                    if self.cutflow_labels is None: self.cutflow_labels = []
                    self.total_events = self.cutflow.counts()[0]
            except ut.KeyInFileError:
                return

        self.normtree = f'{self.fname}:NormWeightTree' 
        with ut.open(self.fname) as f:
            keys = [ key[:-2] for key in f.keys() ]
            histograms = [ key for key in keys if key not in ('sixBtree','h_cutflow','h_cutflow_unweighted','NormWeightTree') ]
            self.histograms = { key : f[key] for key in histograms }

        self.sample, self.xsec = next(
            ((key, value) for key, value in xsecMap.items() if key in self.fname), ("unk", 1))

        self.scale = self.xsec / \
            self.total_events if type(self.xsec) == float else 1

        tree = ut.lazy(f'{self.fname}:sixBtree')
        self.raw_events = len(tree)
        self.fields = tree.fields
        del tree

    def write(self, altfile, retry=2, tree=None, types=None, **kwargs):
        dirname, basename = os.path.dirname(self.fname), os.path.basename(self.fname)
        dirname = dirname.replace(eos.url, '')
        output = os.path.join(dirname, altfile.format(base=basename))

        tmp_output = '_'.join(output.split('/'))
        kwargs.update( **self.histograms )

        print(f'Writing {output}')
        for i in range(retry):
            try:
                with ut.recreate(tmp_output) as f:
                    for key, value in kwargs.items():
                        f[key] = value

                    f.mktree('sixBtree', types)
                    f['sixBtree'].extend(tree)
                break
            except ValueError:
                ...

        move_to_eos(tmp_output, output)

def init_files(self, filelist, altfile="{base}", report=True):
    if type(filelist) == str:
        filelist = [filelist]

    def use_altfile(f):
        dirname, basename = os.path.dirname(f), os.path.basename(f)
        basename = altfile.format(base=basename)
        return os.path.join(dirname, basename)

    filelist = [ use_altfile(fn) for flist in filelist for fn in _glob_files(flist) ]
    # self.filelist = [SixBFile(fn) for fn in filelist]

    it = tqdm(filelist) if report else iter(filelist)
    self.filelist = [LazySixBFile(fn) for fn in it]
    self.filelist = [ fn for fn in self.filelist if fn.total_events > 0 ]
    # Fix normalization when using multiple files of the same sample
    samples = defaultdict(lambda:0)
    for f in self.filelist:
        samples[f.sample] += f.total_events

    histograms = defaultdict(list)
    for f in self.filelist:
        f.total_events = samples[f.sample]
        f.scale = f.xsec / \
            f.total_events if type(f.xsec) == float else 1
        
        for key, histogram in f.histograms.items():
            from ..plotUtils import Histo
            histograms[key].append( Histo.from_th1d(histogram, scale=f.scale) )
        
    self.histograms = {
        key: functools.reduce(Histo.add, histogram)
        for key, histogram in histograms.items()
    }
    self.lazy = True

def init_sample(self):  # Helper Method For Tree Class
    self.is_data = any("Data" in fn.fname for fn in self.filelist)
    self.is_signal = all("NMSSM" in fn.fname for fn in self.filelist)
    self.is_model = False
    
    sample_tag = [next((tag for key, tag in tagMap.items(
    ) if key in fn.sample), None) for fn in self.filelist]
    if (sample_tag.count(sample_tag[0]) == len(sample_tag)):
        self.sample = sample_tag[0]
    else:
        self.sample = "MC-Bkg"

    if self.is_data:
        self.sample = "Data"

    if self.is_signal:
        points = [ re.findall('MX_\d+_MY_\d+',fn.fname)[0] for fn in self.filelist ]
        if len(set(points)) == 1:
            self.sample = points[0]
            mx, my = self.sample.split("_")[1::2]
            self.mass = f'({mx}, {my})'
        
    self.color = colorMap.get(self.sample, None)
    if self.color is not None and not isinstance(self.color, str): self.color = next(self.color)


def init_tree(self, use_gen=False, cache=None):
    self.fields = list(set.intersection(*[ set(fn.fields) for fn in self.filelist]))

    if self.lazy :
        self.ttree = ut.lazy([ f'{fn.fname}:sixBtree' for fn in self.filelist ], array_cache=cache)
    else:
        self.ttree = ut.lazy([fn.tree for fn in self.filelist])

    self.ttree = self.ttree[self.fields]

    scale = self.ttree['genWeight'] if (use_gen and 'genWeight' in self.fields) else 1

    self.extend(
        sample_id=ak.concatenate([ak.Array([i]*fn.raw_events)
                                 for i, fn in enumerate(self.filelist)]),
        scale=scale*ak.concatenate([np.full(shape=fn.raw_events, fill_value=fn.scale, dtype=np.float) for fn in self.filelist])
    )

    self.raw_events = sum(fn.raw_events for fn in self.filelist)
    # self.cutflow_labels = max(map(lambda fn : fn.cutflow_labels,self.filelist))
    # if not any(self.cutflow_labels):
    #     self.cutflow_labels = max(map(lambda fn : np.arange(len(fn.cutflow.counts())).astype(str).tolist(), self.filelist))
    # ncutflow = len(self.cutflow_labels) if any(self.cutflow_labels) else 0

    from ..plotUtils import Histo
    self.cutflow = [ Histo.from_th1d(fn.cutflow) for fn in self.filelist]

    def _trim_cutflow(cutflow):
        cutflow.histo = np.trim_zeros(cutflow.histo, 'b')
        cutflow.error = cutflow.error[:len(cutflow.histo)]
        cutflow.bins = np.arange(len(cutflow.histo))
    for cutflow in self.cutflow: _trim_cutflow(cutflow)
    ncutflow = max( len(cutflow.histo) for cutflow in self.cutflow )
    self.cutflow_labels = np.arange(ncutflow).astype(str).tolist()

    def _pad_cutflow(cutflow):
        pad = max(0, ncutflow - len(cutflow.histo))
        if pad > 0: 
            cutflow.histo = np.pad( cutflow.histo, (0, pad), constant_values=0 )
            cutflow.error = np.pad( cutflow.error, (0, pad), constant_values=0 )
        else:
            cutflow.histo = cutflow.histo[:ncutflow]
            cutflow.error = cutflow.error[:ncutflow]

        cutflow.bins = np.arange(ncutflow+1)
        return cutflow
    for cutflow in self.cutflow: _pad_cutflow(cutflow)


    self.systematics = None


def init_selection(self):
    self.all_events_mask = ak.Array([True]*self.raw_events)
    # njet = self["n_jet"]
    # self.all_jets_mask = ak.unflatten(
    #     np.repeat(np.array(self.all_events_mask, dtype=bool), njet), njet)

    # self.mask = self.all_events_mask
    # self.jets_selected = self.all_jets_mask

    # self.sixb_jet_mask = self["jet_signalId"] != -1
    # self.bkgs_jet_mask = self.sixb_jet_mask == False

    # self.sixb_found_mask = self["nfound_presel"] == 6
    
def _regex_field(self, regex):
    matched_fields = list(filter(lambda field : re.match(f"^{regex}$", field), self.ttree.fields))
    if regex not in matched_fields: return   
    item = ak.from_regular(
        ak.unflatten(
            ak.flatten(
                ak.zip(ak.unzip(self.ttree[matched_fields])),
                axis=None,
            ),
            len(matched_fields)
        )
    )
    return item

class Tree:
    def __init__(self, filelist, altfile="{base}", report=True, use_gen=True, cache=None):
        self._recursion_safe_guard_stack = []

        init_files(self, filelist, altfile, report)

        if not any(self.filelist):
            print('[WARNING] unable to open any files.')
            return
        # if not allow_empty:
        #     self.filelist = list(filter(lambda fn : fn.raw_events > 0,self.filelist))

        init_sample(self)
        init_tree(self, use_gen, cache)

        # init_selection(self)

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
        slice = None
        if isinstance(key,str) and any(re.findall(r'\[.*\]', key)):
            slice = re.findall(r'\[.*\]', key)[0]
            key = key.replace(slice, "")

        item = self.ttree[key]
        if slice is not None:
            item = eval(f'item{slice}', {'item': item})
        return item
    def __getattr__(self, key): 
        if len(self._recursion_safe_guard_stack) > 3:
            current_stack = self._recursion_safe_guard_stack
            self._recursion_safe_guard_stack = []
            raise RecursionError(f"recursive missing attribute error detected. key stack -> ({current_stack})")

        self._recursion_safe_guard_stack.insert(0, key)
        item = self[key]
        self._recursion_safe_guard_stack.pop(0)
        return item
    def __len__(self): return len(self.ttree)
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, d):
        self.__dict__ = d
    def get(self, key): return self[key]

    def expected_events(self, lumikey=2018):
        lumi, _ = lumiMap[lumikey]
        return ak.sum(self["scale"])*(1 if self.is_data else lumi)

    def extend(self, *args, **kwargs):
        self.ttree = join_fields(self.ttree, *args, **kwargs)
        self.fields = self.ttree.fields

    def reweight(self, rescale):
        if callable(rescale): rescale = rescale(self)
        if '_scale' not in self.fields:
            self.extend(_scale=self.scale)
        self.extend(scale=rescale*self.scale)

    def set_systematics(self, systematics):
        if not isinstance(systematics, list): systematics = [systematics]
        self.systematics = systematics

    def add_systematic(self, systematic):
        if self.systematics is None: self.systematics = []
        self.systematics.append(systematic)

    def copy(self):
        new_tree = CopyTree(self)
        return new_tree

    def subset(self, range=None, nentries=None, randomize=True):
        tree = self.copy()

        if nentries: range = (0,nentries)

        assert range is not None, "Specify a range (start,stop)"
        assert range[1] > range[0], "Start needs to be less then stop in range"
        assert len(self.ttree) >= range[1], "Specify a range within the tree"

        mask = np.full((len(self.ttree)),False)
        mask[range[0]:range[1]] = True
        if randomize:
            np.random.shuffle(mask)
        tree.ttree = tree.ttree[mask]
        return tree

    def reorder_collection(self,collection,order):
        tree = self.copy()
        collection = get_collection(tree,collection)
        tree.extend( reorder_collection(collection,order) )
        return tree

    def merge(self, other):
        tree = self.copy()
        tree.filelist = tree.filelist + other.filelist
        
        init_sample(tree)
        init_tree(tree)
        init_selection(tree)

        return tree

    def clear(self):
        total, remaining = 0,0
        for cache in self.ttrees.cache:
            total += cache.current
            cache.clear()
            remaining += cache.current
        freed = total - remaining
        return freed 

    def asmodel(self, name='model', color='lavender'):
        tree = self.copy()
        tree.is_model = True 
        tree.sample = name
        tree.color = color
        return tree
    
    def write(self, altfile='new_{base}', retry=2, include=[], exclude=[]):
        if '{base}' not in altfile: altfile += '_{base}'

        exclude += ['^_', '^sample_id$']

        def _prep_to_write_(tree):

            fields = tree.fields
            if any(include):
                regex = lambda field : any( re.match(pattern, field) for pattern in include )
            else:
                regex = lambda field : not any( re.match(pattern, field) for pattern in exclude )

            fields = [ field for field in fields if regex(field) ]
            tree = tree[fields]

            types = dict()
            option_fields = []
            for field in fields:
                types[field] = tree.type.type[field]
                
                if "?" in str(types[field]):
                    option_fields.append(field)

            if any(option_fields):
                option_fields = {
                    field: ak.from_numpy(tree[field].to_numpy().data)
                    for field in option_fields
                }
                tree = join_fields(tree, **option_fields)

                types.update({
                    field:tree.type.type[field]
                    for field in option_fields
                })
                
            return tree, types

        full_tree, types = _prep_to_write_(self.ttree)

        for i, file in tqdm(enumerate(self.filelist), total=len(self.filelist)):
            file_mask = self.sample_id == i
            if ak.sum(file_mask) == 0: continue

            tree = unzip_records(full_tree[ file_mask ])
            cutflow = (self.cutflow[i].histo, self.cutflow[i].bins)
            file.write(altfile, retry=retry, tree=tree, types=types, h_cutflow=cutflow)

class CopyTree(Tree):
    def __init__(self, tree):
        copy_fields(tree, self)


class TreeMethodIter:
    def __init__(self, trees, calls):
        self.trees = trees
        self.calls = calls
        self.calliter = zip(trees, calls)

    def __str__(self): return str(self.calls)
    def __iter__(self): return iter(self.calls)
    def __getitem__(self, key): return self.calls[key]

    def __call__(self, *a, args=lambda t: [], kwargs=lambda t: {}, **kw):
        f_args, f_kwargs = args, kwargs
        if not callable(f_args):
            def f_args(t): return args
        if not callable(f_kwargs):
            def f_kwargs(t): return kwargs

        def build_args(t): return list(a)+list(f_args(t))
        def build_kwargs(t): return dict(**f_kwargs(t), **kw)
        out = [call(*build_args(t), **build_kwargs(t)) for t, call in self.calliter]
        if not any( attr is None for attr in out ):
            return out
        


class TreeIter:
    def __init__(self, trees):
        self.trees = trees

    def __str__(self): return str(self.trees)
    def __iter__(self): return iter(self.trees)
    def __getitem__(self, key): 
        if type(key) == list: return TreeIter([ self.trees[k] for k in key ])
        if isinstance(key,slice): return TreeIter(self.trees[key])
        return self.trees[key]

    def __getattr__(self, key):
        attriter = [getattr(tree, key) for tree in self]
        if callable(attriter[0]):
            attriter = TreeMethodIter(self.trees, attriter)
        if not any( attr is None for attr in attriter ):
            return attriter
        
    def __add__(self,other):
        if type(other) == list: other = TreeIter(other)
        return TreeIter(self.trees+other.trees)
    
    def apply(self,tree_function):
        out = [ tree_function(tree) for tree in self ]
        if not any( attr is None for attr in out ):
            return out
        
    def copy(self):
        return TreeIter([tree.copy() for tree in self])

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
