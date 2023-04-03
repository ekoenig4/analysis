from ..hepUtils import *
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

class RootFile:
    def __init__(self, fname, treename='sixBtree', sample=None, xsec=None, normalization=None):
        self.fname = _check_file(fname)
        
        self.treename = treename

        tree = ut.lazy(f'{self.fname}:{self.treename}')
        self.raw_events = len(tree)
        self.total_events = self.raw_events
        self.fields = tree.fields
        del tree

        if self.fname is None: 
            print(f'[WARNING] skipping {self.fname}, was not found.')
            return

        if normalization is not None:
            if 'cutflow' in normalization:
                self.set_cutflow_normalization(normalization)
            elif normalization:
                self.set_histo_normalization(normalization)   

        _sample, _xsec = next(((key, value) for key, value in xsecMap.items() if key in self.fname), ("unk", 1))
        self.sample = _sample if sample is None else sample
        self.xsec = _xsec if xsec is None else xsec

        self.histograms = {}
        if getattr(self, 'cutflow', None) is None:
            from ..plotUtils import Histo
            self.cutflow = Histo(counts=np.array([self.raw_events]), bins=np.array([0,1]))

    def load_histograms(self, keys=None):
        with ut.open(self.fname) as f:
            keys = [ key[:-2] for key in f.keys() ]
            histograms = [ key for key in keys if key not in ('sixBtree','h_cutflow','NormWeightTree') ]
            self.histograms = { key : f[key] for key in histograms }
    
    def set_cutflow_normalization(self, cutflow):
        with ut.open(f'{self.fname}:{cutflow}') as cutflow:
            self.cutflow_labels = cutflow.axis().labels()
            self.cutflow = cutflow
            if self.cutflow_labels is None: self.cutflow_labels = []
            self.total_events = self.cutflow.counts()[0]

    def set_histo_normalization(self, histo):
        with ut.open(f'{self.fname}:{histo}') as histo:
            self.total_events = histo.Integral()

    def set_tree_normalization(self, tree):
        raise NotImplementedError("Need to implement NormWeightTree normalization")

    def write(self, altfile, retry=2, tree=None, types=None, **kwargs):
        dirname, basename = os.path.dirname(self.fname), os.path.basename(self.fname)
        dirname = dirname.replace(eos.url, '')
        output = os.path.join(dirname, altfile.format(base=basename))

        tmp_output = '_'.join(output.split('/'))
        kwargs.update( **getattr(self, 'histograms', {}) )

        print(f'Writing {output}')
        for i in range(retry):
            try:
                with ut.recreate(tmp_output) as f:
                    for key, value in kwargs.items():
                        f[key] = value
                    f[self.treename] = tree
                break
            except ValueError:
                ...

        move_to_eos(tmp_output, output)

class SixBFile:
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
            histograms = [ key for key in keys if key not in ('sixBtree','h_cutflow','NormWeightTree') ]
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

                    # f.mktree('sixBtree', types)
                    # f['sixBtree'].extend(tree)
                    f['sixBtree'] = tree
                break
            except ValueError:
                ...

        move_to_eos(tmp_output, output)

def init_files(self, filelist, treename, normalization, altfile="{base}", report=True):
    if type(filelist) == str:
        filelist = [filelist]

    def use_altfile(f):
        dirname, basename = os.path.dirname(f), os.path.basename(f)
        basename = altfile.format(base=basename)
        return os.path.join(dirname, basename)

    filelist = [ use_altfile(fn) for flist in filelist for fn in _glob_files(flist) ]

    it = tqdm(filelist) if report else iter(filelist)
    self.filelist = [ RootFile(fn, treename, normalization=normalization) for fn in it ]
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
            histograms[key].append( Histo.convert(histogram, scale=f.scale) )
        
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
            self.mx = int(mx)
            self.my = int(my)
        
    self.color = colorMap.get(self.sample, None)
    if self.color is not None and not isinstance(self.color, str): self.color = next(self.color)
    self.pltargs = dict()


def init_tree(self, use_gen=False, cache=None):
    self.fields = list(set.intersection(*[ set(fn.fields) for fn in self.filelist]))

    if self.lazy :
        self.ttree = ut.lazy([ f'{fn.fname}:{fn.treename}' for fn in self.filelist ])
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

    from ..plotUtils import Histo

    self.cutflow = [ Histo.convert(fn.cutflow) for fn in self.filelist]

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
    def __init__(self, filelist, altfile="{base}", report=True, use_gen=True, treename='sixBtree', normalization='h_cutflow'):
        self._recursion_safe_guard_stack = []

        init_files(self, filelist, treename, normalization, altfile, report)

        if not any(self.filelist):
            print('[WARNING] unable to open any files.')
            return

        init_sample(self)
        init_tree(self, use_gen)
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
        full_tree = remove_counters(full_tree)
        # full_tree = make_regular(full_tree)

        for i, file in tqdm(enumerate(self.filelist), total=len(self.filelist)):
            file_mask = self.sample_id == i
            if ak.sum(file_mask) == 0: continue

            tree = unzip_records(full_tree[ file_mask ])
            cutflow = (self.cutflow[i].histo, self.cutflow[i].bins)
            file.write(altfile, retry=retry, tree=tree, types=types, h_cutflow=cutflow)

class CopyTree(Tree):
    def __init__(self, tree):
        copy_fields(tree, self)