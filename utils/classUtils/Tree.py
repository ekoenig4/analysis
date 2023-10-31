import traceback

from .. import config

from ..hepUtils import *
from ..xsecUtils import *
from ..utils import *

from .AttrArray import AttrArray
# from ..fileUtils import eos

from ..fileUtils import fs

eos = fs.eos

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
    if eos.exists(fname): return eos.fullpath(fname)
    return fname

def _glob_files(pattern):
    files = glob.glob(pattern)
    if any(files): return files
    files = eos.glob(pattern)
    if any(files): return files
    return []

def copy_to_local(fname):
    if os.path.isfile(fname): return fname

    import re
    remote_pattern = re.compile(r'^root://(.*?)//(.*)$')
    match = remote_pattern.match(fname)
    if not match: return fname
    
    local_path = os.path.join(config.local_store, 'root', match.group(1), match.group(2))
    print('Using local path:', local_path)
    if not os.path.isfile(local_path):
        print(f'Copying {fname} to local path')
        fs.xrd.copy(fname, local_path)

    return local_path


class RootFile:
    def __init__(self, fname, treename='sixBtree', sample=None, xsec=None, normalization=None, fields=None):
        self.true_fname = fname
        self.fname = copy_to_local(fname)

        if self.fname is None: 
            print(f'[WARNING] skipping {self.fname}, was not found.')
            return
        
        self.treename = treename

        with ut.open(f'{self.fname}:{self.treename}', timecut=500) as tree:
            self.raw_events = tree.num_entries
            self.total_events = self.raw_events
            self.fields = [ str(branch) for branch in tree.keys() ]

            if fields is not None:
                fields = [ field for field in fields if field in self.fields ]
                self.arrays = tree.arrays(fields, library='ak')
            else:
                self.arrays = None

        if normalization is not None:       
            self.set_normalization(normalization)   
            
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
            
    def set_normalization(self, normalization):
        if isinstance(normalization, dict):
            self.set_dict_normalization(normalization)
        elif ':' in normalization:
            self.set_tree_normalization(normalization)
        elif 'cutflow' in normalization:
            self.set_cutflow_normalization(normalization)
        elif 'eff_histo' in normalization:
            self.set_eff_histo_normalization(normalization)
        elif normalization:
            self.set_histo_normalization(normalization)

    def set_eff_histo_normalization(self, eff_histo):
        with ut.open(f'{self.fname}:{eff_histo}') as eff_histo:
            self.cutflow_labels = eff_histo.axis().labels()
            self.cutflow = eff_histo
            self.total_events = np.sum(self.cutflow.counts()[np.array(self.cutflow_labels) == 'Ntot_w'])

    def set_cutflow_normalization(self, cutflow):
        with ut.open(f'{self.fname}:{cutflow}') as cutflow:
            self.cutflow_labels = cutflow.axis().labels()
            self.cutflow = cutflow
            if self.cutflow_labels is None: self.cutflow_labels = []
            self.total_events = self.cutflow.counts()[0]

    def set_histo_normalization(self, histo):
        with ut.open(f'{self.fname}:{histo}') as histo:
            self.total_events = histo.Integral()

    def set_dict_normalization(self, norms):
        raise NotImplementedError

    def set_tree_normalization(self, treebranch):
        treename, branchname = treebranch.split(':')
        with ut.open(f'{self.fname}:{treename}') as tree:
            self.total_events = ak.sum(tree[branchname].array())

    def write(self, altfile, retry=2, tree=None, types=None, **kwargs):

        fname = eos.cleanpath(self.fname)
        if callable(altfile): output = altfile(fname)
        else:
            dirname, basename = os.path.dirname(fname), os.path.basename(fname)
            output = os.path.join(dirname, altfile.format(base=basename))

        if re.match(r'^root://(.*?)//(.*)$', output):
            tmp_output = '_'.join(output.split('/'))
            copy_to_remote = True
        else:
            tmp_output = output
            copy_to_remote = False

        # character limit for filenames is 255 bytes
        # remove the start of the filename to make it fit
        if len(tmp_output) > 255:
            tmp_output = tmp_output[-255:]

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
        if copy_to_remote:
            eos.move(tmp_output, output)

def init_files(self, filelist, treename, normalization, altfile="{base}", report=True, xsec=None, fields=None):
    if type(filelist) == str:
        if filelist.endswith('.txt'):
            with open(filelist, 'r') as f:
                filelist = f.readlines()
            filelist = [ f.strip() for f in filelist ]
        else:
            filelist = [filelist]
    filelist = list(filelist)

    def use_altfile(f):
        if callable(altfile): return altfile(f)

        dirname, basename = os.path.dirname(f), os.path.basename(f)
        basename = altfile.format(base=basename)
        f_altfile = os.path.join(dirname, basename)
        return f_altfile

    glob_filelist = [ use_altfile(fn) for flist in filelist for fn in _glob_files(flist) ]

    if not any(glob_filelist):
        glob_filelist = [ fn for flist in filelist for fn in _glob_files( use_altfile(flist) ) ]

    if any(glob_filelist):
        filelist = glob_filelist

    it = tqdm(filelist) if report else iter(filelist)
    self.filelist = []
    xsec = AttrArray.init_attr(xsec, None, len(filelist))
    for i, fn in enumerate(it):
        try:
            self.filelist.append( RootFile(fn, treename, normalization=normalization, xsec=xsec[i], fields=fields) )
        except Exception as err:
            traceback.print_exc()
            print(f'[WARNING] skipping {fn}, unable to open.')
            
    # Fix normalization when using multiple files of the same sample
    samples = defaultdict(lambda:0)
    for f in self.filelist:
        samples[f.sample] += f.total_events

    histograms = defaultdict(list)
    for f in self.filelist:
        f.total_events = samples[f.sample]
        if normalization:
            f.scale = f.xsec / \
                f.total_events if type(f.xsec) == float else 1
        else:
            f.scale = 1
        
        for key, histogram in f.histograms.items():
            from ..plotUtils import Histo
            histograms[key].append( Histo.convert(histogram, scale=f.scale) )
        
    self.histograms = {
        key: functools.reduce(Histo.add, histogram)
        for key, histogram in histograms.items()
    }
    
    self.filelist = [ fn for fn in self.filelist if fn.fname and fn.raw_events > 0 ]
    self.lazy = True

def init_sample(self):  # Helper Method For Tree Class

    data_types = ['Data', 'JetHT', 'jetht']
    self.is_data = any( data_type in fn.fname for data_type in data_types for fn in self.filelist )
    
    self.nmssm_signal = all("NMSSM" in fn.fname for fn in self.filelist) 

    dih_types = ['GluGluToHHTo4B','ggHH4b']
    self.dih_signal = any( dih_type in fn.fname for dih_type in dih_types for fn in self.filelist )
    self.is_signal = self.nmssm_signal or self.dih_signal
    self.is_model = False
    
    sample_tags = set(fn.sample for fn in self.filelist)
    if len(sample_tags) == 1:
        self.sample = list(sample_tags)[0]

    else:
        sample_tag = [next((tag for key, tag in tagMap.items(
        ) if key in fn.sample), None) for fn in self.filelist]
        if (sample_tag.count(sample_tag[0]) == len(sample_tag)):
            self.sample = sample_tag[0]
        else:
            self.sample = "MC-Bkg"

        if self.is_data:
            self.sample = "Data"

    if self.nmssm_signal:

        def mass_point(fname):
            point = re.findall('MX_\d+_MY_\d+', fname)
            if any(point): return point[0]
            
            point = re.findall('MX-\d+_MY-\d+', fname)
            if any(point): return point[0].replace('-','_')
        points = [ mass_point(fn.fname) for fn in self.filelist ]
        if len(set(points)) == 1:
            self.sample = points[0]
            mx, my = self.sample.split("_")[1::2]
            self.mass = f'({mx}, {my})'
            self.mx = int(mx)
            self.my = int(my)
        
    self.color = colorMap.get(self.sample, None)
    if self.color is not None and not isinstance(self.color, str): self.color = next(self.color)
    self.pltargs = dict()

def init_empty(self):
    self.ttree = ak.Array(dict())

    self.is_data = False
    self.nmssm_signal = False
    self.is_signal = False
    self.is_model = False
    self.sample = None
    self.color = None
    self.pltargs = dict()
    self.systematics = None

    from ..plotUtils import Histo
    self.cutflow = [Histo(counts=np.array([]), bins=np.array([]))]
    self.cutflow_labels = []

def init_tree(self, weights=['genWeight'], cache=None, normalization=None, fields=None):
    if weights is None: weights = []
    
    self.ttree = ut.lazy([ f'{fn.fname}:{fn.treename}' for fn in self.filelist ])
    self.fields = self.ttree.fields

    if fields is not None:
        fields = [ field for field in fields if field in self.fields ]
        arrays = ak.concatenate([fn.arrays for fn in self.filelist])
        for field in fields:
            self.ttree[field] = arrays[field]

    weights = [ weight for weight in weights if any(field in weight for field in self.fields) ]
    scale = functools.reduce(lambda x, y : x * y, [self[weight] for weight in weights], 1)

    self.extend(
        sample_id=ak.concatenate([ak.Array([i]*fn.raw_events)
                                 for i, fn in enumerate(self.filelist)]),
        scale=scale*ak.concatenate([np.full(shape=fn.raw_events, fill_value=fn.scale, dtype=np.float) for fn in self.filelist])
    )

    self.raw_events = sum(fn.raw_events for fn in self.filelist)

    from ..plotUtils import Histo

    if normalization == 'h_cutflow':
        self.cutflow = [ Histo.convert(fn.cutflow) for fn in self.filelist]
        self.cutflow_labels = max([ fn.cutflow_labels for fn in self.filelist])

        def _trim_cutflow(cutflow):
            cutflow.histo = np.trim_zeros(cutflow.histo, 'b')
            cutflow.error = cutflow.error[:len(cutflow.histo)]
            cutflow.bins = np.arange(len(cutflow.histo))
        for cutflow in self.cutflow: _trim_cutflow(cutflow)
        ncutflow = max( len(cutflow.histo) for cutflow in self.cutflow )
        # self.cutflow_labels = np.arange(ncutflow).astype(str).tolist()

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
    else:
        self.cutflow = []
        self.cutflow_labels = []


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

class AccessCache:

    def __init__(self):
        self.cache = set()
        self.fname = None
        self.save_on_change = False

    def add(self, item : str):
        if item.startswith('_'): return

        if item not in self.cache:
            self.cache.add(item)
            if self.save_on_change and self.fname:
                self.save()

    def update(self, items):
        items = [ item for item in items if not item.startswith('_') ]
        newitems = set(items) - self.cache

        if any(newitems):
            self.cache.update(newitems)
            if self.save_on_change and self.fname:
                self.save()

    def __contains__(self, item):
        return item in self.cache
    def __len__(self):
        return len(self.cache)
    
    def clear(self):
        self.cache.clear()

    def save(self, base=f'{config.GIT_WD}/.cache/treefields/'):
        if not os.path.exists(base): os.makedirs(base)
        fname = os.path.basename(self.fname)
        fname = os.path.join(base, fname)
        with open(fname, 'w') as f:
            f.write('\n'.join(self.cache))

    def load(self, fname, base=f'{config.GIT_WD}/.cache/treefields/', **kwargs):
        self.__dict__.update(kwargs)

        fname = os.path.basename(fname)
        self.fname = fname

        fname = os.path.join(base, fname)
        if not os.path.exists(fname): 
            self.cache = set()
            return None
        
        with open(fname, 'r') as f:
            self.cache = set(f.read().splitlines())

        return self.cache

class Tree:
    accessed_fields = AccessCache()

    @classmethod
    def from_ak(cls, ak_tree, **kwargs):
        tree = cls([])
        tree.extend(ak_tree, scale=np.ones(len(ak_tree)), sample_id=np.zeros(len(ak_tree)))
        tree.__dict__.update(**kwargs)

        return tree


    def __init__(self, filelist, altfile="{base}", report=True, treename='sixBtree', weights=['genWeight'], normalization='h_cutflow', xsec=None, fields=None, **kwargs):
        self._recursion_safe_guard_stack = []
        self.varmap = dict()

        init_files(self, filelist, treename, normalization, altfile, report, xsec=xsec, fields=fields)

        if not any(self.filelist):
            init_empty(self)
            print('[WARNING] unable to open any files with filelist')
            print('          ', filelist)
        else:
            init_sample(self)
            init_tree(self, weights, normalization=normalization)

        self.reductions = dict()
        self.__dict__.update(**kwargs)


    def __str__(self):
        sample_string = [
            f"=== File Info ===",
            f"File: {[fn.fname for fn in self.filelist]}",
            f"Total Events:    {[fn.total_events for fn in self.filelist]}",
            f"Raw Events:      {[fn.raw_events for fn in self.filelist]}",
        ]
        return "\n".join(sample_string)
    
    def __getitem__(self, key): 
        if isinstance(key, list):
            return self.ttree[key]
        
        if hasattr(self, 'varmap') and key in self.varmap and not key in self.fields:
            self.ttree[key] = self.ttree[self.varmap[key]]

        if key in self.ttree.fields:
            Tree.accessed_fields.add(key)
            return self.ttree[key]
        return self.get_expr(key)
    
    def get_expr(self, expr):
        import ast
        scope = dict(ak=ak, np=np, len=len)

        # get fields referenced in expression
        fields = set()
        for node in ast.walk(ast.parse(expr)):
            if isinstance(node, ast.Name):
                fields.add(node.id)
        fields = fields - set(scope.keys())

        # get fields from tree
        fields = { field: self.ttree[field] for field in fields }
        Tree.accessed_fields.update(fields.keys())

        # evaluate expression
        return eval(expr, scope, fields)
        
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
    def get_histogram(self, key):
        if key in self.histograms: return self.histograms[key]

        for file in self.filelist:
            file.load_histograms([key])

        from ..plotUtils import Histo
        histograms = [ Histo.convert(file.histograms[key], scale=file.scale) for file in self.filelist ]
        histogram = functools.reduce(Histo.add, histograms)

        self.histograms[key] = histogram
        return histogram

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

    def copy(self, **kwargs):
        new_tree = CopyTree(self)
        new_tree.__dict__.update(**kwargs)
        return new_tree

    def subset(self, range=None, nentries=None, fraction=None, randomize=True):
        tree = self.copy()

        if fraction: range = (0,int(fraction*len(self.ttree)))
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

    def write(self, fname, include=[], exclude=[], treename='Events'):
        if callable(fname): fname = fname(self)
        if not fname.endswith('.root'): fname += '.root'
        
        if re.match(r'^root://(.*?)//(.*)$', fname):
            import hashlib 
            tmp_fname = f'{hashlib.md5(fname.encode()).hexdigest()}.root'
            copy_to_remote = True
        else:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            tmp_fname = fname
            copy_to_remote = False
        
        exclude += ['_.*', 'sample_id']

        fields = []
        for field in self.fields:
            if any( re.match(f'^{pattern}$', field) for pattern in exclude ): continue
            if any( re.match(f'^{pattern}$', field) for pattern in include ) or not any(include):
                fields.append(field)

        if not any(fields):
            print('[WARNING] no fields to write')
            return

        arrays = { field : self.ttree[field] for field in fields }

        print(f'Writing {self.sample} to {fname}')
        with ut.recreate(tmp_fname) as f:
            f[treename] = arrays

        if copy_to_remote:
            fs.xrd.move(tmp_fname, fname)
    
    def write_by_sample(self, altfile='new_{base}', retry=2, include=[], exclude=[]):

        if not callable(altfile):
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
                    field: ak.fill_none( ak.pad_none(tree[field], 1, axis=-1), -999)
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
            extra = dict()
            try:
                extra['h_cutflow'] = (self.cutflow[i].histo, self.cutflow[i].bins)
            except Exception:
                ...
            file.write(altfile, retry=retry, tree=tree, types=types, **extra)

class CopyTree(Tree):
    def __init__(self, tree):
        copy_fields(tree, self)