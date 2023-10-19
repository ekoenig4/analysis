from ..utils import *
from ..classUtils import ObjIter
import inspect


def get_operation(tags, operation, default):
    k, v = next(((tag, operation[tag])
                for tag in tags if tag in operation), (None, None))
    if k:
        tags.remove(k)
    else:
        v = default
    return v


functions = {'abs': lambda a: np.abs(a), }

methods = {'min': lambda a, v: a > v, 'max': lambda a, v: a < v,
           'emin': lambda a, v: a >= v, 'emax': lambda a, v: a <= v,
           'bit': lambda a, v: ((a >> v) & 1) == 1, 'neq': lambda a, v: a != v,
           'mask': lambda a, v: a[v]}


def update_cutflow(tree, tag):
    tree.cutflow_labels = tree.cutflow_labels+[tag]

    def event_weight(tree, i):
        if 'genWeight' in tree.fields:
            return tree['genWeight']*(tree['sample_id']==i)
        return tree['sample_id']==i

    def _update_cutflow(cutflow, weights):
        from ..plotUtils import Histo
        histo = np.append(cutflow.histo, np.sum(weights))
        error = np.append(cutflow.error, np.sqrt( np.sum(weights**2) ))
        bins = np.arange( len(histo)+1 )
        return Histo(histo, bins, error)

    new_cutflow = [_update_cutflow(cutflow, event_weight(tree, i))
                   for i, cutflow in enumerate(tree.cutflow)]
    tree.cutflow = new_cutflow


def build_event_filter(key, value, functions=functions, methods=methods):
    tags = key.split('_')

    function = get_operation(tags, functions, lambda a: a)
    if function is None:
        def function(a): return a

    method = get_operation(tags, methods, lambda a, v: a == v)
    if method is None:
        def method(a, v): return a == v
    variable, index = '_'.join(tags), None
    if ":" in variable:
        variable, index = variable.split(":")

    def operation(collection):
        array = collection[variable]
        if index is not None:
            array = array[:, int(index)]
        return method(function(array), value)
    return operation


def event_filter(self, tree, cutflow=True):
    tree = tree.copy()

    if self.mask is not None:
        mask = self.mask
    else:
        mask = True

    for filter in self.filters:
        mask = mask & filter(tree)

    mask = ak.to_numpy(mask)
    if self.verbose:
        scale = tree.scale if hasattr(tree, 'scale') else np.ones(len(tree))
        total = np.sum(scale)
        eff = np.sum(scale[mask])/total
        print(f'{tree.sample} {self.name} eff: {eff:.2e}')

    tree.extend(tree.ttree[mask])

    if cutflow:
        update_cutflow(tree, self.name)

    return tree

class Filter:
    def __init__(self, filter):
        self.filter = filter 
        self.hash = f'_filter_{hash(filter)}_'
    def __call__(self, t, **kwargs):
        key = f'{self.hash}_{hash(tuple(kwargs.items()))}_'
        if key in t.fields: return t[key]
        
        value = self.filter(t, **kwargs)
        if hasattr(t, 'extend'): t.extend(**{key:value})
        return value

        


class EventFilter:
    def __init__(self, name, mask=None, filter=None, cutflow=True, verbose=False, **kwargs):
        self.name = name
        self.mask = mask
        self.kwargs = kwargs
        self.filters = [build_event_filter(key, value)
                        for key, value in kwargs.items()]
        if filter is not None:
            self.filters = [filter] + self.filters

        self.cutflow = cutflow
        self.verbose = verbose
            
    def filter(self, tree, filter=None):
        if filter:
            tree = filter.filter(tree)
        if isinstance(tree,list):
            return [event_filter(self, t, self.cutflow) for t in tree]
        return event_filter(self, tree, self.cutflow)
    
    def __str__(self): return f"<EventFilter: {self.name}>"
    def __repr__(self): return f"<EventFilter: {self.name} {self.kwargs}>"
    def __call__(self, tree, filter=None): return self.filter(tree, filter)


def build_collection_filter(name, key, value, functions=functions, methods=methods):
    tags = key.split('_')

    function = get_operation(tags, functions, lambda a: a)
    if function is None:
        def function(a): return a

    method = get_operation(tags, methods, lambda a, v: a == v)
    if method is None:
        def method(a, v): return a == v
    variable = name+'_'+'_'.join(tags)
    def operation(collection): return method(
        function(getattr(collection, variable)), value)
    return operation


def collection_filter(self, tree):
    tree = tree.copy()

    collection = get_collection(tree, self.collection)

    if self.mask is not None:
        collection = collection[self.mask]

    setattr(collection, f"{self.collection}_index", ak.local_index(
        collection[f"{self.collection}_pt"], axis=-1))

    mask = getattr(collection, f'{self.collection}_index') > -1
    for filter in self.filters:
        mask = mask & filter(collection)

    collection_records = {f"n_{self.newname}": ak.sum(mask, axis=-1)}
    collection_records.update({field.replace(self.collection, self.newname): array for field, array in zip(
        collection.fields, ak.unzip(collection[mask])) if not field.endswith('index')})
    tree.extend(**collection_records)

    return tree


class CollectionFilter:
    def __init__(self, collection, newname=None, mask=None, filter=None, **kwargs):
        self.collection = collection
        self.newname = newname if newname else collection
        self.mask = mask
        self.filters = [build_collection_filter(collection, key, value)
                        for key, value in kwargs.items()]
        if filter is not None:
            self.filters = [filter] + self.filters

    def filter(self, tree):
        if isinstance(tree,list):
            return [collection_filter(self, t) for t in tree]
        return collection_filter(self, tree)

    def __call__(self, tree): return self.filter(tree)


class FilterSequence:
    def __init__(self, *filters):
        self.filters = filters

    def filter(self, tree):
        for filter in self.filters:
            tree = filter.filter(tree)
        return tree

    def __call__(self, tree): return self.filter(tree)
