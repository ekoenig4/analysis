from ..utils import *
from ..classUtils import TreeIter


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
           'bit': lambda a, v: (1 << v) == a & (1 << v), 'neq': lambda a, v: a != v,
           'mask': lambda a, v: a[v]}


def update_cutflow(tree, tag):
    tree.cutflow_labels = tree.cutflow_labels+[tag]
    new_cutflow = [np.append(cutflow, ak.sum(tree['sample_id'] == i))
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


def event_filter(self, tree):
    tree = tree.copy()

    collection = tree.ttree
    if self.mask is not None:
        mask = self.mask
    else:
        mask = True
        
    for filter in self.filters:
        mask = mask & filter(collection)
    tree.extend(collection[mask])
    update_cutflow(tree, self.name)

    return tree


class EventFilter:
    def __init__(self, name, mask=None, **kwargs):
        self.name = name
        self.mask = mask
        self.filters = [build_event_filter(key, value)
                        for key, value in kwargs.items()]

    def filter(self, tree, filter=None):
        if filter:
            tree = filter.filter(tree)
        if type(tree) == list:
            return [event_filter(self, t) for t in tree]
        if str(type(tree)) == str(TreeIter):
            return TreeIter([event_filter(self, t) for t in tree])
        return event_filter(self, tree)


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
    def __init__(self, collection, newname=None, mask=None, **kwargs):
        self.collection = collection
        self.newname = newname if newname else collection
        self.mask = mask
        self.filters = [build_collection_filter(collection, key, value)
                        for key, value in kwargs.items()]

    def filter(self, tree):
        if type(tree) == list:
            return [collection_filter(self, t) for t in tree]
        if str(type(tree)) == str(TreeIter):
            return TreeIter([collection_filter(self, t) for t in tree])
        return collection_filter(self, tree)


class FilterSequence:
    def __init__(self, *filters):
        self.filters = filters

    def filter(self, tree):
        for filter in self.filters:
            tree = filter.filter(tree)
        return tree
