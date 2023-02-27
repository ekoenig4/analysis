import numpy as np
import itertools

def _combinations(items, ks):
    if len(ks) == 1:
        for c in itertools.combinations(items, ks[0]):
            yield (c,)

    else:
        for c_first in itertools.combinations(items, ks[0]):
            items_remaining= set(items) - set(c_first)
            for c_other in \
            _combinations(items_remaining, ks[1:]):
                if len(c_first)!=len(c_other[0]) or c_first<c_other[0]:
                    yield (c_first,) + c_other

def get_combinations(nitems, ks):
    combs = []
    for cs in _combinations(np.arange(nitems), ks):
        _combs = []
        for c in cs:
            _combs.append(list(c))
        combs.append(_combs)

    array = np.array(combs)
    return array

def _flatten(ks):
    if len(ks) == 0:
        return ks
    if isinstance(ks[0], list):
        return _flatten(ks[0]) + _flatten(ks[1:])
    return ks[:1] + _flatten(ks[1:])
def _grouped(ks):
    return [ (len(k) if isinstance(k,list) else 1) for k in ks ]

def combinations(nitems, ks):
    ks_flatten = _flatten(ks)
    ks_grouped = _grouped(ks)

    cb_flatten = get_combinations(nitems, ks_flatten)
    cb_grouped = get_combinations(sum(ks_grouped), ks_grouped)
    combs = np.concatenate([ cb_flatten.T[:, np.array(cb_grouped[:,i].tolist()).T ] for i in range(cb_grouped.shape[1]) ], axis=1)
    combs = combs.reshape(*combs.shape[:-2], -1).T
    return combs

def to_pair_combinations(o1_index, o2_index, nobjs=None):
    if nobjs is None:
      nobjs = max( np.max(o1_index), np.max(o2_index) )+1
    i, j = o1_index, o2_index
    k = (-0.5*i*i + (nobjs-0.5)*i + j - i - 1).astype(int)
    return k

def map_to_collection(o1_index, o2_index, collection):
    """Maps a tensor of combination indicies to a collection of combinations
    """
    max_id = np.max(collection)+1
    hash_func = lambda i,j : i + max_id*j
    hash_map = { int(k):v for v, k in enumerate( hash_func(collection[0], collection[1]) ) }
    return np.vectorize(hash_map.get)(np.vectorize(hash_func)(o1_index, o2_index))
    # k = hash_func(o1_index, o2_index).apply_(lambda k : hash_map[k])
    # return k