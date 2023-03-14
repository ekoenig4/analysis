import re
from typing import Callable

import awkward as ak
import numpy as np
import torch
import vector
from tqdm import tqdm

vector.register_awkward()

def check_instance(instance, classtype):
    if isinstance(instance, classtype): return True

    # if isinstance( getattr(instance, 'ttree', None), classtype): return True

    return False

def unzip_records(records):
    return {field: array for field, array in zip(records.fields, ak.unzip(records))}

def make_regular(records):
    variable_arrays = { field: ak.to_regular( records[field], axis=-1 ) for field in records.fields if 'var' in str(records[field].type) }
    regular_arrays =  { field: records[field] for field in records.fields if not 'var' in str(records[field].type) }

    arrays = dict(**regular_arrays, **variable_arrays)
    return ak.zip(arrays, depth_limit=1 )

def remove_counters(records):
    fields = [
        field 
        for field in records.fields
        if not ( field.startswith('n') and field[1:] in records.fields )
    ]

    return records[fields]

def join_fields(awk1, *args, **kwargs):
    args_unzipped = [unzip_records(awk) for awk in args]
    new_fields = {}
    for unzipped in args_unzipped:
        new_fields.update(unzipped)
    new_fields.update(kwargs)

    awk1_unzipped = {field: array for field,
                     array in zip(awk1.fields, ak.unzip(awk1))}
    awk1_unzipped.update(**new_fields)
    return ak.zip(awk1_unzipped, depth_limit=1)


def remove_name(collection, name):
    unzipped = {field.replace(name+'_', ''): array for field,
                array in zip(collection.fields, ak.unzip(collection))}
    return ak.zip(unzipped, depth_limit=1)


def get_collection(tree, name, named=True):
    collection_branches = list(
        filter(lambda branch: branch.startswith(name+'_'), tree.fields))

    branches = tree[collection_branches]
    if named:
        return branches
    return remove_name(branches, name)


def rename_collection(collection, newname, oldname=None):
    def rename_field(field):
        if oldname is not None:
            return field.replace(f"{oldname}_", f"{newname}_")
        return f"{newname}_{field}"
    fields = map(rename_field, collection.fields)
    collection_unzipped = {field: array for field,
                           array in zip(fields, ak.unzip(collection))}
    return ak.zip(collection_unzipped, depth_limit=1)


def reorder_collection(collection, order):
    return collection[order]


def build_collection(tree, pattern, name, ordered=None, ptordered=False, replace=False):
    fields = list(filter(lambda field: re.match(pattern, field), tree.fields))
    components = dict.fromkeys([re.search(pattern, field)[0]
                               for field in fields if re.search(pattern, field)]).keys()

    shared_fields = list(set.intersection(*[set(map(lambda field: field[len(component)+1:], filter(
        lambda field: field.startswith(component), fields))) for component in components]))

    components = [get_collection(tree, component, named=False)[
        shared_fields] for component in components]

    collection = {f'{name}_{field}': ak.concatenate(
        [component[field][:, None] for component in components], axis=-1) for field in shared_fields}

    if ptordered:
        ordered = "pt"

    if ordered:
        order = ak.argsort(-collection[f'{name}_{ordered}'], axis=-1)
        collection = {key: array[order] for key, array in collection.items()}
        collection.update(**{f'{name}_localid': order})

    tree.extend(**collection)


def _flatten(array):
    if check_instance(array, ak.Array):
        array = ak.flatten(array, axis=None)
        return ak.to_numpy(array)
    if check_instance(array, torch.Tensor):
        array = torch.flatten(array)
        return array.cpu().numpy()
    if check_instance(array, list):
        array = np.array(array)
        return array.reshape(-1).tolist()
    return np.array(array).reshape(-1)


def flatten(array, clean=True):
    array = _flatten(array)
    # mask = ~( np.isnan(array) | np.isinf( np.abs(array) ) )
    # return array[mask]
    return array


def cast_array(array):
    if check_instance(array, torch.Tensor):
        return array.cpu().numpy()
    return array

def restrict_array(array, bins, **params):
    x_lo = array >= bins[0]
    x_hi = array < bins[-1]
    array = array[x_lo & x_hi]
    if len(params) > 0:
        params = [param[x_lo & x_hi] for param in params.values()]
        return array, *params
    return array

def get_avg_std(array, weights=None, bins=None):
    mask = ~np.isnan(array) & ~(np.abs(array) == np.inf)
    array = ak.flatten(array[mask], axis=None)
    if weights is None:
        if bins is not None:
            array = restrict_array(array, bins)
        avg = ak.mean(array)
        std = ak.std(array)
    else:
        weights = ak.flatten(weights[mask], axis=None)
        if bins is not None:
            array, weights = restrict_array(array, bins, weights=weights)
        if len(array) == 0:
            return np.nan, np.nan
        avg = np.average(array, weights=weights)
        if bins is not None:
            avg = np.clip(avg, bins[0], bins[-1])
        std = np.sqrt(np.average((array-avg)**2, weights=weights))
    return avg, std

def ak_stack(arrays, axis=1):
    reshape = tuple([slice(None) for _ in range(axis)] + [None])

    return ak.concatenate(
        [array[reshape] for array in arrays],
        axis=axis
    )


def ak_rank(array, axis=1):
    return ak.argsort(ak.argsort(array, axis=axis), axis=axis)


def ak_binned(array, bins, axis=1, overflow=False):
    lo_edges, hi_edges = bins[:-1], bins[1:]

    if overflow:
        centers = (hi_edges+lo_edges)/2
        array = np.clip(array, centers[0], centers[-1])

    binned = [(array >= lo) & (array < hi)
              for lo, hi in zip(lo_edges, hi_edges)]

    return 1*ak_stack(binned, axis=axis)


def build_p4(array, prefix=None, use_regressed=False, extra=[]):
    kin = ['pt', 'eta', 'phi', 'm']+extra
    regmap = {}
    if use_regressed:
        regmap = {'pt': 'ptRegressed', 'm': 'mRegressed'}

    if prefix:
        def get_var(var): return f'{prefix}_{var}'
    else:
        def get_var(var): return var

    return ak.zip(
        {
            var: array[get_var(regmap.get(var, var))]
            for var in kin
        }, with_name='Momentum4D'
    )

def sum_p4(p4s):
    sum = p4s[0]
    for p4 in p4s[1:]:
        sum += p4
    return sum

def _chunks_(total_events, batches):
    k, m = divmod(total_events, batches)
    for i in range(batches):
        yield i*k+min(i, m), (i+1)*k+min(i+1, m)


def chunk_method(array: ak.Array, array_method: Callable, batches=25, events=None, report=False) -> ak.Array:
    """Apply a method on an array in chunks. The final result is then concatenated together.

    Args:
        array (ak.Array): Awkward Array like structure
        array_method (Callable): Method that takes an array as the first argument and returns an array
        batches (int, optional): Number of batches to chunk the array into. Defaults to None.
        events (int, optional): Approximate number of events per chunk. Defaults to None.
        report (bool, optional): Gives TQDM reporting. Defaults to True.

    Returns:
        ak.Array: Awkward Array that is a concatenation of the results of the array_method
    """
    total_events = len(array)
    if batches:
        batches = batches
    if events:
        batches = total_events//events

    it = enumerate(_chunks_(total_events, batches))
    if report:
        it = tqdm(it, total=batches)

    builder = [array_method(array[start:stop]) for i, (start, stop) in it]
    return ak.concatenate(builder)
