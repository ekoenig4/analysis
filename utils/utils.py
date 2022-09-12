import awkward as ak
from matplotlib.pyplot import isinteractive
import torch
import git
import numpy as np
import re
from typing import Callable
import itertools
import vector
from tqdm import tqdm
vector.register_awkward()

GIT_WD = git.Repo('.', search_parent_directories=True).working_tree_dir


def _flatten(array):
    if isinstance(array, ak.Array):
        array = ak.flatten(array, axis=None)
        return ak.to_numpy(array)
    if isinstance(array, torch.Tensor):
        array = torch.flatten(array)
        return array.cpu().numpy()
    if isinstance(array, list):
        array = np.array(array)
        return array.reshape(-1).tolist()
    return np.array(array).reshape(-1)


def flatten(array, clean=True):
    array = _flatten(array)
    # mask = ~( np.isnan(array) | np.isinf( np.abs(array) ) )
    # return array[mask]
    return array


def ordinal(n): return "%d%s" % (
    n, {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th"))


def array_min(array, value): return ak.min(ak.concatenate(
    ak.broadcast_arrays(value, array[:, np.newaxis]), axis=-1), axis=-1)


def is_iter(array):
    try:
        it = iter(array)
    except TypeError:
        return False
    return True


def loop_iter(iterable):
    return itertools.cycle(iterable)


def get_batch_ranges(total, batch_size):
    batch_ranges = np.arange(0, total, batch_size)

    if total - batch_ranges[-1] > 0.5*batch_size:
        batch_ranges = np.append(batch_ranges, total)
    else:
        batch_ranges[-1] = total
    return batch_ranges


def init_attr(attr, init, size):
    if attr is None:
        return [init]*size
    if not isinstance(attr, list):
        attr = [attr]
    return (attr + size*[init])[:size]


def copy_fields(obj, copy):
    for key, value in vars(obj).items():
        setattr(copy, key, value)


def get_bin_centers(bins):
    return np.array([(lo+hi)/2 for lo, hi in zip(bins[:-1], bins[1:])])


def get_bin_widths(bins):
    return np.array([(hi-lo)/2 for lo, hi in zip(bins[:-1], bins[1:])])


def get_bin_line(bins):
    return np.linspace(bins[0], bins[-1], len(bins) - 1)


def safe_divide(a, b, default=None):
    a, b = np.array(a), np.array(b)
    tmp = np.full_like(a, default, dtype=float)
    np.divide(a, b, out=tmp, where=(b != 0))
    return tmp


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s < m]


def restrict_array(array, bins, **params):
    x_lo = array >= bins[0]
    x_hi = array < bins[-1]
    array = array[x_lo & x_hi]
    if len(params) > 0:
        params = [param[x_lo & x_hi] for param in params.values()]
        return array, *params
    return array


def _autobin_(data, nstd=3, nbins=30):
    ndata = ak.size(data)
    mean = ak.mean(data)
    stdv = ak.std(data)
    minim, maxim = ak.min(data), ak.max(data)
    is_int = issubclass(data.dtype.type, np.integer)

    if is_int:
        xlo, xhi, nbins = min(minim, 0), maxim+1, maxim-minim
    else:
        xlo, xhi = max([minim, mean-nstd*stdv]), min([maxim, mean+nstd*stdv])
    return xlo, xhi, nbins, int(is_int)


def autobin(data, bins=None, nstd=3, nbins=30):
    if isinstance(bins, tuple):
        return np.linspace(*bins)
    if isinstance(bins, list):
        return np.array(bins)
    if bins is not None:
        return bins

    if type(data) == list:
        datalist = list(data)
        databins = np.array([_autobin_(data, nstd, nbins)
                            for data in datalist])
        xlo = np.nanmin(databins[:, 0])
        xhi = np.nanmax(databins[:, 1])
        nbins = int(np.min(databins[:, 2]))
        is_int = databins[:, 3][0]
    else:
        xlo, xhi, nbins, is_int = _autobin_(data, nstd, nbins)
    if is_int == 1:
        return np.arange(xlo, xhi+1)
    return np.linspace(xlo, xhi, nbins)


def unzip_records(records):
    return {field: array for field, array in zip(records.fields, ak.unzip(records))}


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


def build_collection(tree, pattern, name, ptordered=False, replace=False):
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
        order = ak.argsort(-collection[f'{name}_pt'], axis=-1)
        collection = {key: array[order] for key, array in collection.items()}

    tree.extend(**collection)
    # return collection


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
        std = np.sqrt(np.average((array-avg)**2, weights=weights))
    return avg, std


def build_p4(array, prefix=None, use_regressed=False):
    kin = ['pt', 'eta', 'phi', 'm']
    regmap = {}
    if use_regressed:
        regmap = {'pt': 'ptRegressed', 'm': 'mRegressed'}

    if prefix:
        get_var = lambda var : f'{prefix}_{var}'
    else:
        get_var = lambda var : var

    return ak.zip(
        {
            var: array[get_var(regmap.get(var,var))]
            for var in kin
        }, with_name='Momentum4D'
    )

def sum_p4(p4s):
    sum = p4s[0]
    for p4 in p4s[1:]:
        sum += p4
    return sum


def p4_to_awk(p4):
    return ak.zip({kin: getattr(p4, kin) for kin in ('pt', 'm', 'eta', 'phi')})


def ak_stack(arrays, axis=1):
    reshape = tuple([slice(None) for _ in range(axis)] + [None])

    return ak.concatenate(
        [array[reshape] for array in arrays],
        axis=axis
    )



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