import awkward as ak
import git
import numpy as np
import re

GIT_WD = git.Repo('.', search_parent_directories=True).working_tree_dir


def flatten(array): return ak.to_numpy(ak.flatten(array, axis=None))


def ordinal(n): return "%d%s" % (
    n, {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th"))


def array_min(array, value): return ak.min(ak.concatenate(
    ak.broadcast_arrays(value, array[:, np.newaxis]), axis=-1), axis=-1)


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
    attr = list(attr)
    return (attr + size*[init])[:size]


def copy_fields(obj, copy):
    for key, value in vars(obj).items():
        setattr(copy, key, value)


def get_bin_centers(bins):
    return [(lo+hi)/2 for lo, hi in zip(bins[:-1], bins[1:])]


def get_bin_widths(bins):
    return [(hi-lo)/2 for lo, hi in zip(bins[:-1], bins[1:])]


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


def restrict_array(array, bins, weights=None):
    x_lo = array >= bins[0]
    x_hi = array < bins[-1]
    array = array[x_lo & x_hi]
    if weights is not None:
        weights = weights[x_lo & x_hi]
        return array, weights
    return array


def _autobin_(data, nstd=3):
    ndata = ak.size(data)
    mean = ak.mean(data)
    stdv = ak.std(data)
    minim, maxim = ak.min(data), ak.max(data)
    is_int = issubclass(data.dtype.type, np.integer)

    if is_int:
        xlo, xhi, nbins = 0, maxim+1, maxim-minim
    else:
        xlo, xhi = max([minim, mean-nstd*stdv]), min([maxim, mean+nstd*stdv])
        nbins = 30  # min(int(1+np.sqrt(ndata)), 30)
    return xlo, xhi, nbins, int(is_int)


def autobin(data, nstd=3):
    if type(data) == list:
        datalist = list(data)
        databins = np.array([_autobin_(data, nstd) for data in datalist])
        xlo = np.min(databins[:, 0])
        xhi = np.max(databins[:, 1])
        nbins = int(np.min(databins[:, 2]))
        is_int = databins[:, 3][0]
    else:
        xlo, xhi, nbins, is_int = _autobin_(data, nstd)
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


def get_collection(tree, name, named=True):
    collection_branches = list(
        filter(lambda branch: branch.startswith(name+'_'), tree.fields))

    branches = tree[collection_branches]
    if named:
        return branches

    branches_unzipped = {field.replace(name+'_', ''): array for field,
                         array in zip(branches.fields, ak.unzip(branches))}
    return ak.zip(branches_unzipped, depth_limit=1)


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


def build_collection(tree, pattern, name):
    fields = list(filter(lambda field: re.match(pattern, field), tree.fields))
    components = dict.fromkeys([re.search(pattern, field)[0]
                               for field in fields if re.search(pattern, field)]).keys()
    shared_fields = list(set.intersection(*[set(map(lambda field: field[len(component)+1:], filter(
        lambda field: field.startswith(component), fields))) for component in components]))

    components = [get_collection(tree, component, named=False)[
        shared_fields] for component in components]

    collection = {f'{name}_{field}': ak.concatenate(
        [component[field][:, None] for component in components], axis=-1) for field in shared_fields}
    return collection


def get_avg_std(array, weights=None, bins=None):
    mask = ~np.isnan(array)
    array = ak.flatten(array[mask], axis=None)
    if weights is None:
        if bins is not None:
            array = restrict_array(array, bins)
        avg = ak.mean(array)
        std = ak.std(array)
    else:
        weights = ak.flatten(weights[mask], axis=None)
        if bins is not None:
            array, weights = restrict_array(array, bins, weights)
        avg = np.average(array, weights=weights)
        std = np.sqrt(np.average((array-avg)**2, weights=weights))
    return avg, std
