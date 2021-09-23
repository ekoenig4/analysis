import awkward as ak
import git
import numpy as np

GIT_WD = git.Repo('.', search_parent_directories=True).working_tree_dir


def flatten(array): return ak.to_numpy(ak.flatten(array, axis=None))


def ordinal(n): return "%d%s" % (
    n, {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th"))


def array_min(array, value): return ak.min(ak.concatenate(
    ak.broadcast_arrays(value, array[:, np.newaxis]), axis=-1), axis=-1)


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


def safe_divide(a, b):
    tmp = np.full_like(a, None, dtype=float)
    np.divide(a, b, out=tmp, where=(b != 0))
    return tmp


def autobin(data, nstd=3):
    ndata = ak.size(data)
    mean = ak.mean(data)
    stdv = ak.std(data)
    minim, maxim = ak.min(data), ak.max(data)
    xlo, xhi = max([minim, mean-nstd*stdv]), min([maxim, mean+nstd*stdv])
    nbins = min(int(1+np.sqrt(ndata)), 50)
    return np.linspace(xlo, xhi, nbins)


def unzip_records(records):
    return {field: array for field, array in zip(records.fields, ak.unzip(records))}

def join_fields(awk1, *args, **kwargs):
    args_unzipped = [ unzip_records(awk) for awk in args ]
    new_fields= {}
    for unzipped in args_unzipped:
        new_fields.update(unzipped)
    new_fields.update(kwargs)

    awk1_unzipped= {field: array for field,
                     array in zip(awk1.fields, ak.unzip(awk1))}
    awk1_unzipped.update(**new_fields)
    return ak.zip(awk1_unzipped, depth_limit=1)


def get_collection(tree, name):
    collection_branches = list(
        filter(lambda branch: branch.startswith(name+'_'), tree.fields))
    return tree[collection_branches]
