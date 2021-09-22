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
