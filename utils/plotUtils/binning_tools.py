import numpy as np
import awkward as ak

def get_bin_centers(bins):
    if bins.dtype.type == np.str_: return get_bin_centers( np.arange(len(bins)) )
    return np.array([(lo+hi)/2 for lo, hi in zip(bins[:-1], bins[1:])])

def get_bin_widths(bins):
    if bins.dtype.type == np.str_: return get_bin_widths( np.arange(len(bins)) )
    return np.array([(hi-lo)/2 for lo, hi in zip(bins[:-1], bins[1:])])

def get_bin_line(bins):
    if bins.dtype.type == np.str_: return get_bin_line( np.arange(len(bins)) )
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


def autobin(data, bins=None, nstd=3, nbins=30, minim=None, maxim=None):
    if isinstance(bins, dict):
        return autobin(data, **bins)
    elif isinstance(bins, tuple):
        return np.linspace(*bins)
    elif isinstance(bins, list):
        return np.array(bins)
    elif bins is not None:
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

    if minim is not None:
        xlo = minim
    if maxim is not None:
        xhi = maxim

    return np.linspace(xlo, xhi, nbins)
