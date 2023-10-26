import awkward as ak
from matplotlib.pyplot import isinteractive
import utils.compat.torch as torch
import numpy as np
import itertools

from .ak_tools import *

def ordinal(n): return "%d%s" % (
    n, {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th"))

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

def copy_fields(obj, copy):
    for key, value in vars(obj).items():
        setattr(copy, key, value)

