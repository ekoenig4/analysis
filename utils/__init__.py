import uproot as ut
import awkward as ak
import numpy as np
import sympy as sp
import string
import re
import vector
from tqdm import tqdm


from .xsecUtils import *
from . import fileUtils as fc
from .cutConfig import *

def init_atr(atr,init,size):
    if atr is None: return [init]*size
    atr = list(atr)
    return (atr + size*[init])[:size]

from .selectUtils import *
from .plotUtils import *
from .studyUtils import *
from .classUtils import *
from .orderUtils import *
from .testUtils import *
