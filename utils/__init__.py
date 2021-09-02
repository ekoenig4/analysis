import os
import sys
import git

GIT_WD = git.Repo('.', search_parent_directories=True).working_tree_dir

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
from .varConfig import varinfo

def init_attr(attr,init,size):
    if attr is None: return [init]*size
    attr = list(attr)
    return (attr + size*[init])[:size]

from .selectUtils import *
from .plotUtils import *
from .studyUtils import *
from .classUtils import *
from .orderUtils import *
from .testUtils import *
