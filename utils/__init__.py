from .plotUtils import *
from .testUtils import *
from .orderUtils import *
from .classUtils import *
from .studyUtils import *
from .selectUtils import *
from .varConfig import varinfo
from .cutConfig import *
from . import fileUtils as fc
from .xsecUtils import *
from tqdm import tqdm
import vector
import re
import string
import sympy as sp
import numpy as np
import awkward as ak
import uproot as ut
import os
import sys
import git

GIT_WD = git.Repo('.', search_parent_directories=True).working_tree_dir


def init_attr(attr, init, size):
    if attr is None:
        return [init]*size
    attr = list(attr)
    return (attr + size*[init])[:size]
