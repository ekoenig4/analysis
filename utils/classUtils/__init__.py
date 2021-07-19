from .. import *

def copy_fields(obj,copy):
    for key,value in vars(obj).items():
        setattr(copy,key,value)

from .Tree import Tree
from .Selection import Selection
