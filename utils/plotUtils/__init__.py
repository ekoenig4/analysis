from .. import *


def get_bin_centers(bins):
    return [ (lo+hi)/2 for lo,hi in zip(bins[:-1],bins[1:]) ]
def get_bin_widths(bins):
    return [ (hi-lo)/2 for lo,hi in zip(bins[:-1],bins[1:]) ]

def safe_divide(a,b):
    tmp = np.full_like(a,None,dtype=float)
    np.divide(a,b,out=tmp,where=(b!=0))
    return tmp

def autobin(data,nstd=3):
    ndata = ak.size(data)
    mean = ak.mean(data)
    stdv = ak.std(data)
    minim,maxim = ak.min(data),ak.max(data)
    xlo,xhi = max([minim,mean-nstd*stdv]),min([maxim,mean+nstd*stdv])
    nbins = min(int(1+np.sqrt(ndata)),50)
    return np.linspace(xlo,xhi,nbins)

from .classes import *
from .plotUtils import *
