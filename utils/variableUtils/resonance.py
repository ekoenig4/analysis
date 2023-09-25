import awkward as ak
import numpy as np
from .variable_tools import cache_variable
    
@cache_variable(bins=(0,300,30))
def hm_chi(t):
    return np.sqrt( ak.sum((t.h_m-125)**2, axis=1) )

@cache_variable(bins=(0,1,30))
def flatten_mxmy(tree):
    mx_bins = np.linspace(0,2000,30)
    my_bins = np.linspace(0,1000,30)

    mx = np.digitize(tree.X_m, mx_bins)
    my1 = np.digitize(tree.Y1_m, my_bins)

    mxmy = len(my_bins)*mx + my1
    mxmy = mxmy/np.max(mxmy)
    return mxmy

@cache_variable(bins=(0,1,30))
def hilbert_mxmy(tree):
    import hilbert 
    
    mx_bins = np.linspace(0,2000,30)
    my_bins = np.linspace(0,1000,30)

    mx = np.digitize(tree.X_m, mx_bins)
    my1 = np.digitize(tree.Y1_m, my_bins)

    mxmy = hilbert.encode(np.stack([mx, my1], axis=1).to_numpy(), 2, 32)
    mxmy = mxmy/(np.max(mxmy)+1)
    return mxmy

@cache_variable(bins=(0,1,30))
def hilbert_mxmy2(tree):
    import hilbert 
    
    mx_bins = np.linspace(0,2000,30)
    my_bins = np.linspace(0,1000,30)

    mx = np.digitize(tree.X_m, mx_bins)
    my1 = np.digitize(tree.Y1_m, my_bins)
    my2 = np.digitize(tree.Y2_m, my_bins)

    mxmy2 = np.stack([mx,my1, my2], axis=1).to_numpy()
    mxmy2 = hilbert.encode(mxmy2, 3, 20)
    mxmy2 = mxmy2/(np.max(mxmy2)+1)
    return mxmy2