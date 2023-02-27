from .binning_tools import *
from .histogram import histogram
from ..xsecUtils import lumiMap
from ..classUtils import ObjIter,AttrArray
from . import function
from .graph import Graph
from ..ak_tools import *
from ..utils import is_iter
import numpy as np
import numba
import awkward as ak
import re
import copy

from .histogram import Histo

class Stats:
    def __init__(self,histo):
        self.nevents = np.sum(histo.weights)
        self.ess = np.sum(histo.weights)**2/np.sum(histo.weights**2)

        self.x_mean, self.x_stdv = get_avg_std(histo.x_array, histo.weights, histo.x_bins)
        self.x_minim, self.x_maxim = np.min(histo.x_array), np.max(histo.x_array)
        
        self.y_mean, self.y_stdv = get_avg_std(histo.y_array, histo.weights, histo.y_bins)
        self.y_minim, self.y_maxim = np.min(histo.y_array), np.max(histo.y_array)
        
    def __str__(self):
        return '\n'.join([ f'{key}={float(value):0.3e}' for key,value in vars(self).items() ])

@numba.jit(nopython=True, parallel=True, fastmath=True)
def numba_weighted_histo2d(x_array : np.array, y_array : np.array, x_bins : np.array, y_bins : np.array, weights : np.array) -> np.array:
    counts = np.zeros( (x_bins.shape[0]-1, y_bins.shape[0]-1) )
    for i,(x_lo, x_hi) in enumerate(zip(x_bins[:-1],x_bins[1:])):
        x_mask = (x_array >= x_lo) & (x_array < x_hi)
        for j,(y_lo, y_hi) in enumerate(zip(y_bins[:-1], y_bins[1:])):
            y_mask = (y_array >= y_lo) & (y_array < y_hi)
            counts[i,j] = np.sum(weights[x_mask & y_mask])
    errors = np.sqrt(counts)
    return counts.T, errors.T

@numba.jit(nopython=True, parallel=True, fastmath=True)
def numba_weighted_histo2d_sumw2(x_array : np.array, y_array : np.array, x_bins : np.array, y_bins : np.array, weights : np.array) -> np.array:
    counts = np.zeros( (x_bins.shape[0]-1, y_bins.shape[0]-1) )
    errors = np.zeros( (x_bins.shape[0]-1, y_bins.shape[0]-1) )
    weights2 = weights**2
    for i,(x_lo, x_hi) in enumerate(zip(x_bins[:-1],x_bins[1:])):
        x_mask = (x_array >= x_lo) & (x_array < x_hi)
        for j,(y_lo, y_hi) in enumerate(zip(y_bins[:-1], y_bins[1:])):
            y_mask = (y_array >= y_lo) & (y_array < y_hi)
            counts[i,j] = np.sum(weights[x_mask & y_mask])
            errors[i,j] = np.sum(weights2[x_mask & y_mask])
    errors = np.sqrt(errors)
    return counts.T, errors.T

@numba.jit(nopython=True, parallel=True, fastmath=True)
def numba_unweighted_histo2d(x_array : np.array, y_array : np.array, x_bins : np.array, y_bins : np.array) -> np.array:
    counts = np.zeros( (x_bins.shape[0]-1, y_bins.shape[0]-1) )
    for i,(x_lo, x_hi) in enumerate(zip(x_bins[:-1],x_bins[1:])):
        x_mask = (x_array >= x_lo) & (x_array < x_hi)
        for j,(y_lo, y_hi) in enumerate(zip(y_bins[:-1], y_bins[1:])):
            y_mask = (y_array >= y_lo) & (y_array < y_hi)
            counts[i,j] = np.sum(x_mask & y_mask)
    errors = np.sqrt(counts)
    return counts.T, errors.T


def histogram2d(x_array, y_array, x_bins, y_bins, weights, sumw2=True):
    if weights is None: return numba_unweighted_histo2d(x_array, y_array, x_bins, y_bins)
    elif not sumw2: return numba_weighted_histo2d(x_array, y_array, x_bins, y_bins, weights)
    return numba_weighted_histo2d_sumw2(x_array, y_array, x_bins, y_bins, weights)

class Histo2D:
    def __init__(self, x_array, y_array, x_bins=None, y_bins=None, weights=None,
                 efficiency=False, density=False, cumulative=False, lumi=None, restrict=False,
                 label_stat='events', is_data=False, is_signal=False, is_model=False, sumw2=True, scale=1, __id__=None, fit=None,
                 continous=False, ndata=None, nbins=30, **kwargs):
        self.__id__ = __id__
        if weights is not None: weights = flatten(ak.ones_like(x_array)*weights)

        self.x_array = flatten(x_array)
        self.y_array = flatten(y_array)
        self.counts = len(self.x_array)

        
        if weights is not None:
            self.weights = flatten(weights)
            if self.weights.shape != self.x_array.shape:
                self.weights = flatten(ak.ones_like(x_array)*weights)
        else:
            self.weights= np.ones((self.counts,))

        self.ndata = self.counts if weights is None else weights.sum()

        self.x_bins = autobin(self.x_array, bins=x_bins, nbins=nbins)
        self.y_bins = autobin(self.y_array, bins=y_bins, nbins=nbins)
        self.sumw2 = sumw2

        if restrict:
            x_lo = self.x_array >= self.x_bins[0]
            x_hi = self.x_array < self.x_bins[-1]
            y_lo = self.y_array >= self.y_bins[0]
            y_hi = self.y_array < self.y_bins[-1]
            restrict = x_lo & x_hi & y_lo & y_hi

            self.x_array = self.x_array[restrict]
            self.y_array = self.y_array[restrict]
            self.weights = self.weights[restrict]
            self.counts = len(self.x_array)
            self.ndata = self.weights.sum()
        
        self.is_data = is_data 
        self.is_signal = is_signal 
        self.is_bkg = not (is_data or is_signal)
        self.is_model = is_model
        
        lumi,_ = lumiMap.get(lumi,(lumi,None))
        self.lumi = lumi
        if not is_data: self.weights = lumi * self.weights
    
        self.density = density 
        self.cumulative = cumulative
        self.efficiency = efficiency
        self.continous = continous
        
        self.stats = Stats(self)      
        if scale is 'xs': scale = scale/lumi
        if density or efficiency or cumulative: scale = 1/np.sum(self.weights)
        self.rescale(scale)

        self.label = kwargs.get('label', None)
        self.kwargs = kwargs
        self.kwargs['cmap'] = kwargs.get('cmap', 'YlOrRd' if not self.is_bkg else 'YlGnBu')
        self.set_label(label_stat)
    
    def set_label(self, label_stat='events'):
        if label_stat is None: pass
        elif callable(label_stat):
            label_stat = label_stat(self)
        elif any(re.findall(r'{(.*?)}', label_stat)):
            label_stat = label_stat.format(**vars(self))
        elif label_stat == 'events':
            label_stat = f'{self.stats.nevents:0.2e}'
        elif label_stat == 'mean':
            label_stat = f'$\mu=({self.stats.x_mean:0.2e}, {self.stats.y_mean:0.2e})$'
        else: label_stat = f'{getattr(self.stats,label_stat):0.2e}'

        self.kwargs['label'] = self.label
        if label_stat is not None:
            if self.label is not None:
                self.kwargs['label'] = f'{self.kwargs["label"]} ({label_stat})'
            else:
                self.kwargs['label'] = f'{label_stat}'
    
    def rescale(self,scale):
        if scale is not None:
            if is_iter(scale): 
                scale = flatten(scale)      

            self.weights = scale * self.weights
        
            # if scale != 1 and not is_iter(scale) and isinstance(scale,int):
            #     self.label = f'{self.label} x {scale}'
                
        self.histo2d, self.error2d = histogram2d(self.x_array, self.y_array, self.x_bins, self.y_bins, self.weights, self.sumw2)
        
        if np.any( self.histo2d < 0 ):
            self.error2d = np.where(self.histo2d < 0, 0, self.error2d)
            self.histo2d = np.where(self.histo2d < 0, 0, self.histo2d)

        if self.density:
            self.x_widths = get_bin_widths(self.x_bins)
            self.y_widths = get_bin_widths(self.y_bins)
            wx, wy = np.meshgrid(self.x_widths, self.y_widths)
            self.areas = wx*wy
            self.histo2d /= self.areas
            self.error2d /= self.areas

    def x_corr(self, **kwargs):
        x_points, y_points = get_bin_centers(self.x_bins), get_bin_centers(self.y_bins)
        X_points, Y_points = np.meshgrid(x_points, y_points)

        mean = np.sum(self.histo2d * Y_points, axis=0)/np.sum(self.histo2d,axis=0)
        stdv = np.sqrt(np.sum(self.histo2d * (Y_points - mean)**2, axis=0)/np.sum(self.histo2d,axis=0))

        return Graph(x_points, mean, yerr=stdv, color='grey', **kwargs)

    
    def y_corr(self, **kwargs):
        x_points, y_points = get_bin_centers(self.x_bins), get_bin_centers(self.y_bins)
        X_points, Y_points = np.meshgrid(x_points, y_points)

        mean = np.sum(self.histo2d * X_points, axis=1)/np.sum(self.histo2d,axis=1)
        stdv = np.sqrt(np.sum(self.histo2d * (X_points.T - mean).T**2, axis=1)/np.sum(self.histo2d,axis=1))

        return Graph(mean, y_points, xerr=stdv, order='y', color='grey', **kwargs)

class Histo2DList(ObjIter):
    def __init__(self, x_arrays, y_arrays, x_bins=None, y_bins=None, **kwargs):
        attrs = AttrArray(x_arrays=x_arrays, y_arrays=y_arrays,**kwargs)
        kwargs = attrs[attrs.fields[1:]]
        
        x_multi_binned = isinstance(x_bins, list)
        y_multi_binned = isinstance(y_bins, list)
    
        histo2dlist = []
        for i,(x_array, y_array) in enumerate(zip(x_arrays, y_arrays)):
            _x_bins = x_bins[i] if x_multi_binned else x_bins
            _y_bins = y_bins[i] if y_multi_binned else y_bins

            histo2d = Histo2D(x_array, y_array,x_bins=_x_bins, y_bins=_y_bins, **{ key:value[i] for key,value in kwargs.items() })
            if x_bins is None: x_bins = histo2d.x_bins
            if y_bins is None: y_bins = histo2d.y_bins
            histo2dlist.append(histo2d)
        super().__init__(histo2dlist)

class Data2DList(Histo2DList):
    def __init__(self, x_arrays, y_arrays, x_bins=None, y_bins=None, histtype=None, **kwargs):
        super().__init__(x_arrays, y_arrays, x_bins=x_bins, y_bins=y_bins,**kwargs)

class Stack2D(Histo2D):
    def __init__(self, x_arrays, y_arrays, x_bins=None, y_bins=None, weights=None, label_stat='events', **kwargs):
        if isinstance(label_stat, list): label_stat = label_stat[0]

        x_array = ak.concatenate(x_arrays, axis=0)
        y_array = ak.concatenate(y_arrays, axis=0)
        if isinstance(weights, list): 
            weights = ak.concatenate(weights) if any(weight is not None for weight in weights) else None

        kwargs['color'] = kwargs.get('color','grey')
        kwargs['label'] = kwargs.get('label','MC-Bkg')
        kwargs = { key:value[0] if isinstance(value, list) else value for key, value in kwargs.items() }

        super().__init__(x_array, y_array, x_bins=x_bins, y_bins=y_bins, weights=weights, label_stat=label_stat, **kwargs)
