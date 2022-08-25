from curses import resize_term
from ..utils import flatten, autobin, get_bin_centers, get_bin_widths, is_iter, get_avg_std, init_attr, restrict_array, get_bin_centers
from ..xsecUtils import lumiMap
from ..classUtils import ObjIter,AttrArray
from . import function
from .graph import Graph
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

    histo2d, _, _ = np.histogram2d(x_array, y_array, bins=(x_bins, y_bins), weights=weights)
    error2d, _, _ = np.histogram2d(x_array, y_array, bins=(x_bins, y_bins), weights=weights**2)
    error2d = np.sqrt(error2d)

    return histo2d.T, error2d.T

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
        self.weights = np.ones((self.counts,)) if weights is None else weights
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

        if self.density:
            self.x_widths = get_bin_widths(self.x_bins)
            self.y_widths = get_bin_widths(self.y_bins)
            wx, wy = np.meshgrid(self.x_widths, self.y_widths)
            self.areas = wx*wy
            self.histo2d /= self.areas
            self.error2d /= self.areas

    def x_corr(self, **kwargs):
        def _get_bin_avg_(lo, hi):
            mask = (self.x_array >= lo)&(self.x_array < hi)
            if not np.any(mask): 
                return np.nan, np.nan
            mu, sig = get_avg_std(self.y_array[mask], self.weights[mask], self.y_bins)
            return mu, sig

        corr = np.array([ _get_bin_avg_(lo,hi) for lo,hi in zip(self.x_bins[:-1], self.x_bins[1:]) ])
        return Graph(get_bin_centers(self.x_bins), corr[:,0], yerr=corr[:,1], color='grey', **kwargs)

    
    def y_corr(self, **kwargs):
        def _get_bin_avg_(lo, hi):
            mask = (self.y_array >= lo)&(self.y_array < hi)
            if not np.any(mask):    
                return np.nan, np.nan
            mu, sig = get_avg_std(self.x_array[mask], self.weights[mask], self.x_bins)
            return mu, sig

        corr = np.array([ _get_bin_avg_(lo,hi) for lo,hi in zip(self.y_bins[:-1], self.y_bins[1:]) ])
        return Graph(corr[:,0], get_bin_centers(self.y_bins), xerr=corr[:,1], order='y', color='grey', **kwargs)
