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
        self.nevents = np.sum(histo.histo2d)
        # self.ess = np.sum(histo.weights)**2/np.sum(histo.weights**2)
        self.ess = self.nevents**2/np.sum(histo.error2d**2)
        
        if self.nevents > 0:
            if histo.x_array is not None:
                self.x_mean, self.x_stdv = get_avg_std(histo.x_array, histo.weights, histo.x_bins)
                self.x_minim, self.x_maxim = np.min(histo.x_array), np.max(histo.x_array)
                self.y_mean, self.y_stdv = get_avg_std(histo.y_array, histo.weights, histo.y_bins)
                self.y_minim, self.y_maxim = np.min(histo.y_array), np.max(histo.y_array)
            # else:
        
        
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

    @classmethod
    def from_array(cls, x_array, y_array, x_bins=None, y_bins=None, weights=None, sumw2=True, **kwargs):
        if weights is not None:
            if len(weights) != len(x_array):
                raise ValueError(f'shape of the first dimension must be the same for array and weight. Got array ({len(x_array)}) and weight ({len(weights)}')
            weights = cast_array(weights)

        x_array = cast_array(x_array)
        y_array = cast_array(y_array)

        if x_array.ndim > 1:
            nobjs = ak.to_numpy(ak.count(x_array, axis=1))
        x_array = flatten(x_array)
        y_array = flatten(y_array)

        if weights is not None:
            weights = flatten(weights)
            if weights.shape != x_array.shape:
                weights = np.repeat(weights, nobjs)
        else:
            weights= np.ones(x_array.shape)

        x_bins = autobin(x_array, bins=x_bins, nbins=30)
        y_bins = autobin(y_array, bins=y_bins, nbins=30)
        
        raw_counts, _ = histogram2d(x_array, y_array, x_bins, y_bins, np.ones_like(weights))
        counts, error = histogram2d(x_array, y_array, x_bins, y_bins, weights, sumw2=sumw2)

        histo2d = cls(counts, x_bins=x_bins, y_bins=y_bins, error2d=error, raw_counts=raw_counts, x_array=x_array, y_array=y_array, weights=weights, **kwargs)
        return histo2d

    @classmethod
    def from_graph2d(cls, graph2d, **kwargs):
        x, xerr = graph2d.x_array, graph2d.xerr
        y, yerr = graph2d.y_array, graph2d.yerr
        z, zerr = graph2d.z_array, graph2d.zerr

        x_lo = np.unique(x - xerr)
        x_hi = np.unique(x + xerr)
        x_bins = np.concatenate([x_lo[:1], x_hi])
        x_digit = np.digitize(x, x_bins) - 1

        y_lo = np.unique(y - yerr)
        y_hi = np.unique(y + yerr)
        y_bins = np.concatenate([y_lo[:1], y_hi])
        y_digit = np.digitize(y, y_bins) - 1

        Z = np.zeros((len(x_bins)-1, len(y_bins)-1))
        Zerr = np.zeros((len(x_bins)-1, len(y_bins)-1))
        Z[x_digit, y_digit] = z
        Zerr[x_digit, y_digit] = zerr

        return cls(Z.T, x_bins, y_bins, error2d=Zerr.T, **kwargs)

    def __init__(self, counts, x_bins, y_bins, error2d=None, x_array=None, y_array=None, weights=None,
                 efficiency=False, density=False, lumi=None, restrict=False, systematics=None,raw_counts=None,
                 label_stat='events', is_data=False, is_signal=False, is_model=False, scale=1, plot_scale=1, __id__=None, fit=None,
                 continous=False, ndata=None, **kwargs):
        self.__id__ = __id__

        self.histo2d = counts 
        self.x_bins = x_bins
        self.y_bins = y_bins

        self.error2d = error2d
        if error2d is None:
            self.error2d = np.sqrt(counts)
        self.systematics = systematics
        
        self.x_array = x_array 
        self.y_array = y_array 
        self.weights = weights
        self.raw_counts = raw_counts if raw_counts is not None else counts
        
        self.ndata = ndata 
        if ndata is None:
            self.ndata = np.sum(self.histo2d)
        
        self.is_data = is_data 
        self.is_signal = is_signal 
        self.is_bkg = not (is_data or is_signal)
        self.is_model = is_model

        lumi,_ = lumiMap.get(lumi,(lumi,None))
        self.lumi = lumi
        if not self.is_data: 
            self.histo2d = lumi*self.histo2d
            self.error2d = lumi*self.error2d
            if self.weights is not None:
                self.weights = lumi*self.weights
            
        self.continous = continous
        
        self.stats = Stats(self)
        self.rescale(scale, efficiency=efficiency, density=density)

        fit_kwargs = { key[4:]:value for key,value in kwargs.items() if key.startswith('fit_') }
        self.kwargs = { key:value for key,value in kwargs.items() if not key.startswith('fit_') }
        self.fit = vars(function).get(fit, None)
        if self.fit is not None:
            self.fit = self.fit.fit_histo(self, **fit_kwargs)

        self.set_attrs(label_stat=label_stat, plot_scale=plot_scale)

    def set_attrs(self, label_stat=None, plot_scale=1, systematics=None, **kwargs):
        if systematics is not None:
            self.add_systematics(systematics)

        kwargs = dict(self.kwargs, **kwargs)
        
        self.label = kwargs.get('label',None)
        self.plot_scale = plot_scale
        
        if self.plot_scale != 1:
            self.label = f"{self.label}($\\times${self.plot_scale})"

        self.kwargs = kwargs
        self.set_label(label_stat)
        return self
    
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
    
    def rescale(self, scale=None, efficiency=False, density=False):
        if scale is None: scale = 1
        
        self.density = density 
        self.efficiency = efficiency
        self.scale = scale

        if density or efficiency: scale = 1/np.sum(self.histo2d)

        self.histo2d = scale * self.histo2d
        self.error2d = scale * self.error2d
        self.add_systematics()

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
        return self
    
    def add_systematics(self, systematics=None):
        if systematics is None: systematics = self.systematics
        if systematics is None: return
        if not isinstance(systematics, list): systematics = [systematics]

        # for systematic in systematics:
        #     self.error = apply_systematic(self.histo, self.error, systematic)

    def evaluate(self, x, y, nan=None):
        x, y = flatten(x), flatten(y)

        xb = np.digitize(x, self.x_bins)-1
        xb = np.clip(xb, 0, len(self.x_bins)-2)

        yb = np.digitize(y, self.y_bins)-1
        yb = np.clip(yb, 0, len(self.y_bins)-2)


        z = self.histo2d[yb, xb]
        if nan is not None:
            z[np.isnan(z)] = nan
        return z

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

    @classmethod
    def from_arrays(cls, x_arrays, y_arrays, x_bins=None, y_bins=None, **kwargs):
        attrs = AttrArray(x_arrays=x_arrays, y_arrays=y_arrays,**kwargs)
        kwargs = attrs[attrs.fields[1:]]
        
        x_multi_binned = isinstance(x_bins, list)
        y_multi_binned = isinstance(y_bins, list)
    
        histo2dlist = []
        for i,(x_array, y_array) in enumerate(zip(x_arrays, y_arrays)):
            _x_bins = x_bins[i] if x_multi_binned else x_bins
            _y_bins = y_bins[i] if y_multi_binned else y_bins

            histo2d = Histo2D.from_array(x_array, y_array,x_bins=_x_bins, y_bins=_y_bins, **{ key:value[i] for key,value in kwargs.items() })
            if x_bins is None: x_bins = histo2d.x_bins
            if y_bins is None: y_bins = histo2d.y_bins
            histo2dlist.append(histo2d)
        return cls(histo2dlist)

    
    def __repr__(self): return f"Histo2DList<{repr(self.objs)}>"

class Data2DList(Histo2DList):
    ...
    # @classmethod
    # def from_arrays(cls, x_arrays, y_arrays, x_bins=None, y_bins=None, histtype=None, **kwargs):
    #     return Histo2DList.from_arrays(x_arrays, y_arrays, x_bins=x_bins, y_bins=y_bins,**kwargs)

class Stack2D(Histo2D):
    @classmethod
    def from_arrays(cls, x_arrays, y_arrays, x_bins=None, y_bins=None, weights=None, label_stat='events', **kwargs):
        if isinstance(label_stat, list): label_stat = label_stat[0]

        x_array = ak.concatenate(x_arrays, axis=0)
        y_array = ak.concatenate(y_arrays, axis=0)
        if isinstance(weights, list): 
            weights = ak.concatenate(weights) if any(weight is not None for weight in weights) else None

        kwargs['color'] = kwargs.get('color','grey')
        kwargs['label'] = kwargs.get('label','MC-Bkg')
        kwargs = { key:value[0] if isinstance(value, list) else value for key, value in kwargs.items() }

        return cls.from_array(x_array, y_array, x_bins=x_bins, y_bins=y_bins, weights=weights, label_stat=label_stat, **kwargs)
