from matplotlib.pyplot import isinteractive
from .binning_tools import get_bin_centers, get_bin_widths, safe_divide
from ..ak_tools import *
from ..classUtils import ObjIter, AttrArray
from . import function
import numpy as np
import awkward as ak
import numbers
import re
from scipy import stats as f_stats

class Stats:
    def __init__(self,graph):
        self.x_mean, self.x_std = np.mean(graph.x_array), np.std(graph.x_array)
        self.x_min, self.x_max = np.min(graph.x_array), np.max(graph.x_array)
        self.y_mean, self.y_std = np.mean(graph.y_array), np.std(graph.y_array)
        self.y_min, self.y_max = np.min(graph.y_array), np.max(graph.y_array)
        self.x_sum,  self.y_sum = np.sum(graph.x_array), np.sum(graph.y_array)
        self.area = np.trapz(graph.y_array, graph.x_array)
        self.ndf = len(graph.y_array)
    def __format__(self, spec):
        return self.__str__(spec=spec)
    def __str__(self, spec='0.3e'):
        return '\n'.join([ f'{key}={float(value):{spec}}' for key,value in vars(self).items() ])

class Graph:

    @classmethod
    def from_th1d(cls, th1d, scale=1, **kwargs):
        bins = th1d.axis().edges()
        x, xerr = get_bin_centers(bins), get_bin_widths(bins)
        y, yerr = scale*th1d.counts(), scale*th1d.errors()

        return cls(x, y, xerr=xerr, yerr=yerr, **kwargs)

    @classmethod 
    def from_histo(cls, histo, **kwargs):
        y, yerr = histo.histo, histo.error
        x, xerr = get_bin_centers(histo.bins), get_bin_widths(histo.bins)

        return cls(x, y, xerr=xerr, yerr=yerr)

    def __init__(self, x_array, y_array, weights=None, xerr=None, yerr=None, order='x', ndata=None, label_stat=None, inv=False, smooth=False, fit=None, histtype=None, **kwargs):
        if inv: x_array, y_array = y_array, x_array
        x_array = flatten(x_array)
        y_array = flatten(y_array)

        if x_array.dtype.type is np.str_:
            self.xlabel = list(x_array)
            x_array = np.arange(len(self.xlabel))

        mask_inf = np.isinf(x_array) | np.isinf(y_array) | np.isnan(x_array) | np.isnan(y_array)
        x_array, y_array = x_array[~mask_inf], y_array[~mask_inf]
        if xerr is not None: 
            xerr = xerr[~mask_inf]
        if yerr is not None: 
            yerr = yerr[~mask_inf]
        if weights is not None:
            weights = weights[~mask_inf]

        order = x_array.argsort() if order == 'x' else y_array.argsort()
        
        self.x_array = x_array[order]
        self.y_array = y_array[order]
        self.xerr = xerr[order] if xerr is not None else xerr
        self.yerr = yerr[order] if yerr is not None else yerr
        self.weights = weights[order] if weights is not None else weights
        self.ndata = ndata

        if self.ndata is None:
            self.ndata = self.x_array.shape[0] if weights is None else 1/np.sum(weights**2)

        if smooth: self.smooth(smooth)

        self.kwargs = kwargs
        self.kwargs['marker'] = kwargs.get('marker', 'o' if self.x_array.shape[0] < 30 else None)
        self.kwargs['linewidth'] = kwargs.get('linewidth', 2)

        self.stats = Stats(self)    

        self.fit = vars(function).get(fit, None)
        if self.fit is not None:
            self.fit = self.fit.fit_graph(self)

        self.set_label(label_stat)

    def set_attrs(self, label_stat=None, **kwargs):
        kwargs = dict(self.kwargs, **kwargs)
        self.label = kwargs.get('label',None)

        self.kwargs = kwargs
        self.set_label(label_stat)
        return self

    def smooth(self, kernel):
        if isinstance(kernel, bool): kernel = 5
        if isinstance(kernel, int): kernel = np.ones(kernel, dtype=float)
        if isinstance(kernel, list): kernel = np.array(kernel, dtype=float)
        kernel /= kernel.sum()

        size = kernel.shape[0]
        y_array = np.pad(self.y_array, size//2, mode='edge')
        yerr = np.pad(self.yerr,size//2, mode='edge') if self.yerr is not None else None

        self.y_array = np.convolve(y_array, kernel, mode='valid')
        if yerr is not None:
            self.yerr = np.sqrt(np.convolve(yerr**2, kernel, mode='valid'))

        if size%2 == 0:
            self.y_array = get_bin_centers(self.y_array)
            if yerr is not None:
                self.yerr = np.sqrt(get_bin_centers(self.yerr**2)/2)

    def interpolate(self, xlim=None):
        if xlim is None:
            xlim = np.min(self.x_array), np.max(self.x_array)
        x_lo, x_hi = xlim

        x = np.linspace(x_lo, x_hi, 100)
        y = np.interp(x, self.x_array, self.y_array)
        yerr = self.yerr
        if yerr is not None:
            yerr = np.abs(np.interp(x, self.x_array, self.y_array+self.yerr) - y)
        
        self.x_array, self.xerr = x, None
        self.y_array, self.yerr = y, yerr

    def evaluate(self, x):
        return np.interp(x, self.x_array, self.y_array)
    
    def set_label(self, label_stat='area'):
        if label_stat is None: pass
        elif callable(label_stat):
            label_stat = label_stat(self)
        elif any(re.findall(r'{(.*?)}', label_stat)):
            label_stat = label_stat.format(**vars(self))
        elif label_stat == 'area':
            label_stat = f'$A={self.stats.area:0.2f}$'
        elif label_stat.endswith('_mean_std'):
            z = label_stat.split('_')[0]
            mean = getattr(self.stats,z+'_mean')
            stdv = getattr(self.stats,z+'_std')
            label_stat = f'$\mu_{z}={mean:0.2f} \pm {stdv:0.2f}$'
        elif label_stat.endswith('_std'):
            z = label_stat.split('_')[0]
            label_stat = f'$\sigma_{z}={getattr(self.stats,label_stat):0.2f}$'
        elif label_stat.endswith('_mean'):
            z = label_stat.split('_')[0]
            label_stat = f'$\mu_{z}={getattr(self.stats,label_stat):0.2f}$'
        elif label_stat.endswith('_sum'):
            z = label_stat.split('_')[0]
            label_stat = f'$\Sigma_{z}={getattr(self.stats,label_stat):0.2f}$'
        elif label_stat == 'chi2':
            self.stats.chi2 = self.y_array.sum()
            self.stats.ndf = self.y_array.shape[0] - 1
            self.stats.chi2_pvalue = f_stats.chi2.sf(self.stats.chi2, self.stats.ndf)


            label_stat = f'$\chi^2/ndf$={self.stats.chi2/self.stats.ndf:0.2} ({self.stats.chi2_pvalue:0.2})'
        elif label_stat == 'ks':
            self.stats.ks = np.abs(self.y_array).max()
            self.stats.ks_pvalue = f_stats.kstwobign.sf(self.stats.ks)

            label_stat = f'KS={self.stats.ks:0.2} ({self.stats.ks_pvalue:0.2})'
        else: label_stat = f'{getattr(self.stats,label_stat):0.2f}'
        
        if label_stat is not None:
            if 'label' in self.kwargs:
                label_stat = f'{self.kwargs["label"]} ({label_stat})'
            self.kwargs['label'] = f'{label_stat}'
        
    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            return Multiply(self, other)
        y_array = other*self.y_array
        yerr = other*self.yerr if self.yerr is not None else self.yerr
        return Graph(self.x_array, y_array, weights=self.weights, xerr=self.xerr, yerr=yerr, **self.kwargs)
    def __truediv__(self, other):
        if not isinstance(other, numbers.Number):
            return Ratio(self, other)
        y_array = self.y_array/other
        yerr = self.yerr/other if self.yerr is not None else self.yerr
        return Graph(self.x_array, y_array, weights=self.weights, xerr=self.xerr, yerr=yerr, **self.kwargs)
    def __add__(self, other):
        if not isinstance(other, numbers.Number):
            return Summation(self, other)
        y_array = self.y_array + other
        return Graph(self.x_array, y_array, weights=self.weights, xerr=self.xerr, yerr=self.yerr, **self.kwargs)
    def __sub__(self, other):
        if not isinstance(other, numbers.Number):
            return Difference(self, other)
        y_array = self.y_array - other
        return Graph(self.x_array, y_array, weights=self.weights, xerr=self.xerr, yerr=self.yerr, **self.kwargs)
    def __pow__(self, pow):
        y_array = self.y_array ** pow
        yerr = y_array*(pow*self.yerr/self.y_array) if self.yerr is not None else self.yerr
        return Graph(self.x_array, y_array, weights=self.weights, xerr=self.xerr, yerr=yerr, **self.kwargs)
    def __neg__(self):
        y_array = -self.y_array
        return Graph(self.x_array, y_array, weights=self.weights, xerr=self.xerr, yerr=self.yerr, **self.kwargs)

class GraphList(ObjIter):
    def __init__(self, x_arrays, y_arrays,  **kwargs):
        x_arrays = np.array(x_arrays)
        y_arrays = np.array(y_arrays)

        if x_arrays.ndim == 1:
            x_arrays = np.array([x_arrays]*y_arrays.shape[0])
        elif y_arrays.ndim == 1:
            y_arrays = np.array([y_arrays]*x_arrays.shape[0])
        narrays = len(y_arrays)
        
        for key,value in kwargs.items(): 
            if not isinstance(value,list): value = AttrArray.init_attr(None,value,narrays)
            kwargs[key] = AttrArray.init_attr(value,None,narrays)
            
        super().__init__([
            Graph(x_array,y_array, **{ key:value[i] for key,value in kwargs.items() })
            for i,(x_array,y_array) in enumerate(zip(x_arrays,y_arrays))
        ])

class GraphFromHisto(Graph):
    def __init__(self, histo, **kwargs):
        y, yerr = histo.histo, histo.error
        x, xerr = get_bin_centers(histo.bins), get_bin_widths(histo.bins)

        super().__init__(x, y, xerr=xerr, yerr=yerr)
    
def _to_graph(obj):
    if hasattr(obj, 'histo'): return GraphFromHisto(obj)
    return obj

def get_data(obj):
    y, yerr = obj.y_array, obj.yerr
    x, xerr = obj.x_array, obj.xerr

    if xerr is None: xerr = np.zeros_like(x)
    if yerr is None: yerr = np.zeros_like(y)

    return x, y, xerr, yerr

def _get_kwargs(num, den, **kwargs):
    kwargs['color'] = kwargs.get('color',den.kwargs.get('color',None))
    kwargs['linestyle'] = kwargs.get('linestyle',den.kwargs.get('linestyle',None))

    return kwargs

        
class Multiply(Graph):
    def __init__(self, num, den, inv=False, method=None, num_transform=None, den_transform=None, **kwargs):
        kwargs = _get_kwargs(num, den, **kwargs)

        if inv: num, den = den, num
        num, den = _to_graph(num), _to_graph(den)

        try:
            np.testing.assert_allclose(num.x_array, den.x_array)
        except AssertionError:
            x_lo = max(np.min(num.x_array), np.min(den.x_array))
            x_hi = min(np.max(num.x_array), np.max(den.x_array))
            num.interpolate((x_lo,x_hi))
            den.interpolate((x_lo,x_hi))

        num_x, num_y, num_xerr, num_yerr = get_data(num)
        den_x, den_y, den_xerr, den_yerr = get_data(den)

        if callable(num_transform): num_y = num_transform(num_y)
        if callable(den_transform): den_y = den_transform(den_y)

        ratio = num_y * den_y
        error = ratio * np.sqrt( safe_divide(num_yerr, num_y, np.nan)**2 + safe_divide(den_yerr, den_y, np.nan)**2 )

        super().__init__(num_x, ratio, xerr=num_xerr, yerr=error, **kwargs)


class Ratio(Graph):
    def __init__(self, num, den, inv=False, method=None, num_transform=None, den_transform=None, **kwargs):
        kwargs = _get_kwargs(num, den, **kwargs)

        if inv: num, den = den, num
        num, den = _to_graph(num), _to_graph(den)

        try:
            np.testing.assert_allclose(num.x_array, den.x_array)
        except AssertionError:
            x_lo = max(np.min(num.x_array), np.min(den.x_array))
            x_hi = min(np.max(num.x_array), np.max(den.x_array))
            num.interpolate((x_lo,x_hi))
            den.interpolate((x_lo,x_hi))

        num_x, num_y, num_xerr, num_yerr = get_data(num)
        den_x, den_y, den_xerr, den_yerr = get_data(den)

        if callable(num_transform): num_y = num_transform(num_y)
        if callable(den_transform): den_y = den_transform(den_y)

        
        if method == 'sumd':
            den_y = num_y + den_y
            den_yerr = np.sqrt(num_yerr**2 + den_yerr**2)

        if method == 'rootd':
            den_y = np.sqrt(den_y)
            den_yerr = 0.5*(den_yerr/den_y)

        ratio = safe_divide(num_y, den_y, np.nan)
        error = ratio * np.sqrt( safe_divide(num_yerr, num_y, np.nan)**2 + safe_divide(den_yerr, den_y, np.nan)**2 )


            # r_mean = np.sum(num_y)/np.sqrt( np.sum(den_y) )
            # ratio /= r_mean
            # error /= r_mean

        if method == 'g-test':
            error = np.abs(error/ratio)
            ratio = np.log(ratio)
            error = np.sqrt( (num_yerr/num_y)**2 + (error/ratio)**2 )
            ratio = 2*num_y*ratio
            error = ratio*error

        super().__init__(num_x, ratio, xerr=num_xerr, yerr=error, **kwargs)

class Difference(Graph):
    def __init__(self, num, den, inv=False, method=None, **kwargs):
        kwargs = _get_kwargs(num, den, **kwargs)

        if inv: num, den = den, num
        num, den = _to_graph(num), _to_graph(den)

        try:
            np.testing.assert_allclose(num.x_array, den.x_array)
        except AssertionError:
            x_lo = max(np.min(num.x_array), np.min(den.x_array))
            x_hi = min(np.max(num.x_array), np.max(den.x_array))
            num.interpolate((x_lo,x_hi))
            den.interpolate((x_lo,x_hi))

        num_x, num_y, num_xerr, num_yerr = get_data(num)
        den_x, den_y, den_xerr, den_yerr = get_data(den)

        difference = num_y - den_y
        error = np.sqrt( num_yerr**2 + den_yerr**2 )

        if method == 'percerr':
            difference = 1 - den_y/num_y
            error = difference*np.sqrt( (num_yerr/num_y)**2 + (den_yerr/den_y)**2 )

        if method == 'normalize':
            mean = (num_y+den_y)/2
            difference = difference/mean
            error = error/mean

        if method == 'standardize': 
            std = np.std(difference)
            difference = difference/std
            error = error/std
        
        if method == 'stderr': 
            difference = difference/error
            error = error/error

        if method == 'chi2':
            error = 2*difference*error
            difference = difference**2

            error = np.sqrt( (error/difference)**2 + (den_yerr/den_y)**2 )
            difference = difference/den_y
            error = difference*error

        if method == 'r2':
            std = np.std(num_y)
            error = 2*difference*error
            difference = difference**2/std
            error = error/std

        if method == 'ks':
            nm = np.sqrt((num.ndata*den.ndata)/(num.ndata+den.ndata))
            difference = nm*difference
            error = nm*error

        super().__init__(num_x, difference, xerr=num_xerr, yerr=error, **kwargs)

class Summation(Graph):
    def __init__(self, num, den, inv=False, method=None, **kwargs):
        kwargs = _get_kwargs(num, den, **kwargs)

        if inv: num, den = den, num
        num, den = _to_graph(num), _to_graph(den)

        try:
            np.testing.assert_allclose(num.x_array, den.x_array)
        except AssertionError:
            x_lo = max(np.min(num.x_array), np.min(den.x_array))
            x_hi = min(np.max(num.x_array), np.max(den.x_array))
            num.interpolate((x_lo,x_hi))
            den.interpolate((x_lo,x_hi))

        num_x, num_y, num_xerr, num_yerr = get_data(num)
        den_x, den_y, den_xerr, den_yerr = get_data(den)

        difference = num_y + den_y
        error = np.sqrt( num_yerr**2 + den_yerr**2 )

        super().__init__(num_x,difference, xerr=num_xerr, yerr=error, **kwargs)

class Correlation(Graph):
    def __init__(self, num, den, inv=False, method=None, **kwargs):
        kwargs = _get_kwargs(num, den, **kwargs)

        if inv: num, den = den, num
        num, den = _to_graph(num), _to_graph(den)

        try:
            np.testing.assert_allclose(num.x_array, den.x_array)
        except AssertionError:
            x_lo = max(np.min(num.x_array), np.min(den.x_array))
            x_hi = min(np.max(num.x_array), np.max(den.x_array))
            num.interpolate((x_lo,x_hi))
            den.interpolate((x_lo,x_hi))

        num_x, num_y, num_xerr, num_yerr = get_data(num)
        den_x, den_y, den_xerr, den_yerr = get_data(den)

        x, y = num_y, den_y

        if method == 'roc':
            x = np.sort(num_y)
            y = np.sort(den_y)

            x = np.pad(num_y, 1)
            y = np.pad(den_y, 1)
            x[-1] = y[-1] = 1

            area = np.abs(np.trapz(y, x))
            if area < 0.5:
                x = 1 - x
                y = 1 - y
            

        if method == 'ad':
            x = (num.ndata*num_y + den.ndata*den_y)/(num.ndata + den.ndata)
            y = num_y - den_y
            # nm = (num.ndata*den.ndata)/(num.ndata + den.ndata)
            y = y**2 / (x * (1 - x))

        
        super().__init__(x , y, **kwargs)

