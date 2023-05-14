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
import scipy.interpolate as interp

class Stats:
    def __init__(self,graph):
        self.x_mean, self.x_std = np.mean(graph.x_array), np.std(graph.x_array)
        self.y_mean, self.y_std = np.mean(graph.y_array), np.std(graph.y_array)
        self.z_mean, self.z_std = np.mean(graph.z_array), np.std(graph.z_array)
        self.x_sum,  self.y_sum, self.z_sum = np.sum(graph.x_array), np.sum(graph.y_array), np.sum(graph.z_array)
        self.ndf = len(graph.y_array)
    def __str__(self):
        return '\n'.join([ f'{key}={float(value):0.3e}' for key,value in vars(self).items() ])

class Graph2D:
    def __init__(self, x_array, y_array, z_array, xerr=None, yerr=None, zerr=None, order='x', ndata=None, label_stat=None, inv=False, smooth=False, fit=None, histtype=None, **kwargs):
        if inv: x_array, y_array = y_array, x_array
        x_array = flatten(x_array)
        y_array = flatten(y_array)
        z_array = flatten(z_array)
        
        if x_array.dtype.type is np.str_:
            self.xlabel = list(x_array)
            x_array = np.arange(len(self.xlabel))

        mask_inf = np.isinf(x_array) | np.isinf(y_array) | np.isinf(z_array) | \
                   np.isnan(x_array) | np.isnan(y_array) | np.isnan(z_array)

        x_array, y_array, z_array = x_array[~mask_inf], y_array[~mask_inf], z_array[~mask_inf]

        if xerr is not None: 
            xerr = flatten(xerr)[~mask_inf]
        if yerr is not None: 
            yerr = flatten(yerr)[~mask_inf]
        if zerr is not None:
            zerr = flatten(zerr)[~mask_inf]

        order = x_array.argsort() if order == 'x' else y_array.argsort()

        self.x_array = x_array[order]
        self.y_array = y_array[order]
        self.z_array = z_array[order]

        self.xerr = xerr[order] if xerr is not None else xerr
        self.yerr = yerr[order] if yerr is not None else yerr
        self.zerr = zerr[order] if zerr is not None else zerr
        self.ndata = ndata
        
        if self.ndata is None:
            self.ndata = self.x_array.shape[0]

        if smooth: self.smooth(smooth)
        
        self.kwargs = kwargs
        self.stats = Stats(self)  
        self.set_label(label_stat)

    def smooth(self, kind='linear'):
        if kind is True: kind = 'linear'
        X, Y, Z = self.x_array, self.y_array, self.z_array

        interp_kind = dict(
            linear=interp.LinearNDInterpolator,
            clough=interp.CloughTocher2DInterpolator,
        ).get(kind, interp.LinearNDInterpolator)

        f = interp_kind(np.array([X,Y]).T, Z)
        nx = np.linspace(np.min(self.x_array), np.max(self.x_array), 100)
        ny = np.linspace(np.min(self.y_array), np.max(self.y_array), 100)
        nx, ny = np.meshgrid(nx, ny)
        nz = f(nx, ny)

        return Graph2D(nx, ny, nz, **self.kwargs)
    
    def evaluate(self, x, y, nan=None):
        import scipy.interpolate as interp
        linear=interp.LinearNDInterpolator
        f = linear(np.array([self.x_array, self.y_array]).T, self.z_array)

        x, y = flatten(x), flatten(y)
        z = f(x, y)
        
        if nan is not None:
            z[np.isnan(z)] = nan
        return z

    
    def set_attrs(self, label_stat=None, **kwargs):
        kwargs = dict(self.kwargs, **kwargs)
        self.label = kwargs.get('label',None)

        self.kwargs = kwargs
        self.set_label(label_stat)
        return self

    def set_label(self, label_stat='area'):
        if label_stat is None: pass
        elif callable(label_stat):
            label_stat = label_stat(self)
        elif any(re.findall(r'{(.*?)}', label_stat)):
            label_stat = label_stat.format(**vars(self))
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
        
        if label_stat is not None:
            if 'label' in self.kwargs:
                label_stat = f'{self.kwargs["label"]} ({label_stat})'
            self.kwargs['label'] = f'{label_stat}'

class Graph2DList(ObjIter):
    def __init__(self, x_arrays, y_arrays, z_arrays, **kwargs):
        x_arrays = np.array(x_arrays)
        y_arrays = np.array(y_arrays)
        z_arrays = np.array(z_arrays)

        narrays = len(y_arrays)
        
        for key,value in kwargs.items(): 
            if not isinstance(value,list): value = AttrArray.init_attr(None,value,narrays)
            kwargs[key] = AttrArray.init_attr(value,None,narrays)
          
        super().__init__([
            Graph2D(x_array,y_array,z_array, **{ key:value[i] for key,value in kwargs.items() })
            for i,(x_array,y_array,z_array) in enumerate(zip(x_arrays,y_arrays,z_arrays))
    ])
        

class Graph2DFromHisto2D(Graph2D):
    def __init__(self, histo, **kwargs):
        z, zerr = histo.histo2d, histo.error2d
        x, xerr = get_bin_centers(histo.x_bins), get_bin_widths(histo.x_bins)
        y, yerr = get_bin_centers(histo.y_bins), get_bin_widths(histo.y_bins)

        X, Y = np.meshgrid(x, y)
        Xerr, Yerr = np.meshgrid(xerr, yerr)

        super().__init__(X, Y, z, xerr=Xerr, yerr=Yerr, zerr=zerr)
        
def _to_graph(obj):
    if hasattr(obj, 'histo2d'): return Graph2DFromHisto2D(obj)
    return obj

def _get_kwargs(num, den, **kwargs):
    kwargs['color'] = kwargs.get('color',den.kwargs.get('color',None))
    kwargs['linestyle'] = kwargs.get('linestyle',den.kwargs.get('linestyle',None))

    return kwargs

def get_data(obj):
    y, yerr = obj.y_array, obj.yerr
    x, xerr = obj.x_array, obj.xerr
    z, zerr = obj.z_array, obj.zerr

    if xerr is None: xerr = np.zeros_like(x)
    if yerr is None: yerr = np.zeros_like(y)
    if zerr is None: zerr = np.zeros_like(z)

    return x, y, z, xerr, yerr, zerr

class Ratio2D(Graph2D):
    def __init__(self, num, den, inv=False, method=None, num_transform=None, den_transform=None, **kwargs):
        kwargs = _get_kwargs(num, den, **kwargs)

        if inv: num, den = den, num
        num, den = _to_graph(num), _to_graph(den)

        try:
            np.testing.assert_allclose(num.x_array, den.x_array)
            np.testing.assert_allclose(num.y_array, den.y_array)
        except AssertionError as e:
            raise e
       
        num_x, num_y, num_z, num_xerr, num_yerr, num_zerr = get_data(num)
        den_x, den_y, den_z, den_xerr, den_yerr, den_zerr = get_data(den)

        ratio = safe_divide(num_z, den_z, np.nan)
        error = ratio * np.sqrt( safe_divide(num_zerr, num_z, np.nan)**2 + safe_divide(den_zerr, den_z, np.nan)**2 ) 

        super().__init__(num_x, num_y, ratio, xerr=num_xerr, yerr=num_yerr, zerr=error, **kwargs)