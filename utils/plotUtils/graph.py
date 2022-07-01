from ..utils import safe_divide, get_bin_centers, get_bin_widths, init_attr
from ..classUtils import ObjIter
from . import function
import numpy as np
import re

class Stats:
    def __init__(self,graph):
        self.x_mean, self.x_std = np.mean(graph.x_array), np.std(graph.x_array)
        self.y_mean, self.y_std = np.mean(graph.y_array), np.std(graph.y_array)
        self.x_sum,  self.y_sum = np.sum(graph.x_array), np.sum(graph.y_array)
        self.area = np.trapz(graph.y_array, graph.x_array)
        self.ndf = len(graph.y_array)
        
    def __str__(self):
        return '\n'.join([ f'{key}={float(value):0.3e}' for key,value in vars(self).items() ])

class Graph:
    def __init__(self, x_array, y_array, label_stat="area", inv=False, xerr=None, yerr=None, fit=None, **kwargs):
        if inv: x_array, y_array = y_array, x_array

        mask_inf = np.isinf(x_array) | np.isinf(y_array)
        x_array, y_array = x_array[~mask_inf], y_array[~mask_inf]
        if xerr is not None: 
            xerr = xerr[~mask_inf]
        if yerr is not None: 
            yerr = yerr[~mask_inf]
        
        order = x_array.argsort()
        
        self.x_array = x_array[order]
        self.y_array = y_array[order]
        self.xerr = xerr[order] if xerr is not None else xerr
        self.yerr = yerr[order] if yerr is not None else yerr

        self.kwargs = kwargs
        self.stats = Stats(self)    

        self.fit = vars(function).get(fit, None)
        if self.fit is not None:
            self.fit = self.fit.fit_graph(self)

        self.set_label(label_stat)
    
    def set_label(self, label_stat='area'):
        if label_stat is None: pass
        elif any(re.findall(r'{(.*?)}', label_stat)):
            label_stat = label_stat.format(**vars(self.stats))
        elif label_stat == 'area':
            label_stat = f'$A={self.stats.area:0.2e}$'
        elif label_stat.endswith('_mean_std'):
            z = label_stat.split('_')[0]
            mean = getattr(self.stats,z+'_mean')
            stdv = getattr(self.stats,z+'_std')
            label_stat = f'$\mu_{z}={mean:0.2e} \pm {stdv:0.2e}$'
        elif label_stat.endswith('_std'):
            z = label_stat.split('_')[0]
            label_stat = f'$\sigma_{z}={getattr(self.stats,label_stat):0.2e}$'
        elif label_stat.endswith('_mean'):
            z = label_stat.split('_')[0]
            label_stat = f'$\mu_{z}={getattr(self.stats,label_stat):0.2e}$'
        elif label_stat.endswith('_sum'):
            z = label_stat.split('_')[0]
            label_stat = f'$\Sigma_{z}={getattr(self.stats,label_stat):0.2e}$'
        elif label_stat == 'chi2':
            label_stat = f'$\chi^2/$ndf={self.stats.chi2:0.2}/{self.stats.ndf}'
        else: label_stat = f'{getattr(self.stats,label_stat):0.2e}'
        
        if label_stat is not None:
            if 'label' in self.kwargs:
                label_stat = f'{self.kwargs["label"]} ({label_stat})'
            self.kwargs['label'] = f'{label_stat}'
        
class GraphList(ObjIter):
    def __init__(self, x_arrays, y_arrays,  **kwargs):
        x_arrays = np.array(x_arrays)
        y_arrays = np.array(y_arrays)
        narrays = len(y_arrays)

        if x_arrays.shape != y_arrays.shape:
            x_arrays = np.array( init_attr(None,x_arrays,narrays))        
        
        for key,value in kwargs.items(): 
            if not isinstance(value,list): value = init_attr(None,value,narrays)
            kwargs[key] = init_attr(value,None,narrays)
            
        super().__init__([
            Graph(x_array,y_array, **{ key:value[i] for key,value in kwargs.items() })
            for i,(x_array,y_array) in enumerate(zip(x_arrays,y_arrays))
        ])
    
def get_data(obj):
    if hasattr(obj, 'histo'):
        y, yerr = obj.histo, obj.error
        x, xerr = get_bin_centers(obj.bins), get_bin_widths(obj.bins)
    else:
        y, yerr = obj.y_array, obj.yerr
        x, xerr = obj.x_array, obj.xerr
    return x, y, xerr, yerr
        
class Ratio(Graph):
    def __init__(self, num, den, inv=False, method=None, num_transform=None, den_transform=None, **kwargs):
        if inv: num, den = den, num

        num_x, num_y, num_xerr, num_yerr = get_data(num)
        den_x, den_y, den_xerr, den_yerr = get_data(den)

        np.testing.assert_allclose(num_x, den_x)

        if callable(num_transform): num_y = num_transform(num_y)
        if callable(den_transform): den_y = den_transform(den_y)

        ratio = safe_divide(num_y, den_y, np.nan)
        error = ratio * safe_divide(den_yerr, den_y, np.nan)

        if method == 'g-test':
            error = np.abs(error/ratio)
            ratio = np.log(ratio)
            error = np.sqrt( (num_yerr/num_y)**2 + (error/ratio)**2 )
            ratio = 2*num_y*ratio
            error = ratio*error

        kwargs['color'] = kwargs.get('color',den.kwargs.get('color',None))
        super().__init__(num_x, ratio, xerr=num_xerr, yerr=error, **kwargs)

class Difference(Graph):
    def __init__(self, num, den, inv=False, method=None, **kwargs):
        if inv: num, den = den, num

        num_x, num_y, num_xerr, num_yerr = get_data(num)
        den_x, den_y, den_xerr, den_yerr = get_data(den)

        np.testing.assert_allclose(num_x, den_x)

        difference = num_y - den_y
        error = np.sqrt( num_yerr**2 + den_yerr**2 )

        if method is 'normalize':
            mean = (num_y+den_y)/2
            difference = difference/mean
            error = error/mean

        if method is 'standardize': 
            std = np.std(difference)
            difference = difference/std
            error = error/std

        if method is 'chi2':
            error = 2*difference*error
            difference = difference**2

            error = np.sqrt( (error/difference)**2 + (num_yerr/num_y)**2 )
            difference = difference/num_y
            error = difference*error

        if method is 'r2':
            std = np.std(num_y)
            error = 2*difference*error
            difference = difference**2/std
            error = error/std
        
        kwargs['color'] = kwargs.get('color',den.kwargs.get('color',None))
        super().__init__(num_x,difference, xerr=num_xerr, yerr=error, **kwargs)

        