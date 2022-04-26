from ..utils import safe_divide, get_bin_centers, get_bin_widths, init_attr
from ..classUtils import ObjIter
import numpy as np

class Stats:
    def __init__(self,graph):
        self.x_mean, self.x_std = np.mean(graph.x_array), np.std(graph.x_array)
        self.y_mean, self.y_std = np.mean(graph.y_array), np.std(graph.y_array)
        self.area = np.trapz(graph.y_array, graph.x_array)
        self.chi2 = np.sum(graph.y_array**2)
        self.ndf = len(graph.y_array)
        
    def __str__(self):
        return '\n'.join([ f'{key}={float(value):0.3e}' for key,value in vars(self).items() ])

class Graph:
    def __init__(self, x_array, y_array, label_stat="area", inv=False, **kwargs):
        if inv: x_array, y_array = y_array, x_array
        
        order = x_array.argsort()
        
        self.x_array = x_array[order]
        self.y_array = y_array[order]
        self.kwargs = kwargs
        
        self.stats = Stats(self)    
        self.set_label(label_stat)
    
    def set_label(self, label_stat='area'):
        if label_stat is None: return
        if label_stat == 'area':
            label_stat = f'$A={self.stats.area:0.2f}$'
        elif label_stat.endswith('_std'):
            z = label_stat.split('_')[0]
            label_stat = f'$\sigma_{z}={getattr(self.stats,label_stat):0.2f}$'
        elif label_stat.endswith('_mean'):
            z = label_stat.split('_')[0]
            label_stat = f'$\mu_{z}={getattr(self.stats,label_stat):0.2f}$'
        elif label_stat == 'chi2':
            label_stat = f'$\chi^2/$ndf={self.stats.chi2:0.2}/{self.stats.ndf}'
        else: label_stat = f'{getattr(self.stats,label_stat):0.2f}'
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
    
        
class Ratio(Graph):
    def __init__(self, num, den, inv=False, num_transform=None, den_transform=None, **kwargs):
        num_histo, den_histo = num.histo, den.histo 
        if inv: num_histo, den_histo = den.histo, num.histo

        if callable(num_transform): num_histo = num_transform(num_histo)
        if callable(den_transform): den_histo = den_transform(den_histo)

        ratio = safe_divide(num_histo, den_histo, np.nan)
        error = ratio * safe_divide(den.error, den.histo, np.nan)
        
        bin_centers = get_bin_centers(num.bins)
        bin_widths = get_bin_widths(num.bins)
        
        kwargs['color'] = kwargs.get('color',den.kwargs.get('color',None))
        kwargs['xerr'] = bin_widths
        kwargs['yerr'] = error
        super().__init__(bin_centers,ratio,**kwargs)

class Difference(Graph):
    def __init__(self, num, den, inv=False, standardize=False, **kwargs):
        difference = num.histo - den.histo if not inv else den.histo - num.histo
        error = np.sqrt( num.error**2 + den.error**2 )

        if standardize: 
            std = np.std(difference)
            difference = difference/std
            # error = error/std

        bin_centers = get_bin_centers(num.bins)
        bin_widths = get_bin_widths(num.bins)

        kwargs['color'] = kwargs.get('color',den.kwargs.get('color',None))
        kwargs['xerr'] = bin_widths
        kwargs['yerr'] = error
        super().__init__(bin_centers,difference,**kwargs)

        