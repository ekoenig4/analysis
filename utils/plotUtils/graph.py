from ..utils import safe_divide, get_bin_centers, get_bin_widths, init_attr
from ..classUtils import ObjIter
import numpy as np


class Stats:
    def __init__(self,graph):
        self.x_mean, self.x_std = np.mean(graph.x_array), np.std(graph.x_array)
        self.y_mean, self.y_std = np.mean(graph.y_array), np.std(graph.y_array)
        self.area = np.trapz(graph.y_array, graph.x_array)
        
    def __str__(self):
        return '\n'.join([ f'{key}={float(value):0.3e}' for key,value in vars(self).items() ])

class Graph:
    def __init__(self, x_array, y_array, label_stat="area", **kwargs):
        order = x_array.argsort()
        
        self.x_array = x_array[order]
        self.y_array = y_array[order]
        self.kwargs = kwargs
        
        self.stats = Stats(self)    
        self.set_label(label_stat)
    
    def set_label(self, label_stat='area'):
        if label_stat == 'area':
            label_stat = f'$A={self.stats.area:0.2f}$'
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
    def __init__(self, num, den, inv=False, **kwargs):
        ratio = safe_divide(num.histo, den.histo, 0) if not inv else safe_divide(den.histo,num.histo,0)
        error = ratio * safe_divide(den.error, den.histo, 0)
        
        bin_centers = get_bin_centers(num.bins)
        bin_widths = get_bin_widths(num.bins)
        
        kwargs['color'] = kwargs.get('color',den.kwargs.get('color',None))
        kwargs['xerr'] = bin_widths
        kwargs['yerr'] = error
        super().__init__(bin_centers,ratio,**kwargs)
        