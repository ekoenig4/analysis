from ..utils import flatten, autobin, is_iter, get_avg_std, init_attr
from ..xsecUtils import lumiMap
from ..classUtils import ObjIter,AttrArray
import numpy as np
import numba
import awkward as ak

class Stats:
    def __init__(self,histo):
        self.nevents = np.sum(histo.weights)
        self.mean, self.stdv = get_avg_std(histo.array,histo.weights,histo.bins)
        self.minim, self.maxim = np.min(histo.array), np.max(histo.array)
        
    def __str__(self):
        return '\n'.join([ f'{key}={float(value):0.3e}' for key,value in vars(self).items() ])


def histogram(array, bins, weights):
    # steps = bins[1:]-bins[:-1]
    # if steps.max()-steps.min() < 1e10:
    #     step = steps.mean()
    #     range_mask = (array >= bins[0]) & (array < bins[-1])
    #     array,weights = array[range_mask],weights[range_mask]
    #     return np.bincount((array/step).astype(int),weights=weights)
    return np.histogram(array,bins,(bins[0],bins[-1]),None,weights)[0]

class Histo:
    def __init__(self, array, bins=None, weights=None, density=False, cumulative=False, lumi=None, 
                 label_stat='events', is_data=False, is_signal=False, sumw2=True, scale=1,__id__=None,
                 **kwargs):
        if weights is not None: weights = flatten(ak.ones_like(array)*weights)

        self.array = flatten(array)
        self.counts = len(self.array)
        
        self.bins = autobin(self.array) if bins is None else bins
        self.weights = np.ones((self.counts,)) if weights is None else weights
        self.sumw2 = sumw2
        
        self.kwargs = kwargs
        self.label = kwargs.get('label','')
        
        self.is_data = is_data 
        self.is_signal = is_signal 
        self.is_bkg = not (is_data or is_signal)
        
        lumi,_ = lumiMap.get(lumi,(lumi,None))
        if not is_data: self.weights = lumi * self.weights
            
        self.stats = Stats(self)      
        if scale == 'xs': scale = 1/lumi
        if density or cumulative: scale = 1/np.sum(self.weights)
        self.rescale(scale)
        
            
        self.density = density 
        self.cdf(cumulative)
        
        self.set_label(label_stat)  
        
    def rescale(self,scale):
        if scale is not None:
            if is_iter(scale): scale = flatten(scale)            
            self.weights = scale * self.weights
        
            if scale != 1 and not is_iter(scale) and isinstance(scale,int):
                self.label = f'{self.label} x {scale}'
                
        self.histo = histogram(self.array, self.bins, self.weights)
        
        if self.sumw2: 
            self.error = histogram(self.array,self.bins,self.weights**2)
            self.error = np.sqrt(self.error)
        else:
            self.error = np.sqrt(self.histo)
        
        # self.error = np.sqrt(self.histo) if not self.sumw2 else np.sqrt(histogram(self.array,bins=self.bins,weights=self.weights**2))
                
    def cdf(self, cumulative):
        if cumulative == 1: # CDF Below 
            self.histo = np.cumsum(self.histo)
            self.error = np.sqrt( np.cumsum(self.error**2) )
        elif cumulative == -1: # CDF Above
            self.histo = np.cumsum(self.histo[::-1])[::-1]
            self.error = np.sqrt(np.cumsum(self.error[::-1]**2))[::-1]
        self.cumulative = cumulative
                    
    def set_label(self, label_stat='events'):
        nevents = self.stats.nevents
        mean = self.stats.mean
        stdv = self.stats.stdv
        
        if label_stat == 'events':
            label_stat = f'{nevents:0.2e}'
        if label_stat == 'mean':
            exponent = int(np.log10(mean))
            exp_str = "" if exponent == 0 else "\\times 10^{"+str(exponent)+"}"
            label_stat = f'$\mu={mean/(10**exponent):0.2f} {exp_str}$'
        if label_stat == 'mean_stdv':
            exponent = int(np.log10(mean))
            exp_str = "" if exponent == 0 else "\\times 10^{"+str(exponent)+"}"
            label_stat = f'$\mu={mean/(10**exponent):0.2f} \pm {stdv/(10**exponent):0.2f} {exp_str}$'
        
        self.kwargs['label'] = f'{self.label} ({label_stat})'
        
class HistoList(ObjIter):
    def __init__(self, arrays, bins=None, **kwargs):
        attrs = AttrArray(arrays=arrays,**kwargs)
        kwargs = attrs[attrs.fields[1:]]
        
        if bins is not None: bins = np.array(bins)
        multi_binned = False if bins is None else len(bins.shape) > 1
    
        histolist = []
        for i,array in enumerate(arrays):
            _bins = bins[i] if multi_binned else bins
            histo = Histo(array,bins=_bins, **{ key:value[i] for key,value in kwargs.items() })
            if bins is None: bins = histo.bins
            histolist.append(histo)
        super().__init__(histolist)
            
class DataList(HistoList):
    def __init__(self, arrays, bins=None, histtype=None, **kwargs):
        super().__init__(arrays,bins=bins,**kwargs)

class Stack(HistoList):
    def __init__(self, arrays, bins=None, density=False, cumulative=False, **kwargs):
        super().__init__(arrays, bins=bins, **kwargs)
        
        if density or cumulative: 
            nevents = self.stats.nevents.npy.sum()
            self.apply(lambda histo : histo.rescale(1/nevents))
        if cumulative:
            self.apply(lambda histo : histo.cdf(cumulative))
        