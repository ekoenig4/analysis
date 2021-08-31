from . import *

flatten = lambda array : ak.to_numpy( ak.flatten(array,axis=None) )

class Sample:
    def __init__(self,data,bins=None,weight=None,density=False,lumi=1,label="",is_data=False,is_signal=False,sumw2=True,scale=True,**attrs):
        self.data = flatten(data)
        self.nevents = len(self.data)
        
        self.is_data = is_data
        self.is_signal = is_signal
        self.is_bkg = not( is_data or is_signal )
        self.color = attrs.get("color",None)

        self.attrs = attrs

        self.bins = autobin(self.data) if bins is None else bins
        self.weight = np.array([1.0]*self.nevents) if (weight is None or not scale) else flatten(weight)
        if weight is not None and not self.is_data and scale: # scale by luminosity is a weight is given and sample is not data
            self.weight = lumi * self.weight

        self.scaled_nevents = ak.sum(self.weight)
        self.label = f"{label} ({self.scaled_nevents:0.2e})"

        if density: self.weight = self.weight/self.scaled_nevents
        self.histo = np.histogram(self.data,bins=self.bins,weights=self.weight)[0]

        if sumw2:
            sumw2      = np.histogram(self.data,bins=self.bins,weights=self.weight**2)[0]
            self.error = np.sqrt(sumw2)
        else:
            self.error = np.sqrt(self.histo)

class Samplelist(list):
    def __init__(self,datalist,bins,weights=None,density=False,lumi=1,labels="",is_datas=False,is_signals=False,sumw2=True,scale=True,**attrs):
        self.bins = bins
        self.density = density
        self.lumi = lumi
        self.nsample = len(datalist)
        defaults = dict(
            histtypes="bar" if self.nsample == 1 else "step",
        )
        is_signals = init_attr(is_signals,False,self.nsample)
        is_datas = init_attr(is_datas,False,self.nsample)
        labels = init_attr(labels,"",self.nsample)
        weights = init_attr(weights,None,self.nsample)
        for key in attrs: attrs[key] = init_attr(attrs[key],defaults.get(key,None),self.nsample)

        for i,data in enumerate(datalist):
            sample = Sample(data,bins=self.bins,weight=weights[i],lumi=lumi,density=density,label=labels[i],sumw2=sumw2,scale=scale,
                            is_data=is_datas[i],is_signal=is_signals[i],**{key[:-1]:value[i] for key,value in attrs.items()})
            if self.bins is None: self.bins = sample.bins
            self.append(sample)

        self.has_data = any(sample.is_data for sample in self)
        self.nmc      = sum(not(sample.is_data or sample.is_signal) for sample in self)

class Stack(list):

    def datalist(self): return [ sample.data for sample in self ]
    def weights(self): return [ sample.weight for sample in self ]
    def histos(self): return np.array([ sample.histo for sample in self ])
    def errors(self): return np.array([ sample.error for sample in self ])
    def labels(self): return [ sample.label for sample in self ]
    def attrs(self):
        attrs = {}
        for sample in self: attrs.update(**{ key:None for key in sample.attrs})
        for key in attrs: attrs[key] = [ sample.attrs.get(key,None) for sample in self  ]
        return attrs
        
    def add(self,*samples):
        for sample in samples:
            if type(sample) == list:
                self.add(*sample)
                continue
            self.append(sample)
