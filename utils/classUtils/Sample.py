from ..utils import *


class Sample:
    def __init__(self, data, bins=None, weight=None, density=False, cumulative=False, lumi=1, label="",label_stat='events', is_data=False, is_signal=False, sumw2=True, scale=True, xsec_scale=1, **attrs):
        self.data = flatten(data)
        self.nevents = len(self.data)

        self.is_data = is_data
        self.is_signal = is_signal
        self.is_bkg = not(is_data or is_signal)
        self.color = attrs.get("color", None)

        self.attrs = attrs

        self.bins = autobin(self.data) if bins is None else bins
        self.weight = np.array(
            [1.0]*self.nevents) if (weight is None or not scale) else flatten(weight)
        
        if weight is not None and scale and xsec_scale is not None:
            self.weight = xsec_scale*self.weight
            
            if not is_iter(xsec_scale) and xsec_scale != 1:
                label = f"{label} x {xsec_scale}"
        
        # scale by luminosity is a weight is given and sample is not data
        if weight is not None and not self.is_data and scale:
            self.weight = lumi * self.weight

        # if scale == "xs" scale by luminosity
        if scale == "xs":
            self.weight = self.weight / lumi

        self.scaled_nevents = ak.sum(self.weight)
        
        if label_stat == 'events':
            self.label_stat = f'{self.scaled_nevents:0.2e}'
        if label_stat == 'mean':
            mean,stdv = get_avg_std(self.data,self.weight)
            exponent = int(np.log10(mean))
            exp_str = "" if exponent == 0 else "\\times 10^{"+str(exponent)+"}"
            self.label_stat = f'$\mu={mean/(10**exponent):0.2f} {exp_str}$'
        if label_stat == 'mean_stdv':
            mean,stdv = get_avg_std(self.data,self.weight,bins)
            exponent = int(np.log10(mean))
            exp_str = "" if exponent == 0 else "\\times 10^{"+str(exponent)+"}"
            self.label_stat = f'$\mu={mean/(10**exponent):0.2f} \pm {stdv/(10**exponent):0.2f} {exp_str}$'
        
        self.label = label if label_stat is None else f"{label} ({self.label_stat})"

        if density or cumulative:
            self.weight = self.weight/self.scaled_nevents
        self.histo = np.histogram(
            self.data, bins=self.bins, weights=self.weight)[0]

        if sumw2:
            sumw2 = np.histogram(self.data, bins=self.bins,
                                 weights=self.weight**2)[0]
            self.error = np.sqrt(sumw2)
        else:
            self.error = np.sqrt(self.histo)
            
        if cumulative == 1:
            self.histo = np.cumsum(self.histo)
            self.error = np.cumsum(self.error)
        if cumulative == -1:
            self.histo = np.cumsum(self.histo[::-1])[::-1]
            self.error = np.cumsum(self.error[::-1])[::-1]

class Samplelist(list):
    def __init__(self, datalist, bins, weights=None, density=False, cumulative=False, lumi=1, labels="",label_stat='events', is_datas=False, is_signals=False, sumw2=True, scale=True, **attrs):
        datalist = [ flatten(data) for data in datalist ]
        self.bins = bins
        if bins is None:
            self.bins = autobin(datalist)
        
        self.density = density
        self.lumi = lumi
        self.nsample = len(datalist)
        defaults = dict(
            histtypes="bar" if self.nsample == 1 else "step",
        )
        is_signals = init_attr(is_signals, False, self.nsample)
        is_datas = init_attr(is_datas, False, self.nsample)
        labels = init_attr(labels, "", self.nsample)
        weights = init_attr(weights, None, self.nsample)
        for key in attrs:
            attrs[key] = init_attr(
                attrs[key], defaults.get(key, None), self.nsample)

        for i, data in enumerate(datalist):
            sample = Sample(data, bins=self.bins, weight=weights[i], lumi=lumi, density=density, cumulative=cumulative, label=labels[i], label_stat=label_stat, sumw2=sumw2, scale=scale,
                            is_data=is_datas[i], is_signal=is_signals[i], **{key[:-1]: value[i] for key, value in attrs.items()})
            if self.bins is None:
                self.bins = sample.bins
                
            if sample.nevents > 0:
                self.append(sample)

        self.has_data = any(sample.is_data for sample in self)
        self.nmc = sum(not(sample.is_data or sample.is_signal)
                       for sample in self)
