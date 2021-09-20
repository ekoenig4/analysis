from ..utils import *


class Stack(list):

    def datalist(self): return [sample.data for sample in self]
    def weights(self): return [sample.weight for sample in self]
    def histos(self): return np.array([sample.histo for sample in self])
    def errors(self): return np.array([sample.error for sample in self])
    def labels(self): return [sample.label for sample in self]

    def attrs(self):
        attrs = {}
        for sample in self:
            attrs.update(**{key: None for key in sample.attrs})
        for key in attrs:
            attrs[key] = [sample.attrs.get(key, None) for sample in self]
        return attrs

    def add(self, *samples):
        for sample in samples:
            if type(sample) == list:
                self.add(*sample)
                continue
            self.append(sample)
