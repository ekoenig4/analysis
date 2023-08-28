import ROOT
import numpy as np
import awkward as ak

def get_bins(axis):
    return np.array([axis.GetBinLowEdge(i) for i in range(1, axis.GetNbins()+2)])

class TEfficiency:
    @classmethod
    def from_root(cls, fname, name):
        tfile = ROOT.TFile.Open(fname, 'read')
        teff = tfile.Get(name)

        if teff == None:
            raise RuntimeError('Cannot find object {} in file {}'.format(name, fname))

        h_total = teff.GetTotalHistogram()

        if h_total.InheritsFrom('TH2'):
            return TEfficiency2D(teff)

        return TEfficiency1D(teff)
    
    def GetAbsoluteErrorDown(self, *x):
        return self.GetEfficiency(*x) - self.GetRelativeErrorDown(*x)

    def GetAbsoluteErrorUp(self, *x):
        return self.GetEfficiency(*x) + self.GetRelativeErrorUp(*x)

    def GetPercentErrorDown(self, *x):
        return 1 - self.GetRelativeErrorDown(*x) / self.GetEfficiency(*x)

    def GetPercentErrorUp(self, x):
        return 1 + self.GetRelativeErrorUp(*x) / self.GetEfficiency(*x)

class TEfficiency1D(TEfficiency):

    @classmethod
    def from_root(cls, fname, name):
        tfile = ROOT.TFile.Open(fname, 'read')
        teff = tfile.Get(name)
        if teff == None:
            raise RuntimeError('Cannot find object {} in file {}'.format(name, fname))
        return cls(teff)

    def __init__(self, teff):
        self.teff = teff

        self.xbins = get_bins(teff.GetTotalHistogram().GetXaxis())
        self.xcenters = (self.xbins[1:] + self.xbins[:-1])/2

        self.eff = np.vectorize(lambda x : self.teff.GetEfficiency(self.teff.FindFixBin(x)))(self.xcenters)
        self.eff_lo = np.vectorize(lambda x : self.teff.GetEfficiencyErrorLow(self.teff.FindFixBin(x)))(self.xcenters)
        self.eff_hi = np.vectorize(lambda x : self.teff.GetEfficiencyErrorUp(self.teff.FindFixBin(x)))(self.xcenters)

    def GetEfficiency(self, x):
        num = ak.num(x)
        x = ak.flatten(x)

        index = np.digitize(x, self.xbins) - 1
        valid = (index >= 0) & (index < len(self.xbins)-1)
        index = np.where(valid, index, 0)
        return ak.unflatten(np.where(valid, self.eff[index], 0), num)
    
    def GetRelativeErrorDown(self, x):
        num = ak.num(x)
        x = ak.flatten(x)

        index = np.digitize(x, self.xbins) - 1
        valid = (index >= 0) & (index < len(self.xbins)-1)
        index = np.where(valid, index, 0)
        return ak.unflatten(np.where(valid, self.eff_lo[index], 0), num)

    def GetRelativeErrorUp(self, x):
        num = ak.num(x)
        x = ak.flatten(x)

        index = np.digitize(x, self.xbins) - 1
        valid = (index >= 0) & (index < len(self.xbins)-1)
        index = np.where(valid, index, 0)
        return ak.unflatten(np.where(valid, self.eff_hi[index], 0), num)
    
class TEfficiency2D(TEfficiency):

    @classmethod
    def from_root(cls, fname, name):
        tfile = ROOT.TFile.Open(fname, 'read')
        teff = tfile.Get(name)
        if teff == None:
            raise RuntimeError('Cannot find object {} in file {}'.format(name, fname))
        return cls(teff)

    def __init__(self, teff):
        self.teff = teff

        self.xbins = get_bins(teff.GetTotalHistogram().GetXaxis())
        self.xcenters = (self.xbins[1:] + self.xbins[:-1])/2

        self.ybins = get_bins(teff.GetTotalHistogram().GetYaxis())
        self.ycenters = (self.ybins[1:] + self.ybins[:-1])/2

        X, Y = np.meshgrid(self.xcenters, self.ycenters)

        self.eff = np.vectorize(lambda x,y : self.teff.GetEfficiency(self.teff.FindFixBin(x,y)))(X, Y).reshape(X.shape).T
        self.eff_lo = np.vectorize(lambda x,y : self.teff.GetEfficiencyErrorLow(self.teff.FindFixBin(x,y)))(X, Y).reshape(X.shape).T
        self.eff_hi = np.vectorize(lambda x,y : self.teff.GetEfficiencyErrorUp(self.teff.FindFixBin(x,y)))(X, Y).reshape(X.shape).T

    def GetEfficiency(self, x, y):
        num = ak.num(x)
        x, y = ak.flatten(x), ak.flatten(y)

        x_index = np.digitize(x, self.xbins) - 1
        y_index = np.digitize(y, self.ybins) - 1

        valid = (x_index >= 0) & (x_index < len(self.xbins)-1) & (y_index >= 0) & (y_index < len(self.ybins)-1)
        x_index = np.where(valid, x_index, 0)
        y_index = np.where(valid, y_index, 0)

        return ak.unflatten(np.where(valid, self.eff[x_index, y_index], 0), num)
    
    def GetRelativeErrorDown(self, x, y):
        num = ak.num(x)
        x, y = ak.flatten(x), ak.flatten(y)

        x_index = np.digitize(x, self.xbins) - 1
        y_index = np.digitize(y, self.ybins) - 1

        valid = (x_index >= 0) & (x_index < len(self.xbins)-1) & (y_index >= 0) & (y_index < len(self.ybins)-1)
        x_index = np.where(valid, x_index, 0)
        y_index = np.where(valid, y_index, 0)

        return ak.unflatten(np.where(valid, self.eff_lo[x_index, y_index], 0), num)

    def GetRelativeErrorUp(self, x, y):
        num = ak.num(x)
        x, y = ak.flatten(x), ak.flatten(y)

        x_index = np.digitize(x, self.xbins) - 1
        y_index = np.digitize(y, self.ybins) - 1

        valid = (x_index >= 0) & (x_index < len(self.xbins)-1) & (y_index >= 0) & (y_index < len(self.ybins)-1)
        x_index = np.where(valid, x_index, 0)
        y_index = np.where(valid, y_index, 0)

        return ak.unflatten(np.where(valid, self.eff_hi[x_index, y_index], 0), num)