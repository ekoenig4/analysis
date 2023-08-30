import ROOT
import numpy as np
import awkward as ak
from scipy.interpolate import LinearNDInterpolator

def get_bins(axis):
    return np.array([axis.GetBinLowEdge(i) for i in range(1, axis.GetNbins()+2)])

def interp1d(x, y, mask):
    flat_x = x[mask]
    flat_y = y[mask]
    interp_y = np.interp(x, flat_x, flat_y)
    interp_y = np.where(np.isnan(interp_y), 0, interp_y)

    return interp_y

def interp2d(x, y, z, mask):
    flat_x = x[mask]
    flat_y = y[mask]
    flat_z = z[mask]

    f = LinearNDInterpolator(np.array([flat_x, flat_y]).T, flat_z)
    interp_z = f(x, y)
    interp_z = np.where(np.isnan(interp_z), 0, interp_z)

    return interp_z

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

        valid = (self.eff >= 0) & (self.eff <= 1)
        self.eff = interp1d(self.xcenters, self.eff, valid)
        self.eff_lo = interp1d(self.xcenters, self.eff_lo, valid)
        self.eff_hi = interp1d(self.xcenters, self.eff_hi, valid)

    def GetEfficiency(self, x):
        num = ak.num(x)
        x = ak.flatten(x)

        index = np.digitize(x, self.xbins) - 1
        index = np.clip(index, 0, len(self.xbins)-2)
        return ak.unflatten(self.eff[index], num)
    
    def GetRelativeErrorDown(self, x):
        num = ak.num(x)
        x = ak.flatten(x)

        index = np.digitize(x, self.xbins) - 1
        index = np.clip(index, 0, len(self.xbins)-2)
        return ak.unflatten(self.eff_lo[index], num)

    def GetRelativeErrorUp(self, x):
        num = ak.num(x)
        x = ak.flatten(x)

        index = np.digitize(x, self.xbins) - 1
        index = np.clip(index, 0, len(self.xbins)-2)
        return ak.unflatten(self.eff_hi[index], num)
    
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

        self.eff = np.vectorize(lambda x,y : self.teff.GetEfficiency(self.teff.FindFixBin(x,y)))(X, Y)
        self.eff_lo = np.vectorize(lambda x,y : self.teff.GetEfficiencyErrorLow(self.teff.FindFixBin(x,y)))(X, Y)
        self.eff_hi = np.vectorize(lambda x,y : self.teff.GetEfficiencyErrorUp(self.teff.FindFixBin(x,y)))(X, Y)

        valid = (self.eff >= 0) & (self.eff <= 1)
        self.eff = interp2d(X, Y, self.eff, valid)
        self.eff_lo = interp2d(X, Y, self.eff_lo, valid)
        self.eff_hi = interp2d(X, Y, self.eff_hi, valid)

    def GetEfficiency(self, x, y):
        num = ak.num(x)
        x, y = ak.flatten(x), ak.flatten(y)

        x_index = np.digitize(x, self.xbins) - 1
        y_index = np.digitize(y, self.ybins) - 1

        x_index = np.clip(x_index, 0, len(self.xbins)-2)
        y_index = np.clip(y_index, 0, len(self.ybins)-2)

        return ak.unflatten(self.eff[x_index, y_index], num)
    
    def GetRelativeErrorDown(self, x, y):
        num = ak.num(x)
        x, y = ak.flatten(x), ak.flatten(y)

        x_index = np.digitize(x, self.xbins) - 1
        y_index = np.digitize(y, self.ybins) - 1

        x_index = np.clip(x_index, 0, len(self.xbins)-2)
        y_index = np.clip(y_index, 0, len(self.ybins)-2)

        return ak.unflatten(self.eff_lo[x_index, y_index], num)

    def GetRelativeErrorUp(self, x, y):
        num = ak.num(x)
        x, y = ak.flatten(x), ak.flatten(y)

        x_index = np.digitize(x, self.xbins) - 1
        y_index = np.digitize(y, self.ybins) - 1

        x_index = np.clip(x_index, 0, len(self.xbins)-2)
        y_index = np.clip(y_index, 0, len(self.ybins)-2)

        return ak.unflatten(self.eff_hi[x_index, y_index], num)