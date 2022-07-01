import scipy.stats as f_stats
from scipy.optimize import curve_fit  
from scipy import integrate, interpolate
import numpy as np

from utils.utils import get_bin_centers

class Function:
  __instance__ = ['pdf','func','cdf','rvs','sf']

  def __init__(self, x, params, pcov=None, n_obs=0, **kwargs):
    self.nparams = len(params)
    for key,value in params.items(): setattr(self, key, value)
    for key in self.__instance__: 
      if hasattr(self,  f'_{key}'):
        setattr(self, key, getattr(self, f'_{key}'))

    self.plot(x)
    self.pcov = pcov
    self.n_obs = n_obs
    self.kwargs = kwargs
    self.kwargs['linestyle'] = kwargs.get('linestyle','--')
    self.kwargs['linewidth'] = kwargs.get('linewidth', 2)

  def plot(self, x):
    self.x_array, self.xerr = x, None
    self.y_array, self.yerr = self.func(x), None

  def chi2(self, x, obs_y):
    exp_y = self.func(x)
    chi2 = np.sum((obs_y - exp_y)**2/exp_y)
    pvalue = f_stats.chi2.sf(chi2, self.n_obs - 1)
    return chi2, pvalue

  def chi2_histo(self, histo):
    x = get_bin_centers(histo.bins)
    y = histo.histo
    return self.chi2(x, y)

  def ks(self, histo):
    ndata = histo.ndata
    ecdf = histo.ecdf()
    cdf = self.cdf(ecdf.x_array)

    ks = np.abs( ecdf.y_array-cdf ).max()
    pvalue = f_stats.kstwo.sf(ks, ndata)
    return ks,pvalue

  def r2(self, x, obs_y):
    exp_y = self.func(x)
    obs_mu = obs_y.mean()
    ss_res = (obs_y - exp_y)**2
    ss_tot = (obs_y - obs_mu)**2
    return 1 - (ss_res.sum())/(ss_tot.sum())

  def r2_histo(self, histo):
    x = get_bin_centers(histo.bins)
    y = histo.histo
    return self.r2(x, y)

  @classmethod
  def fit(cls, x, y, n_obs=None, **kwargs):
    if n_obs is None: n_obs = len(x)

    try:
      popt, pcov = curve_fit(cls.func, x, y, check_finite=False)
    except RuntimeError:
      print("[ERROR] Unable to fit")
      fit = cls(np.array([0]))
      return fit


    x_lo, x_hi = x[0], x[-1]
    x = np.linspace(x_lo, x_hi, 100)

    return cls(x, *popt, pcov=pcov, n_obs=n_obs, **kwargs)

  @classmethod
  def fit_histo(cls, histo, **kwargs):
    x = get_bin_centers(histo.bins)
    fit = cls.fit(x, histo.histo, n_obs=histo.ndata, **kwargs)

    histo.stats.ks, histo.stats.ks_pvalue = fit.ks(histo)
    histo.stats.chi2, histo.stats.chi2_pvalue = fit.chi2_histo(histo)
    histo.stats.r2 = fit.r2_histo(histo)

    if histo.cumulative == 1:
      fit.y_array = fit.cdf(fit.x_array)
    if histo.cumulative == -1:
      fit.y_array = fit.sf(fit.x_array)

    return fit

  @classmethod
  def fit_graph(cls, graph, **kwargs):
    fit = cls.fit(graph.x_array, graph.y_array, **kwargs)


    return fit
    

class gaussian(Function):
  def __init__(self, x, n=1, mu=0, sigma=1, **kwargs):
    super().__init__(x, dict(n=n, mu=mu, sigma=sigma), **kwargs)

  @staticmethod
  def rvs(mu=0, sigma=1, size=1): return f_stats.norm.rvs(mu, sigma, size=size)
  def _rvs(self, size=1): return gaussian.rvs(self.mu, self.sigma, size)

  @staticmethod
  def pdf(x, mu=0, sigma=1): return f_stats.norm.pdf(x, mu, sigma)
  def _pdf(self, x): return gaussian.pdf(x, self.mu, self.sigma)

  @staticmethod
  def cdf(x, mu=0, sigma=1): return f_stats.norm.cdf(x, mu, sigma)
  def _cdf(self, x): return gaussian.cdf(x, self.mu, self.sigma)

  @staticmethod
  def sf(x, mu=0, sigma=1): return f_stats.norm.sf(x, mu, sigma)
  def _sf(self, x): return gaussian.sf(x, self.mu, self.sigma)

  @staticmethod
  def func(x, n=1, mu=0, sigma=1): return n*gaussian.pdf(x, mu, sigma)
  def _func(self, x): return gaussian.func(x, self.n, self.mu, self.sigma)

class norm(Function):
  def __init__(self, x, n=1, sigma=1, **kwargs):
    super().__init__(x, dict(n=n, mu=0, sigma=sigma), **kwargs)

  @staticmethod
  def rvs(sigma=1, size=1): return f_stats.norm.rvs(0, sigma, size=size)
  def _rvs(self, size=1): return gaussian.rvs(self.mu, self.sigma, size)

  @staticmethod
  def pdf(x, sigma=1): return f_stats.norm.pdf(x, 0, sigma)
  def _pdf(self, x): return gaussian.pdf(x, self.mu, self.sigma)

  @staticmethod
  def cdf(x, sigma=1): return f_stats.norm.cdf(x, 0, sigma)
  def _cdf(self, x): return gaussian.cdf(x, self.mu, self.sigma)

  @staticmethod
  def sf(x, sigma=1): return f_stats.norm.sf(x, 0, sigma)
  def _sf(self, x): return gaussian.sf(x, self.mu, self.sigma)

  @staticmethod
  def func(x, n=1, sigma=1): return n*gaussian.pdf(x, 0, sigma)
  def _func(self, x): return gaussian.func(x, self.n, self.mu, self.sigma)


class linear(Function):
  def __init__(self, x, c0=1, c1=1, **kwargs):
    super().__init__(x, dict(c0=c0, c1=c1), **kwargs)

  @staticmethod
  def func(x, c0=1, c1=1): return c1*x + c0 
  def _func(self, x): return linear.func(x, self.c0, self.c1)


class quadratic(Function):
  def __init__(self, x, c0=1, c1=1, c2=1, **kwargs):
    super(self).__init__(x, dict(c0=c0, c1=c1, c2=c2), **kwargs)

  @staticmethod
  def func(x, c0=1, c1=1, c2=1): return c2*x*x + c1*x + c0 
  def _func(self, x): return quadratic.func(x, self.c0, self.c1, self.c2)

class custom_pdf(f_stats.rv_continuous):
  def __init__(self, histo=None, bins=None):
    super().__init__()
    x = get_bin_centers(bins)
    pdf, bins = np.histogram(x, bins=bins, weights=histo, density=True)
    cdf = integrate.cumtrapz(pdf, bins[:-1])
    sf = 1-cdf 

    self._pdf = interpolate.interp1d(x, pdf, fill_value='extrapolate')
    self._cdf = interpolate.interp1d(x[:-1], cdf, fill_value='extrapolate')
    self._sf  = interpolate.interp1d(x[:-1], sf,  fill_value='extrapolate')

class custom_pdf_from_histo(custom_pdf):
    def __init__(self, histo):
      super().__init__(histo.histo, histo.bins)
