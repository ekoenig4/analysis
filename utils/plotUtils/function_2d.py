import scipy.stats as f_stats
from scipy.optimize import curve_fit , fmin
from scipy import integrate, interpolate
import numpy as np
import re

from sympy import Q

from utils.utils import get_avg_std, get_bin_centers, get_bin_widths

class Function2D:
  __instance__ = ['pdf','func','cdf','rvs','sf']

  def __init__(self, x=np.array([0]), y=np.array([0]), params=dict(), pcov=None, n_obs=0, show=False, **kwargs):
    self.params = params
    self.nparams = len(params)
    for key,value in params.items(): setattr(self, key, value)
    for key in self.__instance__: 
      if hasattr(self,  f'_{key}'):
        setattr(self, key, getattr(self, f'_{key}'))

    self.show=show

    self.plot(x, y)
    self.pcov = pcov
    self.n_obs = n_obs
    self.kwargs = kwargs
    self.kwargs['linestyle'] = kwargs.get('linestyle','--')
    self.kwargs['linewidth'] = kwargs.get('linewidth', 2)

  def plot(self, x, y):
    self.x_array, self.xerr = x, None
    self.y_array, self.yerr = y, None
    self.z_array, self.zerr = self.func(x,y), None

  @classmethod
  def fit(cls, x, y, z, zerr=None, n_obs=None, x_bounds=(-np.inf, np.inf), y_bounds=(-np.inf, np.inf), peak=False, **kwargs):
    if n_obs is None: n_obs = len(x)

    mask = (x > x_bounds[0]) & (x < x_bounds[1]) & (y > y_bounds[0]) & (y < y_bounds[1])
    X, Y, Z = x[mask], y[mask], z[mask]
    if zerr is not None: zerr = zerr[mask]

    if peak:
      nparams = cls().nparams
      maxarg = np.argmax(Z)

      m = 2
      mask = Z > Z[maxarg]/m
      while not (mask.sum() > nparams+1):
        m += 1
        mask = Z > Z[maxarg]/m

      X, Y, Z = X[mask], Y[mask], Z[mask]
      if zerr is not None: zerr = zerr[mask]
      

    try:
      p0 = cls.best(X, Y, Z) if hasattr(cls, 'best') else None

      R = np.vstack((X,Y))
      popt, pcov = curve_fit(lambda R, *params : cls.func(*R, *params), R, Z, sigma=zerr, p0=p0, check_finite=False)
    except RuntimeError:
      print("[ERROR] Unable to fit")
      fit = cls(np.array([0]), np.array([0]))
      return fit

    x = np.linspace(x[0], x[-1], 100)
    y = np.linspace(y[0], y[-1], 100)
    return cls(x, y, *popt, pcov=pcov, n_obs=n_obs, **kwargs)

  @classmethod
  def fit_histo2d(cls, histo2d, **kwargs):
    x = get_bin_centers(histo2d.x_bins)
    y = get_bin_centers(histo2d.y_bins)
    fit = cls.fit(x, y, histo2d.histo, yerr=histo2d.error, n_obs=histo2d.ndata, **kwargs)

    return fit

  @classmethod
  def fit_graph(cls, graph2d, **kwargs):
    fit = cls.fit(graph2d.x_array, graph2d.y_array, graph2d.z_array, **kwargs)

    return fit

class gaussian(Function2D):
  def __init__(self, x=np.array([0]), y=np.array([0]), n=1, x_mu=0, x_sigma=1, y_mu=0, y_sigma=1, **kwargs):
    super().__init__(x, y, dict(n=n, x_mu=x_mu, x_sigma=x_sigma, y_mu=y_mu, y_sigma=y_sigma), **kwargs)

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

  @staticmethod
  def best(x, y):
    mu, sigma = get_avg_std(x, y)
    func = lambda n : np.sum((gaussian.func(x, n, mu=mu, sigma=sigma)-y)**2)
    n = fmin(func, x0=np.sum(y), maxiter=10, full_output=False, disp=False)[0]
    return n, mu, sigma

