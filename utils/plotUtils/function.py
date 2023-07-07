import scipy.stats as f_stats
from scipy.optimize import curve_fit , fmin
from scipy import integrate, interpolate
import numpy as np
import re


from .binning_tools import get_bin_centers, get_bin_widths
from ..ak_tools import *

class Function:
  __instance__ = ['pdf','func','cdf','rvs','sf']

  def __init__(self, x=np.array([0]), params=dict(), pcov=None, n_obs=0, show=False, **kwargs):
    self.params = params
    self.nparams = len(params)
    for key,value in params.items(): setattr(self, key, value)
    for key in self.__instance__: 
      if hasattr(self,  f'_{key}'):
        setattr(self, key, getattr(self, f'_{key}'))

    self.show=show

    self.plot(x)
    self.pcov = pcov
    self.n_obs = n_obs
    self.kwargs = kwargs
    self.kwargs['linestyle'] = kwargs.get('linestyle','--')
    self.kwargs['linewidth'] = kwargs.get('linewidth', 2)

  def __format__(self, spec):
    self.spec = spec
    return str(self)

  def evaluate(self, x):
    return self.func(x)

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

  def r2_graph(self, graph):
    x = graph.x_array
    y = graph.y_array
    return self.r2(x, y)

  @classmethod
  def fit(cls, x, y, yerr=None, n_obs=None, bounds=(-np.inf, np.inf), peak=False, **kwargs):
    if n_obs is None: n_obs = len(x)

    mask = (x > bounds[0]) & (x < bounds[1])
    X, Y = x[mask], y[mask]
    if yerr is not None: yerr = yerr[mask]

    if peak:
      nparams = cls().nparams
      maxarg = np.argmax(Y)

      m = 2
      mask = Y > Y[maxarg]/m
      while not (mask.sum() > nparams+1):
        m += 1
        mask = Y > Y[maxarg]/m

      X, Y = X[mask], Y[mask]
      if yerr is not None: yerr = yerr[mask]
      

    try:
      p0 = cls.best(X, Y) if hasattr(cls, 'best') else None
      popt, pcov = curve_fit(cls.func, X, Y, sigma=yerr, p0=p0, check_finite=False)
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
    # fit = cls.fit(x, histo.histo, yerr=histo.error, n_obs=histo.ndata, **kwargs)
    fit = cls.fit(x, histo.histo, n_obs=histo.ndata, **kwargs)

    if getattr(histo, 'array', None) is not None:
      histo.stats.ks, histo.stats.ks_pvalue = fit.ks(histo)
    histo.stats.chi2, histo.stats.chi2_pvalue = fit.chi2_histo(histo)
    histo.stats.r2 = fit.r2_histo(histo)

    # if histo.cumulative == 1:
    #   fit.y_array = fit.cdf(fit.x_array)
    # if histo.cumulative == -1:
    #   fit.y_array = fit.sf(fit.x_array)

    return fit

  @classmethod
  def fit_graph(cls, graph, **kwargs):
    fit = cls.fit(graph.x_array, graph.y_array, **kwargs)
    graph.stats.r2 = fit.r2_graph(graph)

    return fit

class gaussian(Function):
  def __init__(self, x=np.array([0]), n=1, mu=0, sigma=1, **kwargs):
    super().__init__(x, dict(n=n, mu=mu, sigma=sigma), **kwargs)

  def __str__(self):
    if not hasattr(self,"spec"):
      fvar = lambda v : str(v)
    else: 
      fvar = lambda v : f'{v:{self.spec}}'

    return f"${fvar(self.n)}\exp(-0.5(\frac{{x-{fvar(self.mu)}}}{{{fvar(self.sigma)}}})^2)$"

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

class norm(Function):
  def __init__(self, x=np.array([0]), n=1, sigma=1, **kwargs):
    super().__init__(x, dict(n=n, mu=0, sigma=sigma), **kwargs)

  def __str__(self):
    if not hasattr(self,"spec"):
      fvar = lambda v : str(v)
    else: 
      fvar = lambda v : f'{v:{self.spec}}'

    return f"${fvar(self.n)}\exp(-0.5(\frac{{x}}{{{fvar(self.sigma)}}})^2)$"

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

class crystalball(Function):
  def __init__(self, x=np.array([0]), n=1, mu=0, sigma=1, beta=1, m=2, **kwargs):
    super().__init__(x, dict(n=n, mu=mu, sigma=sigma, beta=beta, m=m), **kwargs)

  def __str__(self):
    if not hasattr(self,"spec"):
      fvar = lambda v : str(v)
    else: 
      fvar = lambda v : f'{v:{self.spec}}'

    z = f"\\frac{{x-{fvar(self.mu)}}}{{{fvar(self.sigma)}}}"
    A = (self.m/np.abs(self.beta))**self.m
    B = self.m/np.abs(self.beta)-np.abs(self.beta)
    em = '{'+fvar(-self.m)+'}'

    lines = [f"${fvar(self.n)}[\exp(-0.5({z})^2)$], ${z} < {fvar(self.beta)}$",
             f"${fvar(self.n)}[{fvar(A)}({fvar(B)}+{z})^{em}$, ${z} > {fvar(self.beta)}$"]
    return '\n'.join(lines)
  
  @staticmethod
  def rvs(mu=0, sigma=1, beta=1, m=2, size=1): return (f_stats.crystalball.rvs(beta, m, size=size)-mu)/sigma
  def _rvs(self, size=1): return crystalball.rvs(self.mu, self.sigma, self.beta, self.m, size)

  @staticmethod
  def pdf(x, mu=0, sigma=1, beta=1, m=2): return f_stats.crystalball.pdf(-(x-mu)/sigma, beta, m)
  def _pdf(self, x): return crystalball.pdf(x, self.mu, self.sigma, self.beta, self.m)

  @staticmethod
  def cdf(x, mu=0, sigma=1, beta=1, m=2): return f_stats.crystalball.cdf(-(x-mu)/sigma, beta, m)
  def _cdf(self, x): return crystalball.cdf(x, self.mu, self.sigma, self.beta, self.m)

  @staticmethod
  def sf(x, mu=0, sigma=1, beta=1, m=2): return f_stats.crystalball.sf(-(x-mu)/sigma, beta, m)
  def _sf(self, x): return crystalball.sf(x, self.mu, self.sigma, self.beta, self.m)

  @staticmethod
  def func(x, n=1, mu=0, sigma=1, beta=1, m=2): return n*crystalball.pdf(x, mu, sigma, beta, m)
  def _func(self, x): return crystalball.func(x, self.n, self.mu, self.sigma, self.beta, self.m)
  
  @staticmethod
  def best(x, y):
    fit_peak = gaussian.fit(x, y, peak=True)
    n, mu, sigma = fit_peak.n, fit_peak.mu, fit_peak.sigma
    func = lambda n : np.sum((crystalball.func(x, n, mu, sigma, beta=1, m=2)-y)**2)
    n = fmin(func, x0=n, maxiter=10, full_output=False, disp=False)[0]
    return n, mu, sigma, 1.0, 2.0


class linear(Function):
  def __init__(self, x=np.array([0]), c0=1, c1=1, **kwargs):
    super().__init__(x, dict(c0=c0, c1=c1), **kwargs)

  def __str__(self):
    if not hasattr(self,"spec"):
      fvar = lambda v : str(v)
    else: 
      fvar = lambda v : f'{v:{self.spec}}'
    return f"${fvar(self.c1)} x + {fvar(self.c0)}$"

  @staticmethod
  def func(x, c0=1, c1=1): return c1*x + c0 
  def _func(self, x): return linear.func(x, self.c0, self.c1)


class quadratic(Function):
  def __init__(self, x=np.array([0]), c0=1, c1=1, c2=1, **kwargs):
    super().__init__(x, dict(c0=c0, c1=c1, c2=c2), **kwargs)

  @staticmethod
  def func(x, c0=1, c1=1, c2=1): return c2*x*x + c1*x + c0 
  def _func(self, x): return quadratic.func(x, self.c0, self.c1, self.c2)

class exponential(Function):
  def __init__(self, x=np.array([0]), a=1, b=1, **kwargs):
    super().__init__(x, dict(a=a, b=b), **kwargs)

  def __str__(self):
    if not hasattr(self,"spec"):
      fvar = lambda v : str(v)
    else: 
      fvar = lambda v : f'{v:{self.spec}}'
    return f"${fvar(self.a)} * \exp( {fvar(self.b)} * x )$"

  @staticmethod
  def func(x, a=1, b=1): return a*np.exp(b*x)
  def _func(self, x): return exponential.func(x, self.a, self.b)

class tanh(Function):
  def __init__(self, x=np.array([0]), a=0, **kwargs):
    super().__init__(x, dict(a=a), **kwargs)
    
  def __str__(self):
    if not hasattr(self,"spec"):
      fvar = lambda v : str(v)
    else: 
      fvar = lambda v : f'{v:{self.spec}}'
    return f"$\\tanh( {fvar(self.a)} * x )$"
  
  @staticmethod
  def func(x, a=1): return np.tanh(a*x)
  def _func(self, x): return tanh.func(x, self.a)

  @staticmethod
  def best(x, y):
    return 1/np.mean(x)

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
