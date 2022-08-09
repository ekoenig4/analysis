import numpy as np
import jax
import pyhf
pyhf.set_backend('jax')

from .histogram import Stack

class Model:
  def __init__(self, h_sig, h_bkg, h_data=None):
    if isinstance(h_bkg, Stack): h_bkg = h_bkg.get_histo()

    self.h_sig = h_sig
    self.h_bkg = h_bkg
    self.h_data = h_bkg if h_data is None else h_data

    self.w = pyhf.simplemodels.uncorrelated_background(
      signal=h_sig.histo.tolist(), bkg=h_bkg.histo.tolist(), bkg_uncertainty=h_bkg.error.tolist()
    )
    self.data = self.h_data.histo.tolist()+self.w.config.auxdata

  def upperlimit(self, poi=np.linspace(0,5,11), level=0.05):
    self.h_sig.stats.obs_limit, self.h_sig.stats.exp_limits = pyhf.infer.intervals.upperlimit(
        self.data, self.w, poi, level=level,
    )




