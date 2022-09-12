import numpy as np
import jax
import pyhf
from pyhf.exceptions import FailedMinimization
pyhf.set_backend('jax')

from .histogram import Stack

class Model:
  def __init__(self, h_sig, h_bkg, h_data=None):
    if isinstance(h_bkg, Stack): h_bkg = h_bkg.get_histo()
    if isinstance(h_bkg, list): h_bkg = h_bkg[0]

    self.h_sig = h_sig
    self.h_bkg = h_bkg
    self.h_data = h_bkg if h_data is None else h_data

    self.norm = 2*np.sqrt(np.sum(h_bkg.error**2))/h_sig.stats.nevents
    # self.norm = 1

    self.w = pyhf.simplemodels.uncorrelated_background(
      signal=(self.norm*h_sig.histo).tolist(), bkg=h_bkg.histo.tolist(), bkg_uncertainty=h_bkg.error.tolist()
    )
    self.data = self.h_data.histo.tolist()+self.w.config.auxdata

  def upperlimit(self, poi=np.linspace(0,2,21), level=0.05):
    try:
      obs_limit, exp_limit = pyhf.infer.intervals.upperlimit(
          self.data, self.w, poi, level=level,
      )
    except FailedMinimization:
      obs_limit, exp_limit = np.nan, np.array(5*[np.nan])
    obs_limit, exp_limit = self.norm*obs_limit, [ self.norm*lim for lim in exp_limit ]
    self.h_sig.stats.obs_limit, self.h_sig.stats.exp_limits = obs_limit, exp_limit
    return obs_limit, exp_limit

  def export_to_root(self, saveas="test.root"):
    from array import array
    import ROOT
    ROOT.gROOT.SetBatch(True)

    def to_th1d(histo, name=None, title=None):
        if name is None: name = histo.label
        if title is None: title = ""

        th1d = ROOT.TH1D(name, title, len(histo.bins)-1, array('d', histo.bins))
        for i, (n, e) in enumerate( zip(histo.histo,histo.error) ):
            th1d.SetBinContent(i+1, n)
            th1d.SetBinError(i+1, e)
        return th1d

    tfile = ROOT.TFile(saveas, "recreate")
    tfile.cd()

    t_data = to_th1d(self.h_data,"data_obs",";;Events")
    t_bkg = to_th1d(self.h_bkg, "bkg",";;Events")
    t_sig = to_th1d(self.h_sig, "nmssm",";;Events")

    print(f"Norm: {self.norm}")
        
    t_data.Write()
    t_bkg.Write()
    t_sig.Write()
    tfile.Close()



