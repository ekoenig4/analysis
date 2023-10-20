import numpy as np
import os

from .histogram import Stack
from ..classUtils import ObjIter, ParallelMethod

class f_pyhf_upperlimit(ParallelMethod):
    def __init__(self, poi=np.linspace(0,5,31), level=0.05):
        super().__init__()
        self.poi = poi
        self.level = level
  
    def start(self, model):
        w, data = model.get_pyhf()
        return dict(
            data=data,
            w=w,
            norm=model.norm,
            poi=self.poi,
            level=self.level,
        )
    
    def run(self, data, w, norm, poi, level):
      import pyhf, os
      from pyhf.exceptions import FailedMinimization
      try:
        obs_limit, exp_limit = pyhf.infer.intervals.upperlimit(
            data, w, poi, level=level,
        )
      except FailedMinimization:
        obs_limit, exp_limit = np.nan, np.array(5*[np.nan])

      norm_obs_limit, norm_exp_limit = obs_limit, [ lim for lim in exp_limit ]
      obs_limit, exp_limit = norm*obs_limit, [ norm*lim for lim in exp_limit ]

      return dict(
        obs_limit=obs_limit,
        exp_limit=exp_limit,
        norm_obs_limit=norm_obs_limit, 
        norm_exp_limit=norm_exp_limit,
      )
    
    def end(self, model, norm_obs_limit, norm_exp_limit, obs_limit, exp_limit):
      model.h_sig.stats.norm_obs_limit, model.h_sig.stats.norm_exp_limits = norm_obs_limit, norm_exp_limit
      model.h_sig.stats.obs_limit, model.h_sig.stats.exp_limits = obs_limit, exp_limit
      return obs_limit, exp_limit
    
class f_combine_upperlimit(ParallelMethod):
    def __init__(self, combine_path, datacard):
        super().__init__()
        self.combine_path = combine_path
        self.datacard = datacard

    def start(self, model):
        return dict(
            path=self.combine_path,
            datacard=self.datacard,
            model=model,
        )

    def run(self, path, datacard, model):
        import utils.combineUtils as combine
        with combine.Process(path, datacard, model) as cp:
            output = dict(
                norm_obs_limit=cp.norm_obs_limit,
                norm_exp_limit=cp.norm_exp_limit,

                obs_limit=cp.obs_limit,
                exp_limit=cp.exp_limit,
            )

        return output
    
    def end(self, model, norm_obs_limit, norm_exp_limit, obs_limit, exp_limit):
      model.h_sig.stats.norm_obs_limit, model.h_sig.stats.norm_exp_limits = norm_obs_limit, norm_exp_limit
      model.h_sig.stats.obs_limit, model.h_sig.stats.exp_limits = obs_limit, exp_limit
      return obs_limit, exp_limit


class Model:
  f_upperlimit = f_pyhf_upperlimit

  f_pyhf_upperlimit = f_pyhf_upperlimit
  f_combine_upperlimit = f_combine_upperlimit

  def __getstate__(self):
     return dict(
        h_sig=self.h_sig,
        h_bkg=self.h_bkg,
        h_data=self.h_data,
     )

  def __setstate__(self, state):
    self.__init__(**state)

  def __init__(self, h_sig, h_bkg, h_data=None):
    if isinstance(h_bkg, Stack): h_bkg = h_bkg.get_histo()
    if isinstance(h_bkg, list): h_bkg = h_bkg[0]

    self.h_sig = h_sig

    self.mx = self.my = None
    if isinstance(h_sig.is_signal, str):
      self.mx, self.my = list(map(int, h_sig.is_signal[1:-1].split(',')))

    self.h_bkg = h_bkg
    self.h_data = h_bkg if h_data is None else h_data

    self.norm = 2*np.sqrt(np.sum(h_bkg.error**2))/h_sig.stats.nevents
    # self.norm = 1

  def get_pyhf(self):
    import pyhf, os
    w = pyhf.simplemodels.uncorrelated_background(
      signal=(self.norm*self.h_sig.histo).tolist(), bkg=self.h_bkg.histo.tolist(), bkg_uncertainty=self.h_bkg.error.tolist()
    )
    
    data = self.h_data.histo.tolist()+w.config.auxdata

    return w, data

  # def upperlimit(self, poi=np.linspace(0,2,21), level=0.05):
  #   try:
  #     obs_limit, exp_limit = pyhf.infer.intervals.upperlimit(
  #         self.data, self.w, poi, level=level,
  #     )
  #   except FailedMinimization:
  #     obs_limit, exp_limit = np.nan, np.array(5*[np.nan])
  #   norm_obs_limit, norm_exp_limit = obs_limit, [ lim for lim in exp_limit ]
  #   obs_limit, exp_limit = self.norm*obs_limit, [ self.norm*lim for lim in exp_limit ]
  #   self.h_sig.stats.norm_obs_limit, self.h_sig.stats.norm_exp_limits = norm_obs_limit, norm_exp_limit
  #   self.h_sig.stats.obs_limit, self.h_sig.stats.exp_limits = obs_limit, exp_limit
  #   return obs_limit, exp_limit

  def export_to_root(self, saveas="test.root"):
    from array import array
    import ROOT
    ROOT.gROOT.SetBatch(True)


    def to_th1d(histo, name=None, title=None, norm=1):
        if name is None: name = histo.label
        if title is None: title = ""

        th1d = ROOT.TH1D(name, title, len(histo.bins)-1, array('d', histo.bins))
        for i, (n, e) in enumerate( zip(histo.histo,histo.error) ):
            th1d.SetBinContent(i+1, norm*n)
            th1d.SetBinError(i+1, norm*e)
        return th1d

    saveas = saveas.format(**vars(self))
    tfile = ROOT.TFile(saveas, "recreate")
    tfile.cd()

    t_data = to_th1d(self.h_data,"data_obs",";;Events")
    t_bkg = to_th1d(self.h_bkg, "bkg",";;Events")
    t_sig = to_th1d(self.h_sig, "sig",f"norm={self.norm};;Events", norm=self.norm)
        
    t_data.Write()
    t_bkg.Write()
    t_sig.Write()
    tfile.Close()



