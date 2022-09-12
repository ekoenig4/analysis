import numpy as np
import awkward as ak
from typing import Callable

from .classUtils import ObjIter
from .utils import ak_stack
from .xsecUtils import lumiMap

from hep_ml import reweight

class BDT:
  seed = 123456789
  def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample':0.4}, n_folds=2):
    np.random.seed(self.seed) #Fix any random seed using numpy arrays

    reweighter_base = reweight.GBReweighter(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf,
        gb_args=gb_args)

    self.reweighter = reweight.FoldingReweighter(reweighter_base, random_state=self.seed, n_folds=n_folds, verbose=False)

  def train(self, targ_x, targ_w, estm_x, estm_w, reweight=True):
    self.k_factor = ak.sum(targ_w)/ak.sum(estm_w)
    if reweight:
      self.reweighter.fit(estm_x, targ_x, self.k_factor*estm_w, targ_w)

  def reweight(self, x, w):
    return self.reweighter.predict_weights(x, self.k_factor*w,lambda x: np.mean(x, axis=0))/w
  
  def scale(self, x, w):
    return self.k_factor*w/w

class ABCD(BDT):
  def __init__(self, features: list = None, a: Callable = None, b: Callable = None, c: Callable = None, d: Callable = None, **kwargs):
    super().__init__(**kwargs)
    self.feature_names = features
    self.a, self.b, self.c, self.d = a, b, c, d

  def get_features(self, treeiter: ObjIter, mask=None):
    if not isinstance(treeiter, ObjIter): treeiter = ObjIter([treeiter])

    X = ak_stack([ treeiter[feature].cat for feature in self.feature_names ])
    W = treeiter['scale'].cat
    
    if mask is not None:
      mask = treeiter.apply(mask).cat   
      return X[mask], W[mask]
    return X, W

  def train(self, treeiter: ObjIter, **kwargs):
    c_X, c_W = self.get_features(treeiter, self.c)
    d_X, d_W = self.get_features(treeiter, self.d)
    super().train(c_X, c_W, d_X, d_W, **kwargs)

  def print_results(self, treeiter: ObjIter):
    results = self.results(treeiter)
    print(
      "--- ABCD Results ---\n"
      f"k = {results['k_factor']:0.3e}\n"
      f"k*(b/a)-1  = {results['k_factor_score']:0.2%}\n"
      f"BDT(b)/a-1 = {results['bdt_score']:0.2%}\n"
    )

  def results(self, treeiter: ObjIter):
    _, a_W = self.get_features(treeiter, self.a)
    b_X, b_W = self.get_features(treeiter, self.b)
    
    a_T, b_T = [ np.sum(W) for W in (a_W, b_W,) ]
    b_R = np.sum(b_W*self.reweight(b_X, b_W))

    return dict(
      k_factor=self.k_factor,
      k_factor_score=self.k_factor*(b_T/a_T)-1,
      bdt_score=b_R/a_T-1
    )


  def reweight_tree(self, treeiter: ObjIter):
    X, W = self.get_features(treeiter)
    return self.reweight(X, W)

  def scale_tree(self, treeiter: ObjIter):
    X, W = self.get_features(treeiter)
    return self.scale(X, W)

class DataMC_BDT(BDT):
  def __init__(self, features: list = None, a: Callable = None, b: Callable = None, **kwargs):
    super().__init__(**kwargs)
    self.feature_names = features
    self.a, self.b = a, b,

  def get_features(self, treeiter: ObjIter, mask=None, lumi=None):
    if not isinstance(treeiter, ObjIter): treeiter = ObjIter([treeiter])

    X = ak_stack([ treeiter[feature].cat for feature in self.feature_names ])
    W = treeiter['scale'].cat

    if lumi:
      W = lumi*W
    
    if mask is not None:
      mask = treeiter.apply(mask).cat   
      return X[mask], W[mask]
    return X, W

  def train(self, dataiter: ObjIter, mciter:ObjIter, **kwargs):
    data_X, data_W = self.get_features(dataiter, self.b)
    mc_X, mc_W = self.get_features(mciter, self.b, lumi=lumiMap[2018][0])
    super().train(data_X, data_W, mc_X, mc_W, **kwargs)

  def print_results(self, dataiter: ObjIter, mciter:ObjIter):
    results = self.results(dataiter, mciter)
    print(
      "--- Data MC Results ---\n"
      f"k = {results['k_factor']:0.3e}\n"
      f"k*(mc/data)-1  = {results['k_factor_score']:0.2%}\n"
      f"BDT(mc)/data-1 = {results['bdt_score']:0.2%}\n"
    )

  def results(self, dataiter: ObjIter, mciter:ObjIter):
    _, data_W = self.get_features(dataiter, self.a)
    mc_X, mc_W = self.get_features(mciter, self.a, lumi=lumiMap[2018][0])
    
    data_T, mc_T = [ np.sum(W) for W in (data_W, mc_W,) ]
    mc_R = np.sum(mc_W*self.reweight(mc_X, mc_W))

    return dict(
      k_factor=self.k_factor,
      k_factor_score=self.k_factor*(mc_T/data_T)-1,
      bdt_score=mc_R/data_T-1
    )


  def reweight_tree(self, treeiter: ObjIter):
    X, W = self.get_features(treeiter)
    return self.reweight(X, W)

  def scale_tree(self, treeiter: ObjIter):
    X, W = self.get_features(treeiter)
    return self.scale(X, W)
