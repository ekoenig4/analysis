import numpy as np
import awkward as ak
from typing import Callable

from .classUtils import ObjIter
from .utils import ak_stack

from hep_ml import reweight

class BDT:
  seed = 123456789
  def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=2, min_samples_leaf=100, gb_args={'subsample':0.4}, n_folds=2):
    np.random.seed(self.seed) #Fix any random seed using numpy arrays

    reweighter_base = reweight.GBReweighter(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf,
        gb_args=gb_args)

    self.reweighter = reweight.FoldingReweighter(reweighter_base, random_state=self.seed, n_folds=n_folds, verbose=False)
  
  def train(self, targ_x, targ_w, estm_x, estm_w):
    self.k_factor = ak.sum(targ_w)/ak.sum(estm_w)
    self.reweighter.fit(targ_x, estm_x, targ_w, self.k_factor*estm_w)

  def reweight(self, x, w):
    return self.reweighter.predict_weights(x,self.k_factor*w,lambda x: np.mean(x, axis=0))/w
  
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

  def train(self, treeiter: ObjIter):
    c_X, c_W = self.get_features(treeiter, self.c)
    d_X, d_W = self.get_features(treeiter, self.d)
    super().train(c_X, c_W, d_X, d_W)

  def results(self, treeiter: ObjIter):
    _, a_W = self.get_features(treeiter, self.a)
    b_X, b_W = self.get_features(treeiter, self.b)
    
    a_T, b_T = [ np.sum(W) for W in (a_W, b_W,) ]
    b_R = np.sum(b_W*self.reweight(b_X, b_W))

    print(
      "--- ABCD Results ---\n"
      f"k = {self.k_factor:0.3e}\n"
      f"k*(b/a)-1  = {self.k_factor*(b_T/a_T)-1:0.3e}\n"
      f"BDT(b)/a-1 = {b_R/a_T-1:0.3e}\n"
    )


  def reweight_tree(self, treeiter: ObjIter):
    X, W = self.get_features(treeiter)
    return self.reweight(X, W)

  def scale_tree(self, treeiter: ObjIter):
    X, W = self.get_features(treeiter)
    return self.scale(X, W)