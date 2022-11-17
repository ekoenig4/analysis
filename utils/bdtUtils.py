import numpy as np
import awkward as ak
from typing import Callable

from .classUtils import ObjIter
from .utils import ak_stack, GIT_WD
from .xsecUtils import lumiMap
from .plotUtils import HistoList, Correlation

from hep_ml import reweight

from sklearn.pipeline import Pipeline
from hep_ml.preprocessing import IronTransformer
from hep_ml.gradientboosting import UGradientBoostingClassifier, LogLossFunction

import pickle, os

class BDTReweighter:
  seed = 123456789
  def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample':0.4}, n_folds=2, verbose=False):
    np.random.seed(self.seed) #Fix any random seed using numpy arrays

    reweighter_base = reweight.GBReweighter(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf,
        gb_args=gb_args)

    self.reweighter = reweight.FoldingReweighter(reweighter_base, random_state=self.seed, n_folds=n_folds, verbose=verbose)

  def train(self, targ_x, targ_w, estm_x, estm_w, reweight=True):
    self.k_factor = ak.sum(targ_w)/ak.sum(estm_w)
    if reweight:
      self.reweighter.fit(estm_x, targ_x, self.k_factor*estm_w, targ_w)

  def reweight(self, x, w):
    return self.reweighter.predict_weights(x, self.k_factor*w,lambda x: np.mean(x, axis=0))/w
  
  def scale(self, x, w):
    return self.k_factor*w/w

class ABCD(BDTReweighter):
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

class DataMC_BDT(BDTReweighter):
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


class BDTClassifier:
    def __init__(self, features, scaler='iron', loss='log', n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, **kwargs):
        self.feature_names = features

        scaler = dict(
            iron=IronTransformer
        ).get(scaler)()

        loss = dict(
            log=LogLossFunction
        ).get(loss)()

        self.classifier = Pipeline([
            ('scaler', scaler),
            ('bdt', UGradientBoostingClassifier(loss=loss,
                                                n_estimators=n_estimators,
                                                learning_rate=learning_rate,
                                                max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf,))
        ])

    def save(self, fname='bdt_classifier.pkl', path=f'{GIT_WD}/models'):
        if not fname.endswith('.pkl'): fname += '.pkl'
        fname = os.path.join(path, fname)

        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname, path=f'{GIT_WD}/models'):
        if not fname.endswith('.pkl'): fname += '.pkl'
        fname = os.path.join(path, fname)

        with open(fname, 'rb') as f:
            return pickle.load(f)


    def get_features(self, treeiter : ObjIter):
        if not isinstance(treeiter, ObjIter): treeiter = ObjIter([treeiter])

        X = ak_stack([ treeiter[feature].cat for feature in self.feature_names ])
        W = treeiter['scale'].cat
        W = W/ak.mean(W)
        
        return X.to_numpy(), W.to_numpy()

    def train(self, bkgiter : ObjIter, sigiter : ObjIter):
        X_0, W_0 = self.get_features(bkgiter)
        X_1, W_1 = self.get_features(sigiter)

        X = np.concatenate([X_0, X_1])
        W = np.concatenate([W_0, W_1])
        Y = np.concatenate([np.zeros_like(W_0), np.ones_like(W_1)])

        self.classifier.fit(X, Y, bdt__sample_weight=W)

    def predict_tree(self, treeiter : ObjIter):
        X, _ = self.get_features(treeiter)
        return self.classifier.predict_proba(X)[:,1]

    def results(self, bkgiter : ObjIter, sigiter : ObjIter):
        X_0, W_0 = self.get_features(bkgiter)
        X_1, W_1 = self.get_features(sigiter)

        P_0 = self.classifier.predict_proba(X_0)[:,1]
        P_1 = self.classifier.predict_proba(X_1)[:,1]

        hs = HistoList([P_0,P_1], bins=(0,1,30), weights=[W_0,W_1])
        es = hs.ecdf(sf=True)
        roc = Correlation(es[0], es[1])

        return dict(
            auroc=roc.stats.area
        )

    def print_results(self, bkgiter : ObjIter, sigiter : ObjIter):
        results = self.results(bkgiter, sigiter)
        print(
            "--- BDT Classifier Results ---\n"
            f"AUROC = {results['auroc']:0.3f}"
        )
