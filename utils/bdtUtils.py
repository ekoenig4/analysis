import numpy as np
import awkward as ak
from typing import Callable

from .classUtils import ObjIter, Tree
from .utils import ak_stack
from .config import GIT_WD
from .xsecUtils import lumiMap
from .plotUtils import HistoList, Correlation

from hep_ml import reweight

from sklearn.pipeline import Pipeline
from hep_ml.preprocessing import IronTransformer
from hep_ml.gradientboosting import UGradientBoostingClassifier, LogLossFunction

import pickle, os

class BDTReweighter:
  def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample':0.4}, n_folds=2, verbose=True, seed=1234, load=None):
    self.seed = seed

    reweighter_base = reweight.GBReweighter(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf,
        gb_args=gb_args)

    self.verbose = verbose
    self.reweighter = reweight.FoldingReweighter(reweighter_base, random_state=self.seed, n_folds=n_folds, verbose=False)
    self.k_factor = None 

  def __getstate__(self):
    return dict(
        seed = self.seed,
        reweighter = self.reweighter,
        k_factor = self.k_factor,
    )

  def __setstate__(self, state):
    self.seed = state['seed']
    self.reweighter = state['reweighter']
    self.k_factor = state['k_factor']

  def train(self, targ_x, targ_w, estm_x, estm_w, reweight=True):
    if self.verbose:
      print('... calculating k factor')
    self.k_factor = ak.sum(targ_w)/ak.sum(estm_w)
    if reweight:
      if self.verbose:
        print('... fitting reweighter')
      self.reweighter.fit(estm_x, targ_x, self.k_factor*estm_w, targ_w)

  def reweight(self, x, w):
    return self.reweighter.predict_weights(x, self.k_factor*w,lambda x: np.mean(x, axis=0))/w
  
  def reweight_error(self, x, w):
    return self.reweighter.predict_weights(x, self.k_factor*w,lambda x: np.std(x, axis=0))/w
  
  def scale(self, x, w):
    return self.k_factor*w/w

  def save(self, fname='bdt_reweighter.pkl', path=f'{GIT_WD}/models'):
      if not fname.endswith('.pkl'): fname += '.pkl'
      fname = os.path.join(path, fname)

      dirname = os.path.dirname(fname)
      if not os.path.exists(dirname): os.makedirs(dirname)

      print(f'... saving bdt to {fname}')

      with open(fname, 'wb') as f:
         pickle.dump(self, f)

  @staticmethod
  def load(fname):
      with open(fname, 'rb') as f:
          return pickle.load(f)

class ABCD(BDTReweighter):
  def __init__(self, features: list = None, a: Callable = None, b: Callable = None, c: Callable = None, d: Callable = None, save=None, **kwargs):
    super().__init__(**kwargs)
    self.feature_names = features
    self.a, self.b, self.c, self.d = a, b, c, d
    self.sr = lambda t : self.a(t) | self.b(t)
    self.cr = lambda t : self.c(t) | self.d(t)
    self.tr = lambda t : self.a(t) | self.c(t)
    self.er = lambda t : self.b(t) | self.d(t)
    self.mask = lambda t: self.sr(t) | self.cr(t)

  def load(self, fname):
    with open(fname, 'rb') as f:
      self.__setstate__( pickle.load(f).__getstate__())
    self.hash = f"__abcd_reweight_{hash(self)}__{hash(self.reweighter)}__"
    self._trained_ = True

    return self

  def yields(self, treeiter : ObjIter, lumi=None):
    if not isinstance(treeiter, ObjIter): treeiter = ObjIter([treeiter])

    masks = { r: treeiter.apply( getattr(self, r) ).cat for r in ('a','b','c','d') }
    lumi = lumiMap[lumi][0]
    scale = lumi*treeiter.scale.cat
    yields = { r:ak.sum(scale[mask]) for r, mask in masks.items() }
    yields.update(total=ak.sum(scale))

    return yields

  def print_yields(self, treeiter : ObjIter, lumi=None, return_lines=False):
    if not isinstance(treeiter, ObjIter): treeiter = ObjIter([treeiter])
    yields = self.yields(treeiter, lumi=lumi)

    nevents=  yields['total']
    sr_total = sum( yields[r] for r in ('a','b') )
    cr_total = sum( yields[r] for r in ('c','d') )
    region_total = sr_total + cr_total

    a_total = f'{yields["a"]:00.4e}'.center(8)
    a_effic = f'{yields["a"]/nevents:00.2%}'.center(8)

    b_total = f'{yields["b"]:00.4e}'.center(8)
    b_effic = f'{yields["b"]/nevents:00.2%}'.center(8)

    c_total = f'{yields["c"]:00.4e}'.center(8)
    c_effic = f'{yields["c"]/nevents:00.2%}'.center(8)
    
    d_total = f'{yields["d"]:00.4e}'.center(8)
    d_effic = f'{yields["d"]/nevents:00.2%}'.center(8)

    sample = treeiter.sample.list

    if len(set(sample)) == 1:
      sample = sample[0]
    else:
      sample = 'MC-Bkg'

    lines = [
      f"--- ABCD {sample} Yields ---".center(48),
      f"Total: {region_total:0.2e} ({region_total/nevents:0.2%})".ljust(48),
      f"SR   : {sr_total:0.2e} ({sr_total/nevents:0.2%})".ljust(48),
      f"CR   : {cr_total:0.2e} ({cr_total/nevents:0.2%})".ljust(48),
      f"------------------------------------------------",
      f"|           A          |           B           |",
      f"|       {a_total}       |       {b_total}        |",
      f"|       {a_effic}       |       {b_effic}        |",
      f"------------------------------------------------",
      f"|           C          |           D           |",
      f"|       {c_total}       |       {d_total}        |",
      f"|       {c_effic}       |       {d_effic}        |",
      f"------------------------------------------------"]
    if return_lines:
      return lines
    print('\n'.join(lines))

  def get_features(self, treeiter: ObjIter, masks=None):
    if not isinstance(treeiter, ObjIter): treeiter = ObjIter([treeiter])

    def _ndim(array):
        if array.ndim == 2: return array
        if array.ndim == 1: return array[:,np.newaxis]

    def _get(treeiter, feature):
        if callable(feature):
            return _ndim(treeiter.apply(feature).cat)
        return _ndim(treeiter[feature].cat)

    features = [
        _get(treeiter, feature)
        for feature in self.feature_names
    ]
    X = ak.concatenate(features, axis=1)
    W = treeiter['scale'].cat
    
    if masks is not None:
      if not isinstance(masks, list): masks = [masks]
      masks = [ treeiter.apply(mask).cat for mask in masks ]
      return X, W, masks
    return X, W

  def train(self, treeiter: ObjIter, **kwargs):
    if hasattr(self, '_trained_'): return

    if self.verbose:
      print('... fetching features')
    X, W, (mask_c, mask_d) = self.get_features(treeiter, masks=[self.c, self.d])

    if self.verbose:
        print('... splitting features')
    c_X, c_W = X[mask_c], W[mask_c]
    d_X, d_W = X[mask_d], W[mask_d]

    super().train(c_X, c_W, d_X, d_W, **kwargs)

    self.hash = f"__abcd_reweight_{hash(self)}__{hash(self.reweighter)}__"
    self._trained_ = True


  def print_results(self, treeiter: ObjIter):
    results = self.results(treeiter)
    print(
      "--- ABCD Results ---\n"
      f"k = {results['k_factor']:0.3e}\n"
      f" (k*b)/a-1 = {results['k_factor_score']:0.2%}\n"
      f"BDT(b)/a-1 = {results['bdt_score']:0.2%}\n"
    )

  def results(self, treeiter: ObjIter):
    scale = treeiter.scale.cat
    reweight = treeiter.apply(self.reweight_tree).cat
    mask_a = treeiter.apply(self.a).cat
    mask_b = treeiter.apply(self.b).cat

    a_W, b_W = scale[mask_a], scale[mask_b]
    b_R = reweight[mask_b]

    a_T, b_T = [ np.sum(W) for W in (a_W, b_W,) ]
    b_R = np.sum(b_W*b_R)

    return dict(
      k_factor=self.k_factor,
      k_factor_score=self.k_factor*(b_T/a_T)-1,
      bdt_score=b_R/a_T-1
    )

  def reweight_tree(self, tree: Tree):
    if self.hash in tree.fields:
      return tree[self.hash]

    X, W = self.get_features(tree)
    reweight = self.reweight(X, W)
    reweight_error = self.reweight_error(X, W)

    tree.extend(**{self.hash:reweight, f'{self.hash}_error':reweight_error})
    return reweight
  
  def reweight_error_tree(self, tree: Tree):
    hash = f'{self.hash}_error'
    if hash in tree.fields:
      return tree[hash]

    X, W = self.get_features(tree)
    reweight_error = self.reweight_error(X, W)

    tree.extend(**{hash:reweight_error})
    return reweight_error

  def scale_tree(self, tree: Tree):
    X, W = self.get_features(tree)
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
    def __init__(self, features, scaler='iron', loss='log', n_estimators=200, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, seed=None, **kwargs):
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
                                                min_samples_leaf=min_samples_leaf,
                                                random_state=seed,
                                                ))
        ])

    def save(self, fname='bdt_classifier.pkl', path=f'{GIT_WD}/models'):
        if not fname.endswith('.pkl'): fname += '.pkl'
        fname = os.path.join(path, fname)

        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname): os.makedirs(dirname)

        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname, path=f'{GIT_WD}/models'):
        with open(fname, 'rb') as f:
            return pickle.load(f)


    def get_features(self, treeiter : ObjIter):
        if not isinstance(treeiter, ObjIter): treeiter = ObjIter([treeiter])


        def _ndim(array):
            if array.ndim == 2: return array
            if array.ndim == 1: return array[:,np.newaxis]

        def _get(treeiter, feature):
            if callable(feature):
                return _ndim(treeiter.apply(feature).cat)
            return _ndim(treeiter[feature].cat)

        features = [
            _get(treeiter, feature)
            for feature in self.feature_names
        ]
        
        X = ak.concatenate(features, axis=1)
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

        hs = HistoList.from_arrays([P_0,P_1], bins=(0,1,30), weights=[W_0,W_1])
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

class KFoldBDTClassifier(BDTClassifier):
    def __init__(self, features, kfold=2, seed=42069, **kwargs):
        self.feature_names = features
        self.kfold = kfold
        self.bdts = [BDTClassifier(features, seed=seed + i, **kwargs) for i in range(kfold)]

    def get_split(self, treeiter : ObjIter):
        if not isinstance(treeiter, ObjIter): treeiter = ObjIter([treeiter])
        index = treeiter.apply(lambda t : np.arange(len(t))).cat 
        return index % self.kfold

    def train(self, bkgiter : ObjIter, sigiter : ObjIter, parallel=False):
        X_0, W_0 = self.get_features(bkgiter)
        X_1, W_1 = self.get_features(sigiter)

        S_0 = self.get_split(bkgiter)
        S_1 = self.get_split(sigiter)

        X = np.concatenate([X_0, X_1])
        W = np.concatenate([W_0, W_1])
        Y = np.concatenate([np.zeros_like(W_0), np.ones_like(W_1)])
        S = np.concatenate([S_0, S_1])

        if parallel:
           return self.train_parallel(X, Y, W, S, njobs=parallel)

        for i, bdt in enumerate(self.bdts):
            bdt.classifier.fit(X[S != i], Y[S != i], bdt__sample_weight=W[S != i])

    @staticmethod
    def _train_parallel_(classifier, X, Y, W):
        classifier.fit(X, Y, bdt__sample_weight=W)
        return classifier
    
    def train_parallel(self, X, Y, W, S, njobs=None):
        if njobs is None or isinstance(njobs, bool): njobs = self.kfold

        import multiprocessing as mp
        from functools import partial
        from tqdm import tqdm
       
        with mp.Pool(njobs) as pool:
            results = []
            for i, bdt in enumerate(self.bdts):
                results.append(pool.apply_async(partial(self._train_parallel_, bdt.classifier, X[S != i], Y[S != i], W[S != i])))
            self.bdts = [result.get() for result in tqdm(results)]

    def predict_tree(self, treeiter : ObjIter):
        X, _ = self.get_features(treeiter)
        S = self.get_split(treeiter)

        P = np.zeros(len(X))
        for i, bdt in enumerate(self.bdts):
            P[S == i] = bdt.classifier.predict_proba(X[S == i])[:,1]
        return P
    
    def validate_tree(self, treeiter : ObjIter):
        X, _ = self.get_features(treeiter)
        S = self.get_split(treeiter)

        P = np.zeros(len(X))
        for i, bdt in enumerate(self.bdts):
            P[S != i] = bdt.classifier.predict_proba(X[S != i])[:,1]
        return P

    def results(self, bkgiter : ObjIter, sigiter : ObjIter):
        _, W_0 = self.get_features(bkgiter)
        _, W_1 = self.get_features(sigiter)

        P_0 = self.predict_tree(bkgiter)
        P_1 = self.predict_tree(sigiter)

        hs = HistoList.from_arrays([P_0,P_1], bins=(0,1,30), weights=[W_0,W_1])
        es = hs.ecdf(sf=True)
        roc = Correlation(es[0], es[1])

        return dict(
            auroc=roc.stats.area
        )
