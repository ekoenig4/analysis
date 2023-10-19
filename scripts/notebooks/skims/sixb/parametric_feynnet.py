import os
os.environ['KMP_WARNINGS'] = 'off'
import sys
import git

import uproot as ut
import awkward as ak
import numpy as np
import math
import vector
import sympy as sp

import re
from tqdm import tqdm
import timeit
import re

sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils import *

from utils.notebookUtils import Notebook, required, dependency

varinfo.HX_m = dict(bins=(0,300,30))
varinfo.H1_m = dict(bins=(0,300,30))
varinfo.H2_m = dict(bins=(0,300,30))

varinfo.Y_m = dict(bins=(250,1375,31))
varinfo.X_m = dict(bins=(375,1500,31))


def xyh_from_3h(tree, assignment):
    h_p4 = build_p4(tree, 'h', extra=['signalId'])
    assignment = ak.from_regular(assignment)

    h_p4 = h_p4[assignment]

    hx_p4 = h_p4[:,0]

    hy_p4 = h_p4[:,1:]
    hy_p4 = hy_p4[ ak.argsort(-hy_p4.pt, axis=1) ]

    h1_p4 = hy_p4[:,0]
    h2_p4 = hy_p4[:,1]

    y_p4 = h1_p4 + h2_p4
    x_p4 = hx_p4 + y_p4

    kins = ['pt','m','eta','phi']
    return dict(
        **{ f'X_{field}': getattr(x_p4, field) for field in kins },
        **{ f'Y_{field}': getattr(y_p4, field) for field in kins },
        **{ f'H1_{field}': getattr(h1_p4, field) for field in kins+['signalId'] },
        **{ f'H2_{field}': getattr(h2_p4, field) for field in kins+['signalId'] },
        **{ f'HX_{field}': getattr(hx_p4, field) for field in kins+['signalId'] },
    )


class FeynNet(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--dout', default='sixb/parametric')
        parser.add_argument('--model', required=True)
        parser.add_argument('signal', nargs='+')
        return parser
 
    @required
    def init(self):
        self.signal = ObjIter([ Tree([f], weights=['scale'], normalization=None, xsec=1) for f in self.signal ])

        f_bkg_model = '/cmsuf/data/store/user/ekoenig/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/JetHT_Data_UL/bkg_model_ntuple.root'
        self.bkg_model = ObjIter([Tree([f_bkg_model], weights=['scale'], normalization=None, is_model=True, sample='bkg model', color='lavender')])

        self.gen_X_m = self.signal.mx[0]
        self.gen_Y_m = self.signal.my[0]

        self.dout = os.path.join(self.dout, f'MX-{self.gen_X_m}_MY-{self.gen_Y_m}')
        if not os.path.exists(self.dout):
            os.makedirs(self.dout)

    @required
    def load_model(self, signal, bkg_model):
        self.kfold = sum( f.startswith('model') and f.endswith('.onnx') for f in os.listdir( os.path.join(self.model, 'onnx') ))
        (signal + bkg_model).apply(lambda t : t.extend(kfold=np.arange(len(t)) % self.kfold))

        import utils.weaverUtils as weaver

        for k in range(self.kfold):
            model = weaver.WeaverONNX(self.model, 'onnx', k=k)
            (signal+bkg_model).apply(lambda t : self._predict_tree(k, model, t))

    def _predict_tree(self, k, model, tree):
        features = get_collection(tree, 'h')
        features['h_sinphi'] = np.sin(features['h_phi'])
        features['h_cosphi'] = np.cos(features['h_phi'])
        features['gen_X_m'] = np.ones(len(tree)) * self.gen_X_m
        features['gen_Y_m'] = np.ones(len(tree)) * self.gen_Y_m

        results = model.predict(features)
        assignment = ak.from_regular(results['sorted_h_assignments'], axis=1)
        tree.extend(**{ f'k{k}_h_assignment': assignment })

    def model_validation(self, signal, bkg_model):
        sig_index = signal.kfold.cat
        sig_weights = signal.scale.cat
        sig_y_ms = [ signal.apply(lambda t : xyh_from_3h(t, t[f'k{i}_h_assignment'])['Y_m']).cat for i in range(self.kfold) ]

        bkg_index = bkg_model.kfold.cat
        bkg_weights = bkg_model.scale.cat
        bkg_y_ms = [ bkg_model.apply(lambda t : xyh_from_3h(t, t[f'k{i}_h_assignment'])['Y_m']).cat for i in range(self.kfold) ]

        for i in range(self.kfold):
                # trained discriminant in train/test samples
                    fig, ax, _ = hist_multi(
                        [sig_y_ms[i][sig_index != i], bkg_y_ms[i][bkg_index != i], sig_y_ms[i][sig_index == i], bkg_y_ms[i][bkg_index == i]],
                        weights = [sig_weights[sig_index != i], bkg_weights[bkg_index != i], sig_weights[sig_index == i], bkg_weights[bkg_index == i]],
                        bins=varinfo.Y_m.bins, stacked=False,
                        title=f'FeynNet (Mx={self.gen_X_m}, My={self.gen_Y_m}) (Fold {i})',
                        xlabel='$M_{Y} (GeV)$',
                        is_data = [False, False, True, True],
                        h_label=['S (Train)', 'B (Train)', 'S (Test)', 'B (Test)'],
                        h_color=['blue', 'red', 'blue', 'red'],
                        h_alpha=[0.5, 0.5, 1.0, 1.0],
                        h_histtype=['stepfilled', 'stepfilled', None, None],
                        efficiency=True, legend=True,

                        ratio=True, r_group=((0, 2), (1, 3)), r_ylabel='Test/Train',
                    )
                    ax.plot([self.gen_Y_m, self.gen_Y_m], [0, 1], color='black', linestyle='--', label='Truth')
                    study.save_fig(fig, f'{self.dout}/feynnet_fold{i}.png')

                    for j in range(i + 1, self.kfold):
                        fig, ax, _ = hist_multi(
                            [sig_y_ms[j][sig_index == j], bkg_y_ms[j][bkg_index == j], sig_y_ms[i][sig_index == i], bkg_y_ms[i][bkg_index == i]],
                            weights = [sig_weights[sig_index == j], bkg_weights[bkg_index == j], sig_weights[sig_index == i], bkg_weights[bkg_index == i]],
                            bins=varinfo.Y_m.bins, stacked=False,
                            is_data = [False, False, True, True],
                            title=f'FeynNet (Mx={self.gen_X_m}, My={self.gen_Y_m}) (Fold {i} vs {j})',
                            xlabel='$M_{Y} (GeV)$',
                            h_label=[f'S (F{j})', f'B (F{j})', f'S (F{i})', f'B (F{i})'],
                            h_color=['blue', 'red', 'blue', 'red'],
                            h_alpha=[0.5, 0.5, 1.0, 1.0],
                            h_histtype=['stepfilled', 'stepfilled', None, None],
                            efficiency=True, legend=True,

                            ratio=True, r_group=((0, 2), (1, 3)), r_ylabel=f'F{j}/F{i}',
                        )
                        ax.plot([self.gen_Y_m, self.gen_Y_m], [0, 1], color='black', linestyle='--', label='Truth')
                        study.save_fig(fig, f'{self.dout}/feynnet_fold{i}_vs_{j}.png')

    def reconstruct(self, signal, bkg_model):
        def _reconstruct(tree):
            assignment = ak.zeros_like(tree.h_pt, dtype=np.int32)
            for k in range(self.kfold):
                assignment = ak.where( tree.kfold == k, tree[f'k{k}_h_assignment'], assignment )
            tree.extend(**xyh_from_3h(tree, assignment))
        (signal + bkg_model).apply(_reconstruct)

    @dependency(reconstruct)
    def plot_masses(self, signal, bkg_model):
        study.quick(
            signal + bkg_model,
            legend=True,
            varlist=['X_m','Y_m'],
            efficiency=True,
            saveas=f'{self.dout}/mx_my',
        )
        study.quick2d(
            signal + bkg_model,
            legend=True,
            binlist=[(250,1375,51),(375,1500,51)],
            varlist=['X_m','Y_m'],
            saveas=f'{self.dout}/mx_my_2d',
        )

    @dependency(reconstruct)
    def mass_limits(self, signal, bkg_model):
        model = []
        study.quick(
            signal + bkg_model,
            legend=True,
            plot_scale=[0.1],
            ylim=(0, 120),
            varlist=['X_m'],
            l_store=model,
            limits=True,
            saveas=f'{self.dout}/mx_limits',
        )
        model = model[0][0]
        info = dict(
            sig_yield = np.sum(model.h_sig.histo),
            bkg_yield = np.sum(model.h_bkg.histo),
            exp_lim = model.h_sig.stats.exp_limits[2],
        )
        import pickle
        with open(f'{self.dout}/mycuts_limit_values.pkl', 'wb') as f:
            pickle.dump(info, f)

    @dependency(reconstruct)
    def mass_mycut_limits(self, signal, bkg_model):
        model = []

        my_mean = ak.mean(signal.Y_m.cat)
        my_std = ak.std(signal.Y_m.cat)

        study.quick(
            signal + bkg_model,
            masks=lambda t : abs(t.Y_m - my_mean)  < my_std,
            plot_scale=[0.1],
            ylim=(0, 120),
            legend=True,
            suptitle=f'$|M_Y - {my_mean:0.2f}| < {my_std:0.2f}$',
            varlist=['X_m'],
            l_store=model,
            limits=True,
            saveas=f'{self.dout}/mx_mycut_limits',
        )
         
        model = model[0][0]
        info = dict(
            sig_yield = np.sum(model.h_sig.histo),
            bkg_yield = np.sum(model.h_bkg.histo),
            exp_lim = model.h_sig.stats.exp_limits[2],
        )
        import pickle
        with open(f'{self.dout}/limit_values.pkl', 'wb') as f:
            pickle.dump(info, f)
         

    
if __name__ == '__main__':
    FeynNet.main()