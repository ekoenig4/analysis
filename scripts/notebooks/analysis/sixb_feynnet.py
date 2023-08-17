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

def main():
    notebook = FeynNet.from_parser()
    notebook.hello()
    notebook.run()


def sr(t):
    return ak.sum( np.abs(t.h_m - 125)**2, axis=1 ) < 25

varinfo.HX_m = dict(bins=(0,300,30))
varinfo.H1_m = dict(bins=(0,300,30))
varinfo.H2_m = dict(bins=(0,300,30))

varinfo.Y_m = dict(bins=(0,1500,30))
varinfo.X_m = dict(bins=(0,2000,30))

re_mxmy = re.compile(r'.*MX-(\d+)_MY-(\d+).*')

def get_mxmy(fname):
    m = re_mxmy.match(fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

class FeynNet(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--dout', default='sixb_feynnet')
        # parser.add_argument('--model', default='bkg0.25_hm10')
        parser.add_argument('--model', default='ranker')
        parser.add_argument('--serial', action='store_true')
        return parser
 
    @required
    def init(self):
        self.dout = f'{self.dout}/{self.model}'
        # self.modelpath = '/eos/uscms/store/user/ekoenig/weaver/analysis/models/exp_sixb_diff_aggr/feynnet_ranker_6b/20230728_59b53a4bde5e7da6eb8e6aa522b30859_ranger_lr0.0047_batch2000_withbkg'
        self.modelpath= '/eos/uscms/store/user/ekoenig/weaver/models/exp_sixb_official/feynnet_ranker_6b/20230731_7d266883bbfb88fe4e226783a7d1c9db_ranger_lr0.0047_batch2000_withbkg/'
        # self.modelpath= '/eos/uscms/store/user/ekoenig/weaver/models/exp_sixb_megamind/feynnet_ranker_6b/20230803_495f81bce0c466c3345918572c4e0906_ranger_lr0.0047_batch2000_withbkg'

        fc.sixb = fc.FileCollection('/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/')
        f_signal = np.array([ f'{d}/ntuple.root' for d in fc.sixb.Official_NMSSM.ls if d.endswith('pythia8') ])

        # 210 GeV mass points
        masses = np.array(list(map(get_mxmy, f_signal)))

        # sort by MX, then MY
        order = np.lexsort((masses[:,1],masses[:,0]))
        f_signal = f_signal[order]
        masses = masses[order]

        mask = np.ones(len(f_signal), dtype=bool)
        # remove all 50 GeV mass points
        # mask &= np.all(( (masses / 100) - (masses // 100) ) == 0, axis=1)
        # remove all points above 1400 GeV
        mask &= masses[:,0] <= 1400
        # left with ~ 53 points

        f_signal = list(f_signal[mask])
        self.use_signal = [ (i+1) * (len(f_signal) // 5) for i in range(3) ]

        f_signal = [ f_signal[i] for i in self.use_signal ]
        self.use_signal = [0,1,2]

        f_qcd = fc.sixb.QCD_B_List
        f_tt = list(map(lambda f : f.replace('ntuple_0','ntuple'), fc.sixb.TTJets))

        # %%
        self.signal = ObjIter([Tree([f], report=False) for f in tqdm(f_signal)])
        self.bkg = ObjIter([Tree(f_qcd), Tree(f_tt)])
        # self.bkg = ObjIter([])


    @required
    def load_feynnet(self, signal, bkg):
        print(self.modelpath)
        load_feynnet = sixb.f_load_feynnet_assignment( self.modelpath, onnx=True, order='random' )

        if not self.serial:
            import multiprocess as mp
            with mp.Pool(8) as pool:
                (signal+bkg).parallel_apply(load_feynnet, report=True, pool=pool)
        else:
            (signal+bkg).apply(load_feynnet, report=True)

        (signal+bkg).apply(sixb.assign)

    def _signal_kinematics(self, signal):
        fig, ax = study.get_figax(size=(10,8))
        study.mxmy_phase(
            signal,
            label=signal.mass.list,
            zlabel='Fraction HX with highest $P_T$',
            efficiency=True,

            title='GEN-Matched Events',
            f_var=lambda t: ak.mean( ak.argmax(t.h_pt[t.nfound_select==6],axis=1) ==0),
            g_cmap='jet',

            xlabel='$M_X$ (GeV)',
            ylabel='$M_Y$ (GeV)',

            # xlim=(550,1250),
            # ylim=(200,650),
            zlim=np.linspace(0,1,11),

            figax=(fig,ax),
            saveas=f'{self.dout}/signal_hx_highestpt_2d.png'
        )

    def feynnet_signal_efficiency(self, signal):
        fig, ax = study.get_figax(size=(10,8))

        study.mxmy_phase(
            signal,
            label=signal.mass.list,
            zlabel='Reconstruction Efficiency',
            efficiency=True,

            title='GEN-Matched Events',
            f_var=lambda t: ak.mean(t.x_signalId[t.nfound_select==6]==0),
            g_cmap='jet',

            xlabel='$M_X$ (GeV)',
            ylabel='$M_Y$ (GeV)',

            # xlim=(550,1250),
            # ylim=(200,650),
            zlim=np.linspace(0,1,11),

            figax=(fig,ax),
            saveas=f'{self.dout}/signal_efficiency_2d.png'
        )

        fig, ax = study.get_figax(size=(10,8))
        study.mxmy_phase(
            signal,
            label=signal.mass.list,
            zlabel='Higgs Purity',
            efficiency=True,

            title=r'Higgs Purity = $\frac{N Correct Higgs Paired}{N True Higgs}$',
            f_var=lambda t: ak.sum(t.h_signalId != -1) / ak.sum(t.nfound_presel_h),
            g_cmap='jet',

            xlabel='$M_X$ (GeV)',
            ylabel='$M_Y$ (GeV)',

            # xlim=(550,1250),
            # ylim=(200,650),
            zlim=np.linspace(0,1,11),

            figax=(fig,ax),
            saveas=f'{self.dout}/signal_higgs_purity_2d.png'
        )

    def feynnet_sr_soverb(self, signal, bkg):

        bkg_yield = bkg.apply(lambda t : ak.sum( t.scale[sr(t)] )).npy.sum()

        def soverb(t):
            sig_yield = ak.sum( t.scale[sr(t)] )
            return sig_yield / bkg_yield

        fig, ax = study.get_figax(size=(10,8))
        study.mxmy_phase(
            signal,
            label=signal.mass.list,
            zlabel='S/B',
            efficiency=True,

            title=r'SR Signal / Background$',
            f_var=soverb,
            g_cmap='jet',

            xlabel='$M_X$ (GeV)',
            ylabel='$M_Y$ (GeV)',

            # xlim=(550,1250),
            # ylim=(200,650),
            # zlim=np.linspace(0,1,11),

            figax=(fig,ax),
            saveas=f'{self.dout}/signal_sr_soverb.png'
        )

    def feynnet_signal_ymres(self, signal):

        genmatched = lambda t : t.nfound_select==6
        not_genmatched = lambda t : t.nfound_select!=6

        def get_ymres(t, mask):
            mask = mask(t)
            ymres = t.Y_m/t.gen_Y_m
            return ak.mean(ymres[mask])

        fig, ax = study.get_figax(size=(10,8))
        study.mxmy_phase(
            signal,
            label=signal.mass.list,
            zlabel='Reconstruction Efficiency',
            efficiency=True,

            title='GEN-Matched Events',
            f_var=lambda t : get_ymres(t, genmatched),
            g_cmap='bwr',

            xlabel='$M_X$ (GeV)',
            ylabel='$M_Y$ (GeV)',

            # xlim=(550,1250),
            # ylim=(200,650),
            zlim=np.linspace(0,2,11),

            figax=(fig,ax),
            saveas=f'{self.dout}/signal_genmatched_ymres.png'
        )

        fig, ax = study.get_figax(size=(10,8))
        study.mxmy_phase(
            signal,
            label=signal.mass.list,
            zlabel='Reconstruction Efficiency',
            efficiency=True,

            title='GEN-Matched Events',
            f_var=lambda t : get_ymres(t, not_genmatched),
            g_cmap='bwr',

            xlabel='$M_X$ (GeV)',
            ylabel='$M_Y$ (GeV)',

            # xlim=(550,1250),
            # ylim=(200,650),
            zlim=np.linspace(0,2,11),

            figax=(fig,ax),
            saveas=f'{self.dout}/signal_not_genmatched_ymres.png'
        )

    def plot_signal_mcbkg(self, signal, bkg):
        self.bkg_histos = obj_store()
        study.quick(
            signal[self.use_signal]+bkg,
            legend=True,
            varlist=['X_m','Y_m',None,'HX_m','H1_m','H2_m'],
            efficiency=True,
            store=self.bkg_histos,
            saveas=f'{self.dout}/signal_mcbkg',
        )

        self.bkg_histos = [ fig[0] for fig in self.bkg_histos ]

        study.quick(
            signal[self.use_signal]+bkg,
            masks=sr,
            legend=True,
            varlist=['X_m','Y_m',None,'HX_m','H1_m','H2_m'],
            efficiency=True,
            saveas=f'{self.dout}/signal_sr_mcbkg',
        )

    class _plot_signal_tree(ParallelMethod):
        def __init__(self, bkg_histos, dout, feynreco=True):
            super().__init__()
            self.bkg_histos = bkg_histos
            self.keys = ['X_m','Y_m',None,'HX_m','H1_m','H2_m']
            self.dout = dout
            self.feynreco = feynreco

        def start(self, tree):
            tree_attrs = {
                key : tree[key] for key in (self.keys+['scale']) if key is not None
            }

            if self.feynreco:
                mask = tree['x_signalId'] == 0
                tree_attrs = { key : tree[key][mask] for key in tree_attrs }

            tree_attrs['sample'] = tree.sample

            pkg = dict(
                keys=self.keys,
                tree=tree_attrs,
                bkg_histos=self.bkg_histos,
                dout=self.dout,
                feynreco=self.feynreco,
            )
            return pkg
        
        def run(self, keys, tree, bkg_histos, dout, feynreco):
            fig, axs = study.get_figax(nvar=5)
            ip = 0
            for i, key in enumerate(keys):
                ax = axs.flat[i]
                if key is None:
                    ax.set_visible(False)
                    continue
                bkg = bkg_histos[ip]
                plot_stack(bkg, figax=(fig, ax))
                histo_array(tree[key], bkg.bins, 
                            weights=tree['scale'], efficiency=True, 
                            h_label=tree['sample'], h_color='red',
                            h_histtype='step', figax=(fig, ax))
                ax.set(xlabel=key)
                ax.legend()
                ip += 1
            fig.suptitle(tree['sample'])
            fig.tight_layout()

            if feynreco:
                study.save_fig(fig, f'{dout}/mass_feynreco/{tree["sample"]}_m.png')
            else:
                study.save_fig(fig, f'{dout}/mass/{tree["sample"]}_m.png')

    @dependency(plot_signal_mcbkg)
    def plot_signal_mass(self, signal, bkg_histos):
        import dill
        dill.settings['recurse'] = True

        plot_signal = self._plot_signal_tree(bkg_histos, self.dout, feynreco=False)

        if not self.serial:
            import multiprocess as mp
            with mp.Pool( min(8, len(signal)) ) as pool:
                (signal).parallel_apply(plot_signal, report=True, pool=pool)
        else:
            (signal).apply(plot_signal, report=True)


if __name__ == '__main__':
    main()
