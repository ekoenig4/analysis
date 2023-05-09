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

# %%
def main():
    notebook = FeynNetNotebook.from_parser()
    notebook.print_namespace()
    print(notebook)
    notebook.run()

class FeynNetNotebook(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument(f'--module', default='fc.eightb.feynnet', help='specify the file collection module to use for all samples')
        parser.add_argument("--dout", default='feynnet', help="output directory to write files to")
        parser.add_argument("--model", default="feynnet_sig_innersig", help="model to use for loading feynnet")
        parser.add_argument("--no-signal", action='store_true', help="skip signal loading", default=False)
        return parser

    @required
    def init_dout(self, dout, model):
        self.dout = os.path.join(dout, model)

    @required
    def init_module(self, module):
        def _module(mod):
            local = dict()
            exec(f"module = {mod}", globals(), local)
            return local['module']
        self.module = _module(module)

    @required
    def init_files(self, module):

        self.signal = ObjIter([])
        if not self.no_signal:
            self.signal = ObjIter([Tree(f, report=False, altfile='test_{base}') for f in tqdm(module.feynnet_signal_list)])
            self.unique_mx = np.unique(self.signal.mx.npy)
            self.unique_my = np.unique(self.signal.my.npy)

        self.bkg = ObjIter([Tree(module.Run2_UL18.QCD_B_List, altfile='test_{base}')])

# %%
    @required
    def load_feynnet(self, signal, bkg, model):
        model = eightb.models.get_model(model)
        (signal + bkg).apply( lambda t : eightb.load_yy_quadh_ranker(t, model.storage), report=True, parallel=True, thread_order=lambda thread : len(thread.obj) )

# %%
    def get_bkg_mx_vs_mh(self, bkg):
        def higgs_m_chi2(t):
            return np.sqrt( ak.sum( (t.higgs_m-125)**2, axis=1 ) )
        higgs_m_chi2.bins = (0,300,30)
        study.compare_masks(
            [],bkg,
            label=['Full','mD < 150','mD < 100','mD < 50'],
            masks =[None, lambda t : higgs_m_chi2(t) < 150, lambda t : higgs_m_chi2(t) < 100, lambda t : higgs_m_chi2(t) < 50],
            varlist=['X_m'],
            binlist=[(250,2000,20)],
            legend=True,

            ratio=True, r_inv=True, r_log=True, r_ylim=(1e-3,1.5),
            r_size="90%", 
            # efficiency=True,

            saveas=f'{self.dout}/bkg_mx_vs_dmh.png'
        )
# %%
    @required
    def split_signal(self, signal):
        self.eightb_signal = signal.apply(EventFilter('all_eightb', filter=lambda t : t.nfound_select==8))
        self.partial_signal = signal.apply(EventFilter('partial_eightb', filter=lambda t : t.nfound_select< 8))

# %%
    def get_1d_efficiency(self, eightb_signal):
        fig, ax = study.get_figax(size=(10,8))

        study.statsplot(
            eightb_signal,
            label=eightb_signal.mass.list,
            varlist=['reco_id'],
            xlabels=['Reconstruction Efficiency'],
            efficiency=True,

            stat=lambda h:h.histo[-1],
            stat_err=lambda h:h.error[-1],
            g_grid=True,

            g_exe=lambda graph, **kwargs: print( f'{graph.stats:0.3f}'),
            g_ylim=(0,1),
            g_legend=True,
            figax=(fig,ax),
            saveas=f'{self.dout}/signal_efficiency_1d.png'
        )

# %%
    def get_extrema_efficiency(self, eightb_signal):
        from utils.ak_tools import ak_argavg
        eightb_signal_eff = eightb_signal.apply( lambda t : ak.mean(t.reco_id == 0) ).npy
        self.extrema_signal = [eightb_signal_eff .argmin(), ak_argavg(eightb_signal_eff), eightb_signal_eff.argmax()]
        print(eightb_signal_eff[self.extrema_signal])
        # print('\n'.join( [ f'{eightb_signal[i].mass}: {eightb_signal_eff[i]:0.3f}' for i in self.extrema_signal ] ))

    @dependency(get_extrema_efficiency)
    def get_extrema_resolution(self, eightb_signal, extrema_signal):
        class gen_res(ObjTransform):
            def __init__(self, kin):
                self.kin = kin
            @property
            def xlabel(self): return f'{self.kin} Resolution'
            @property
            def bins(self): return (-2,2,30)
            def __call__(self, t):
                if self.kin is None: return
                return (t[self.kin]-t[f'gen_{self.kin}'])/t[f'gen_{self.kin}']

        for kin in ('pt','m'):
            study.quick(
                eightb_signal[extrema_signal],
                legend=True,
                h_label_stat='${stats.mean:0.2f}\pm{stats.stdv:0.2f}$',
                varlist=[ gen_res(f'{res}_{kin}') for res in ['Y1','Y2','H1Y1','H2Y1','H1Y2','H2Y2']],
                suptitle='FeynNet Signal Reconstruction',
                h_restrict=True,
                lumi=None,
                density=True,
                saveas=f'{self.dout}/signal_extrema_{kin}_resolution.png'
            )

# %%
    def get_2d_efficiency(self, eightb_signal):
        fig, ax = study.get_figax(size=(10,8))

        study.mxmy_phase(
            eightb_signal,
            label=eightb_signal.mass.list,
            zlabel='Reconstruction Efficiency',
            efficiency=True,

            f_var=lambda t: ak.mean(t.reco_id==0),
            g_cmap='jet',

            xlabel='$M_X$ (GeV)',
            ylabel='$M_Y$ (GeV)',

            # xlim=(550,1250),
            # ylim=(200,650),
            zlim=np.linspace(0,1,11),

            figax=(fig,ax),
            saveas=f'{self.dout}/signal_efficiency_2d.png'
        )



# %%
    @dependency(get_extrema_efficiency)
    def get_extrema_score(self, eightb_signal, extrema_signal, bkg):
        study.quick(
            eightb_signal[extrema_signal]+bkg,
            # legend=True,
            h_label_stat=None,
            varlist=['yy_quadh_score','yy_quadh_minscore'],
            xlabels=['FeynNet Max Score','FeynNet Min Score'],
            binlist=[(-0.5,1.5,30)]*2,
            suptitle='FeynNet Reconstruction',
            efficiency=True,
            lumi=None,
            saveas=f'{self.dout}/signal_extrema_score.png'
        )

# %%
    @dependency(get_extrema_efficiency)
    def get_extrema_mass(self, eightb_signal, extrema_signal, bkg):
        varinfo.X_m =   dict(bins=(400,2000,30), xlabel='$M_{X}$ (GeV)')
        varinfo.Y1_m =  dict(bins=(100,1000,30), xlabel='Leading Y Boson Mass (GeV)')
        varinfo.Y2_m =  dict(bins=(100,1000,30), xlabel='Subleading Y Boson Mass (GeV)')
        varinfo.H1Y1_m =   dict(bins=(0,300,30), xlabel='Leading Y\'s Leading Higgs Boson Mass (GeV)')
        varinfo.H2Y1_m =   dict(bins=(0,300,30), xlabel='Leading Y\'s Subleading Higgs Boson Mass (GeV)')
        varinfo.H1Y2_m =   dict(bins=(0,300,30), xlabel='Subleading Y\'s Leading Higgs Boson Mass (GeV)')
        varinfo.H2Y2_m =   dict(bins=(0,300,30), xlabel='Subleading Y\'s Subleading Higgs Boson Mass (GeV)')


        # %%
        study.quick(
            eightb_signal[extrema_signal]+bkg,
            legend=True,
            h_label_stat=None,
            varlist=[None,'Y1_m','Y2_m',None,'H1Y1_m','H2Y1_m','H1Y2_m','H2Y2_m'],
            suptitle='FeynNet Signal Reconstruction',
            lumi=None,
            density=True,
            saveas=f'{self.dout}/signal_extrema_mass.png'
        )

    @dependency(get_extrema_efficiency)
    def get_extrema_mass_full(self, signal, extrema_signal, bkg):
        varinfo.X_m =   dict(bins=(400,2000,30), xlabel='$M_{X}$ (GeV)')
        varinfo.Y1_m =  dict(bins=(100,1000,30), xlabel='Leading Y Boson Mass (GeV)')
        varinfo.Y2_m =  dict(bins=(100,1000,30), xlabel='Subleading Y Boson Mass (GeV)')
        varinfo.H1Y1_m =   dict(bins=(0,300,30), xlabel='Leading Y\'s Leading Higgs Boson Mass (GeV)')
        varinfo.H2Y1_m =   dict(bins=(0,300,30), xlabel='Leading Y\'s Subleading Higgs Boson Mass (GeV)')
        varinfo.H1Y2_m =   dict(bins=(0,300,30), xlabel='Subleading Y\'s Leading Higgs Boson Mass (GeV)')
        varinfo.H2Y2_m =   dict(bins=(0,300,30), xlabel='Subleading Y\'s Subleading Higgs Boson Mass (GeV)')


        # %%
        study.quick(
            signal[extrema_signal]+bkg,
            legend=True,
            h_label_stat=None,
            varlist=[None,'Y1_m','Y2_m',None,'H1Y1_m','H2Y1_m','H1Y2_m','H2Y2_m'],
            suptitle='FeynNet Signal Reconstruction',
            lumi=None,
            density=True,
            saveas=f'{self.dout}/full_signal_extrema_mass.png'
        )



if __name__ == '__main__':
    main()