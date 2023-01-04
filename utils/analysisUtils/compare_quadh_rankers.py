from .. import *
from .. import eightbUtils as eightb

class compare_quadh_rankers(Analysis):
    @classmethod
    def _add_parser(self, parser):
        
        parser.add_argument('--dout', default='template',
                            help='specify directory to save plots into')
        parser.add_argument("--models", nargs="+",
                            help="models to compare")

        parser.set_defaults(
            use_signal='signal_list',
            no_data=True,
        )

        return parser

    def select_t8btag(self, signal, bkg, data):
        def n_presel_jets(t):
            t.extend(n_presel_jet=t.n_jet)
        (signal+bkg+data).apply(n_presel_jets)

        t8btag = CollectionFilter('jet', filter=lambda t: ak_rank(-t.jet_btag, axis=-1) < 8)
        self.signal = signal.apply(t8btag)
        self.bkg = bkg.apply(t8btag)
        self.data = data.apply(t8btag)

    def load_models(self, signal, bkg, models):
        def nfound_higgs(t):
            nhiggs = ak.sum(t.higgs_signalId>-1,axis=-1)
            t.extend(nfound_paired_h=nhiggs)

        class Network:
            def __init__(self, signal, bkg, quadh_path=None):
                self.signal = signal.copy()
                self.bkg = bkg.copy()

                self.model = getattr(eightb.models, quadh_path)

                (self.signal+self.bkg).apply(lambda t : eightb.load_quadh(t, self.model.path), report=True)
                self.signal.apply(nfound_higgs)

        self.models = {
            key : Network(signal, bkg, key)
            for key in models
        }

    def plot_fully_resolved_eff(self, signal, models):
        def hitk1(h):
            return f'{ak.mean( h.array == 4 ):0.2%}'

        fig, axs = study.get_figax(nvar=3, dim=-1)
        for i, sample in enumerate(signal):
            study.quick( 
                [ model.signal[i] for model in models.values() ],
                masks=lambda t : t.nfound_select_h==4,
                label = [ model for model in models ],
                text=(0.0, 1.0, sample.sample),
                text_style=dict(ha='left', va='bottom'),
                h_color=None,
                legend=True,
                efficiency=True,
                ylim=(0,0.85), grid=True,
                varlist=['nfound_paired_h'],
                xlabels=['N Higgs Paired'],
                h_label_stat=hitk1,
                ratio=True, r_inv=True, r_ylim=(0.7501,1.249),
                figax=(fig, axs.flat[i])
            )
        study.save_fig(fig, f'{self.dout}/fully_resolved_eff')

    def plot_t8btag_higgs_m(self, signal, bkg, models):
        fig, axs = study.get_figax(nvar=4*4, dim=(-1,4))

        kwargs = dict(
            label = [ model for model in models ],
            text_style=dict(ha='left', va='bottom'),
            h_color=None,
            legend=True,
            efficiency=True,
            ylim=(0,0.51), grid=True,
            varlist=[f'higgs_m[:,{i}]' for i in range(4)],
            binlist=[(0,500,30)]*4,
            ratio=True, r_inv=True,
        )

        for i, sample in enumerate(signal):
            study.quick( 
                [ model.signal[i] for model in models.values() ],
                text=(0.0, 1.0, sample.sample),
                figax=(fig, axs[:,i]),
                **kwargs
            )

        study.quick_region( 
                *[ model.bkg for model in models.values() ],
                text=(0.0, 1.0, 'MC-Bkg'),
                figax=(fig, axs[:,i+1]),
                **kwargs
            )

        study.save_fig(fig, f'{self.dout}/t8btag_higgs_m')

    def plot_model_higgs_m(self, models):
                
        kwargs = dict(
            text_style=dict(ha='left', va='bottom'),
            legend=True,
            efficiency=True,
            ylim=(0,0.51), grid=True,
            varlist=[f'higgs_m[:,{i}]' for i in range(4)],
            binlist=[(0,500,30)]*4,
            dim=-1,
        )

        for key, model in models.items():
            study.quick( 
                model.signal+model.bkg,
                **kwargs,
                saveas=f'{self.dout}/{key}_higgs_m'
            )

    def plot_model_higgs_m_2d(self, models):
                
        kwargs = dict(
            text_style=dict(ha='left', va='bottom'),
            xvarlist=['higgs_m[:,0]','higgs_m[:,0]','higgs_m[:,0]','higgs_m[:,1]','higgs_m[:,1]','higgs_m[:,2]',],
            yvarlist=['higgs_m[:,1]','higgs_m[:,2]','higgs_m[:,3]','higgs_m[:,2]','higgs_m[:,3]','higgs_m[:,3]',],
            binlist=[(0,500,30)]*12,
            scatter=True,
        )

        for key, model in models.items():
            study.quick2d( 
                model.signal+model.bkg,
                **kwargs,
                saveas=f'{self.dout}/{key}_higgs_m_2d'
            )
    



