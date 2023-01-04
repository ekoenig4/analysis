from .. import *
from .. import eightbUtils as eightb

class compare_yy_quadh_rankers(Analysis):
    _requied_ = [
        'select_t8btag',
        'load_models'
    ]

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

                if self.model.load is None:
                    raise ValueError("Unknown model arch")

                if self.model.load == 'quadh_ranker':
                    def load(t):
                        eightb.load_quadh(t, self.model.path)
                        eightb.pair_y_from_higgs(t, operator=eightb.y_min_mass_asym)
                elif self.model.load == 'yy_4h_reco_ranker':
                    load = lambda t : eightb.load_yy_quadh_ranker(t, self.model.path)

                (self.signal+self.bkg).apply(load, report=True)
                self.signal.apply(nfound_higgs)
                self.best_quadh_dm(self.signal, self.bkg)
            def best_quadh_dm(self, signal, bkg):
                h4m = signal.higgs_m.apply(lambda m : m[:10000]).cat.to_numpy()
                from scipy import optimize

                def calc_dm(center):
                    n = len(center)
                    dm = np.sqrt( np.sum( (h4m[:,:n]-center)**2, axis=-1 ) )
                    return dm

                def _find_best_(center):
                    dm = calc_dm(center)
                    mask = dm < 30*np.sqrt(len(center))
                    return 1-np.mean(mask)
                    # mu = ak.mean(dm[dm < 100])
                    # return mu

                r0 = (125,125,125,125)
                center = optimize.fmin(_find_best_, r0,)
                print(center)
                def get_higgs_dm(t, center=center):
                    dm = [
                        np.abs(t.higgs_m[:,i]-m)
                        for i, m in enumerate(center)
                    ]  
                    dm = ak_stack(dm, axis=1)
                    t.extend(higgs_dm=dm)
                (signal+bkg).apply(get_higgs_dm)

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
    
    def plot_fully_resolved_mass(self, signal, models):

        fig, axs = study.get_figax(nvar=3*6, dim=(-1,3))
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
                ylim=(0,0.51), grid=True,
                varlist=['Y1_m','H1Y1_m','H2Y1_m','Y2_m','H1Y2_m','H2Y2_m'],
                binlist=[(0,1000,30),(0,500,30),(0,500,30)]*2,
                ratio=True, r_inv=True, r_ylim=(0.7501,1.249),
                figax=(fig, axs[:,i])
            )
        study.save_fig(fig, f'{self.dout}/fully_resolved_mass')

    def plot_t8btag_res_m(self, signal, bkg, models):
        fig, axs = study.get_figax(nvar=4*6, dim=(-1,4))

        kwargs = dict(
            label = [ model for model in models ],
            text_style=dict(ha='left', va='bottom'),
            h_color=None,
            legend=True,
            efficiency=True,
            ylim=(0,0.51), grid=True,
            varlist=['Y1_m','H1Y1_m','H2Y1_m','Y2_m','H1Y2_m','H2Y2_m'],
            binlist=[(0,1000,30),(0,500,30),(0,500,30)]*2,
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

        study.save_fig(fig, f'{self.dout}/t8btag_res_m')

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

    def plot_model_res_m(self, models):
                
        kwargs = dict(
            text_style=dict(ha='left', va='bottom'),
            legend=True,
            efficiency=True,
            ylim=(0,0.51), grid=True,
            varlist=['Y1_m','H1Y1_m','H2Y1_m','Y2_m','H1Y2_m','H2Y2_m'],
            binlist=[(0,1000,30),(0,500,30),(0,500,30)]*2,
            # dim=-1,
        )

        for key, model in models.items():
            study.quick( 
                model.signal+model.bkg,
                **kwargs,
                saveas=f'{self.dout}/{key}_res_m'
            )

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

    def plot_model_res_m_2d(self, models):
                
        kwargs = dict(
            text_style=dict(ha='left', va='bottom'),
            xvarlist=['Y1_m','higgs_m[:,0]','higgs_m[:,0]','higgs_m[:,0]','higgs_m[:,1]','higgs_m[:,1]','higgs_m[:,2]',],
            yvarlist=['Y2_m','higgs_m[:,1]','higgs_m[:,2]','higgs_m[:,3]','higgs_m[:,2]','higgs_m[:,3]','higgs_m[:,3]',],
            binlist=[(0,1000,30)]*2+[(0,500,30)]*12,
            scatter=True,
        )

        for key, model in models.items():
            study.quick2d( 
                model.signal+model.bkg,
                **kwargs,
                saveas=f'{self.dout}/{key}_res_m_2d'
            )
    
    def plot_model_quadh_dm(self, models):

        kwargs = dict(
            varlist=[lambda t : np.sqrt( ak.sum(t.higgs_dm**2,axis=-1) )],
            xlabels=['Quad Higgs $|\Delta M|$'],
            binlist=[(0,700,30)],
            legend=True, efficiency=True,
        )

        for key, model in models.items():
            study.quick( 
                model.signal+model.bkg,
                **kwargs,
                saveas=f'{self.dout}/{key}_quadh_dm'
            )
            
            study.pairplot(
                model.signal, 
                legend=True,
                efficiency=True,
                varlist=[f'higgs_dm[:,{i}]' for i in range(4)],
                binlist=[(0, 500, 30)]*4,
                saveas=f'{self.dout}/{key}_sig_h_dm_pairplot'
            )
            
            study.pairplot(
                model.bkg, 
                legend=True,
                efficiency=True,
                varlist=[f'higgs_dm[:,{i}]' for i in range(4)],
                binlist=[(0, 500, 30)]*4,
                saveas=f'{self.dout}/{key}_bkg_h_dm_pairplot'
            )




