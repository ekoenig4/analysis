from .. import *
from .. import eightbUtils as eightb

class reconstruct_resonances(Analysis):
    @staticmethod
    def _add_parser(parser):
        parser.add_argument("--outfile", required=True,
                            help="output file pattern to write file with. Use {base} to substitute existing file")
        parser.add_argument("--reco-algo", choices=["ranker","minmass"],
                            help="type of reconstruction method to apply")

        # path = '/uscms_data/d3/ekoenig/8BAnalysis/studies/weaver-multiH/weaver/models'
        # model_paths = dict(
        #     pn=f'{path}/quadh_ranker/20221115_ranger_lr0.0047_batch512_m7m10m12/',

        #     mp=f'{path}/quadh_ranker_mp/20221124_ranger_lr0.0047_batch512_m7m10m12/',
        #     mp300k=f'{path}/quadh_ranker_mp/20221205_ranger_lr0.0047_batch512_m7m10m12_300k/',
        #     mp500k=f'{path}/quadh_ranker_mp/20221205_ranger_lr0.0047_batch512_m7m10m12_500k/',

        #     mpbkg00=f'{path}/quadh_ranker_mp/20221209_b72001172c5d04183ed7bb294252320b_ranger_lr0.0047_batch1024_m7m10m12_withbkg/',
        #     mpbkg005=f'{path}/quadh_ranker_mp/20221212_293790a7fbfb752ded05771058bf5a25_ranger_lr0.0047_batch1024_m7m10m12_withbkg/',
        #     mpbkg01=f'{path}/quadh_ranker_mp/20221209_be9efb5b61eb1c42aeb209728eec84d7_ranger_lr0.0047_batch1024_m7m10m12_withbkg/',

        #     # mpbkg01_hard25=f'{path}/quadh_ranker_mp/20221214_d595a9703289900d701416bb7274ab71_ranger_lr0.0047_batch1024_m7m10m12_withbkg/',
        #     # mpbkg01_hard50=f'{path}/quadh_ranker_mp/20221214_13676d884fa50cdaffb748fc057f180a_ranger_lr0.0047_batch1024_m7m10m12_withbkg/',

        #     # mpbkg35_hard25=f'{path}/quadh_ranker_mp/20221215_8d087d23e1f72729bdcdd043b3d693e6_ranger_lr0.0047_batch1024_m7m10m12_withbkg/',
        #     # mpbkg01_hard50=f'{path}/quadh_ranker_mp/20221215_13676d884fa50cdaffb748fc057f180a_ranger_lr0.0047_batch1024_m7m10m12_withbkg/',
        #     mpbkg01_hard50=f'{path}/quadh_ranker_mp/20221218_dbe056a55e82ce1d89e004942c741bb3_ranger_lr0.0047_batch1024_m7m10m12_withbkg/',

        #     # mpbkg01_exp=f'{path}/quadh_ranker_mp/20221214_2f889467cb0f6c7a9269c92e93c25c1d_ranger_lr0.0047_batch1024_m7m10m12_withbkg/',
        #     # mpbkg05_exp=f'{path}/quadh_ranker_mp/20221214_34452fc51690ae1d20a150a10c0bafa7_ranger_lr0.0047_batch1024_m7m10m12_withbkg/',
        # )
        parser.add_argument("--model-path", type=lambda f : eightb.models.get_model(f),
                            help="weaver model path for gnn reconstruction")
        return parser

    def _select_t8btag(self, signal, bkg, data):
        def n_presel_jets(t):
            t.extend(n_presel_jet=t.n_jet)
        (signal+bkg+data).apply(n_presel_jets)

        t8btag = CollectionFilter('jet', filter=lambda t: ak_rank(-t.jet_btag, axis=-1) < 8)
        self.signal = signal.apply(t8btag)
        self.bkg = bkg.apply(t8btag)
        self.data = data.apply(t8btag)

    def _load_ranker(self, signal, bkg, data):

        if self.model_path.load is None:
            raise ValueError("Unknown model arch")

        if self.model_path.load == 'quadh_ranker':
            def load(t):
                eightb.load_quadh(t, self.model_path.path)
                eightb.pair_y_from_higgs(t, operator=eightb.y_min_mass_asym)
        elif any( self.model_path.load == model for model in ('yy_4h_reco_ranker','feynnet_x_yy_4h_8b') ):
            load = lambda t : eightb.load_yy_quadh_ranker(t, self.model_path.path)

        (signal+bkg+data).apply(load, report=True)

        def nfound_higgs(t):
            nhiggs = ak.sum(t.higgs_signalId>-1,axis=-1)
            t.extend(nfound_paired_h=nhiggs)
        signal.apply(nfound_higgs)

    def _load_min_mass_asym(self, signal, bkg, data):
        (signal+bkg+data).apply(lambda t : build_collection(t, 'H\dY\d', 'higgs', ordered='pt'))
        def build_dr(t):
            b1_p4 = build_p4(t, 'higgs_b1')
            b2_p4 = build_p4(t, 'higgs_b2')

            h1y1_p4 = build_p4(t, 'H1Y1')
            h2y1_p4 = build_p4(t, 'H2Y1')
            
            h1y2_p4 = build_p4(t, 'H1Y2')
            h2y2_p4 = build_p4(t, 'H2Y2')

            t.extend(
                higgs_jet_dr = calc_dr_p4(b1_p4, b2_p4),
                Y1_higgs_dr = calc_dr_p4(h1y1_p4, h2y1_p4),
                Y2_higgs_dr = calc_dr_p4(h1y2_p4, h2y2_p4),
            )
        (signal+bkg+data).apply(build_dr)

    def reconstruct_quadh(self):
        algo = dict(
            ranker=self._load_ranker,
            minmass=self._load_min_mass_asym,
        ).get( self.reco_algo )
        algo(self.signal, self.bkg, self.data)

    def build_bdt_features(self, signal, bkg, data):
        def _build_bdt_features(t):
            jet_ht = ak.sum(t.jet_ptRegressed,axis=-1)

            j1_phi, j2_phi = ak.unzip(ak.combinations(t.jet_phi, n=2, axis=-1))
            jet_dphi = calc_dphi(j1_phi, j2_phi)

            j1_eta, j2_eta = ak.unzip(ak.combinations(t.jet_eta, n=2, axis=-1))
            jet_deta = calc_deta(j1_eta, j2_eta)

            min_jet_deta = ak.min( np.abs(jet_deta), axis=-1)
            max_jet_deta = ak.max( np.abs(jet_deta), axis=-1)

            jet_dr = np.sqrt( jet_deta**2 + jet_dphi**2 )

            min_jet_dr = ak.min(jet_dr, axis=-1)
            max_jet_dr = ak.max(jet_dr, axis=-1)

            h1_phi, h2_phi = ak.unzip(ak.combinations(t.higgs_phi, n=2, axis=-1))
            higgs_dphi = np.abs(calc_dphi(h1_phi, h2_phi))

            h1_eta, h2_eta = ak.unzip(ak.combinations(t.higgs_eta, n=2, axis=-1))
            higgs_deta = np.abs(calc_deta(h1_eta, h2_eta))

            higgs_comb_id = ak.combinations( np.arange(4), n=2, axis=0).tolist()

            t.extend(
                jet_ht=jet_ht,
                min_jet_deta=min_jet_deta,
                max_jet_deta=max_jet_deta,
                min_jet_dr=min_jet_dr,
                max_jet_dr=max_jet_dr,
                **{
                    f'h{i+1}{j+1}_dphi':higgs_dphi[:,k]
                    for k, (i,j) in enumerate(higgs_comb_id)
                },
                **{
                    f'h{i+1}{j+1}_deta':higgs_deta[:,k]
                    for k, (i,j) in enumerate(higgs_comb_id)
                },
                **{
                    f'h{i+1}_{var}':t[f'higgs_{var}'][:,i]
                    for i in range(4)
                    for var in ('pt','jet_dr')
                },
            )
        (signal+bkg+data).apply(_build_bdt_features, report=True)
    
    def write_trees(self, signal, bkg, data):
        (signal+bkg+data).write(
            self.outfile
        )