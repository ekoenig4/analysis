from .. import *
from .. import eightbUtils as eightb

class reconstruct_resonances(Analysis):
    @staticmethod
    def _add_parser(parser):
        parser.add_argument("--outfile", required=True,
                            help="output file pattern to write file with. Use {base} to substitute existing file")
        parser.add_argument("--reco-algo", choices=["ranker","minmass"],
                            help="type of reconstruction method to apply")
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

        load = lambda t : eightb.load_yy_quadh_ranker(t, self.model_path.storage.replace('weaver/','weaver/analysis/'))

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