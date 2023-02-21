from .. import *
from .. import eightbUtils as eightb

class weaver_input(Analysis):
    @staticmethod
    def _add_parser(parser):
        # parser.add_argument("--altfile", required=True,
        #                     help="output file pattern to write file with. Use {base} to substitute existing file")
        return parser

    def skim_fully_resolved(self, signal):
        fully_resolved = EventFilter('signal_fully_resolved', filter=lambda t: t.nfound_select==8)
        all_bjets = CollectionFilter('jet', filter=lambda t: t.jet_signalId > -1)

        filter = FilterSequence(
            fully_resolved, all_bjets
        )

        self.signal = signal.apply(filter)

    
    def select_t8btag(self, bkg):
        t8btag = CollectionFilter('jet', filter=lambda t: ak_rank(-t.jet_btag, axis=-1) < 8)
        self.bkg = bkg.apply(t8btag)

    # def calc_rescale(self, signal, bkg):
    #     sample_scale = {
    #         # 'MX_700_MY_300':1.2434550400752545e-18, 
    #         # 'MX_800_MY_300':1.6568129499652848e-18,
    #         # 'MX_800_MY_350': 1.921792885099276e-18,
    #         # 'MX_900_MY_300': 2.40911537095554e-18,
    #         # 'MX_900_MY_400': 3.494872548400683e-18,
    #         # 'MX_1000_MY_300': 3.551084678512766e-18,
    #         # 'MX_1000_MY_450':6.4806808031115614e-18, 
    #         # 'MX_1200_MY_500':1.343392630420647e-17, 
    #         # 'TTJets':7.294459735548297e-09
    #     }

    #     def rescale(t):
    #         norm = sample_scale.get(t.sample, None)
    #         if norm is None: return 
    #         norm /= t.filelist[0].scale 
    #         t.extend(scale = norm*t.scale)

    #     (signal + bkg).apply(rescale)

    def normalize_signal(self, signal):
        sample_scale = {
            'MX_700_MY_300':282.6071876702381, 
            'MX_800_MY_300': 225.12014605010498,
            'MX_800_MY_350': 207.70891386133388,
            'MX_900_MY_300': 200.58352182856032,
            'MX_900_MY_400': 167.88681160623509,
            'MX_1000_MY_300': 192.87950460596224,
            'MX_1000_MY_450':144.54467503648686, 
            'MX_1200_MY_500':103.01861324839004
        }
        def use_norm_signal(t):
            norm = sample_scale.get(t.sample, 1)
            t.extend(scale=norm*t.scale)
        signal.apply(use_norm_signal)

    def calc_abs_scale(self, signal, bkg):

        sample_norm = {
            'QCD':0.9990666979650883,
            'TTJets':0.22838519653423503,
        }

        def use_abs_scale(t):
            abs_norm = sample_norm.get(t.sample, 1)
            abs_scale = abs_norm * np.abs(t.scale)
            t.extend(abs_scale=abs_scale)

        (signal + bkg).apply(use_abs_scale)

    def calc_sample_norm(self, signal, bkg):
        signal_norm = {
            # True:0.3333333333333333,
            # True:0.125,
            False:0.04239411075548981
        }

        def use_sample_norm(t):
            norm = signal_norm.get(t.is_signal, 1)
            norm_abs_scale = norm*t.abs_scale
            t.extend(norm_abs_scale=norm_abs_scale)

        (signal+bkg).apply(use_sample_norm)

    def calc_dataset_norm(self, signal, bkg):
        max_norm = 806.3899452606947

        def use_dataset_norm(t):
            norm = max_norm 
            dataset_norm_abs_scale = norm * t.norm_abs_scale
            t.extend(dataset_norm_abs_scale=dataset_norm_abs_scale)
        (signal+bkg).apply(use_dataset_norm)

    def is_bkg(self, signal, bkg):
        signal.apply(lambda t : t.extend(is_bkg=ak.zeros_like(t.Run)))
        bkg.apply(lambda t : t.extend(is_bkg=ak.ones_like(t.Run)))
    
    def write_trees(self, signal, bkg, data):

        if any(signal.objs):
            (signal).write(
                'fully_res_{base}'
            )

        if any(bkg.objs):
            (bkg).write(
                'training_{base}'
            )