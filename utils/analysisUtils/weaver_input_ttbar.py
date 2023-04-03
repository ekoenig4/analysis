from .. import *
from .. import eightbUtils as eightb

class weaver_input_ttbar(Analysis):
    @staticmethod
    def _add_parser(parser):
        # parser.add_argument("--altfile", required=True,
        #                     help="output file pattern to write file with. Use {base} to substitute existing file")
        return parser

    @required
    def init_ttbar(self):
        self.signal = ObjIter([ tree for tree in self.trees if tree.sample == 'TTJets' ])
        for s in self.signal:
            s.is_signal = True

        self.bkg = ObjIter([ tree for tree in self.trees if (tree.sample != 'TTJets') and (tree.sample != 'Data') ])
        for b in self.bkg:
            b.is_signal = False

    def skim_fully_resolved(self, signal):
        fully_resolved = EventFilter('signal_fully_resolved', filter=lambda t: t.nfound_select==6)
        # all_bjets = CollectionFilter('jet', filter=lambda t: t.jet_signalId > -1)

        filter = FilterSequence(
            fully_resolved, 
            # all_bjets
        )

        self.signal = signal.apply(filter)

    def calc_abs_scale(self, signal, bkg):

        sample_norm = {
            'QCD':1.0,
            'TTJets':1.0,
        }

        def use_abs_scale(t):
            abs_norm = sample_norm.get(t.sample, 1)
            abs_scale = abs_norm * np.abs(t.scale)
            t.extend(abs_scale=abs_scale)

        (signal + bkg).apply(use_abs_scale)

    def calc_sample_norm(self, signal, bkg):
        signal_norm = {
            True:0.04222842979772037,
            False:8.438776771744896e-05,
        }

        def use_sample_norm(t):
            norm = signal_norm.get(t.is_signal, 1)
            norm_abs_scale = norm*t.abs_scale
            t.extend(norm_abs_scale=norm_abs_scale)

        (signal+bkg).apply(use_sample_norm)

    def calc_dataset_norm(self, signal, bkg):
        max_norm = 169622.79959450764

        def use_dataset_norm(t):
            norm = max_norm 
            dataset_norm_abs_scale = norm * t.norm_abs_scale
            t.extend(dataset_norm_abs_scale=dataset_norm_abs_scale)
        (signal+bkg).apply(use_dataset_norm)

    def is_bkg(self, signal, bkg):
        signal.apply(lambda t : t.extend(is_bkg=ak.zeros_like(t.Run)))
        bkg.apply(lambda t : t.extend(is_bkg=ak.ones_like(t.Run)))
    
    def write_trees(self, signal, bkg, data):

        include=['^jet','^X','.*scale$','is_bkg','gen_X_m','gen_Y1_m']

        if any(signal.objs):
            (signal).write(
                'fully_res_{base}',
                include=include,
            )

        if any(bkg.objs):
            (bkg).write(
                'training_{base}',
                include=include,
            )