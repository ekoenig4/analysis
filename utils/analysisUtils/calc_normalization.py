from .. import *
from .. import eightbUtils as eightb

class calc_normalization(Analysis):
    @staticmethod
    def _add_parser(parser):
        parser.add_argument("--altfile", required=True,
                            help="output file pattern to write file with. Use {base} to substitute existing file")
        return parser

    def calc_rescale(self, signal, bkg):

        sample_scale = {
            'MX_700_MY_300':1.2434550400752545e-18, 
            'MX_1000_MY_450':6.4806808031115614e-18, 
            'MX_1200_MY_500':1.343392630420647e-17, 
            'TTJets':7.294459735548297e-09
        }

        def rescale(t):
            print(t.sample, t.filelist[0].scale)
        (signal + bkg).apply(rescale)

    def normalize_signal(self, signal):
        sample_scale = {
            'MX_700_MY_300':282.6071876702381, 
            'MX_1000_MY_450':141.71038484276528, 
            'MX_1200_MY_500':103.01861324839004
        }

        def use_norm_signal(t):
            print(t.sample, 1/ak.sum(t.scale))
        signal.apply(use_norm_signal)

    def calc_abs_scale(self, signal, bkg):

        sample_norm = {
            'QCD':0.9990666979650883,
            'TTJets':0.2295229640177209,
        }

        def use_abs_scale(t):
            abs_scale = np.abs(t.scale)
            print(t.sample, ak.sum(t.scale)/ak.sum(abs_scale))
        (signal + bkg).apply(use_abs_scale)
