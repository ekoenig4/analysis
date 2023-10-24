from utils import *
import utils.notebookUtils as nb

class f_select_leading_btag:
    def __init__(self, field, ptcut=20, njet=6):
        self.field = field
        self.ptcut = ptcut
        self.njet = njet

    def __call__(self, tree):
        field = tree[self.field]

        ptmask = tree.ak4_pt > self.ptcut
        btag = tree.ak4_bdisc[ptmask]

        order = ak_rank(btag, axis=1) < 6
        return field[ptmask][order]

class Notebook(nb.Notebook):
    @staticmethod
    def add_parser(parser):
        parser.set_defaults(config='configs/nanohh4b/feynnet_training.yaml')
        parser.add_argument('--dout', default='')

    def init(self):
        self.dout = os.path.join('nanoHH4b', 'feynnet_training', self.dout)

        treekwargs = dict(
            weights=self.weights,
            treename='Events',
            normalization=None,
        )

        self.signal = ObjIter([Tree(self.signal, **treekwargs)])
        self.bkg = ObjIter([Tree(self.qcd, **treekwargs, sample='qcd-b', color='#FFBE00'), Tree(self.ttbar, **treekwargs)])

    def plot_jet_pt_gt_20(self, signal, bkg):
        suptitle = 'jets (pT > 20 GeV)'

        ak4_signalId = f_select_leading_btag('ak4_signalId', ptcut=20, njet=6)
        study.quick(
            signal,
            varlist=[lambda tree: ak.sum(ak4_signalId(tree) != -1, axis=1)],
            suptitle=suptitle,
            xlabels=['Number of GEN Higgs jets'],
            efficiency=True, legend=True, ylim=(0, 0.75), grid=True,
            saveas=os.path.join(self.dout, 'n_gen_higgs_jet_pt_gt_20.png'),
        )

        study.quick(
            signal + bkg,
            varlist=['ak.sum(ak4_pt > 20, axis=1)'],
            suptitle=suptitle,
            xlabels=['Number of jets'],
            binlist=[np.arange(16)],
            efficiency=True, legend=True, ylim=(0, 0.55), grid=True,
            saveas=os.path.join(self.dout, 'n_jet_pt_gt_20.png'),
        )

    def plot_jet_pt_gt_30(self, signal, bkg):
        suptitle = 'jets (pT > 30 GeV)'

        ak4_signalId = f_select_leading_btag('ak4_signalId', ptcut=30, njet=6)
        study.quick(
            signal,
            varlist=[lambda tree: ak.sum(ak4_signalId(tree) != -1, axis=1)],
            suptitle=suptitle,
            xlabels=['Number of GEN Higgs jets'],
            efficiency=True, legend=True, ylim=(0, 0.75), grid=True,
            saveas=os.path.join(self.dout, 'n_gen_higgs_jet_pt_gt_30.png'),
        )

        study.quick(
            signal + bkg,
            varlist=['ak.sum(ak4_pt > 30, axis=1)'],
            suptitle=suptitle,
            xlabels=['Number of jets'],
            binlist=[np.arange(16)],
            efficiency=True, legend=True, ylim=(0, 0.55), grid=True,
            saveas=os.path.join(self.dout, 'n_jet_pt_gt_30.png'),
        )

    def plot_jet_pt_gt_40(self, signal, bkg):
        suptitle = 'jets (pT > 40 GeV)'

        ak4_signalId = f_select_leading_btag('ak4_signalId', ptcut=40, njet=6)
        study.quick(
            signal,
            varlist=[lambda tree: ak.sum(ak4_signalId(tree) != -1, axis=1)],
            suptitle=suptitle,
            xlabels=['Number of GEN Higgs jets'],
            efficiency=True, legend=True, ylim=(0, 0.75), grid=True,
            saveas=os.path.join(self.dout, 'n_gen_higgs_jet_pt_gt_40.png'),
        )

        study.quick(
            signal + bkg,
            varlist=['ak.sum(ak4_pt > 40, axis=1)'],
            suptitle=suptitle,
            xlabels=['Number of jets'],
            binlist=[np.arange(16)],
            efficiency=True, legend=True, ylim=(0, 0.55), grid=True,
            saveas=os.path.join(self.dout, 'n_jet_pt_gt_40.png'),
        )



if __name__ == '__main__':
    Notebook.main()