from tqdm import tqdm

import sys, git
sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils.notebookUtils import Notebook, required, dependency
from utils import Tree, ObjIter, fc

def main():
    notebook = RunAnalysis.from_parser()
    notebook.print_namespace()
    print(notebook)
    notebook.run()

class RunAnalysis(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument(f'--module', default='fc.eightb.preselection.t8btag_minmass', help='specify the file collection module to use for all samples')
        parser.add_argument("--altfile", default='{base}',
                            help="output file pattern to write file with. Use {base} to substitute existing file")
        
        parser.add_argument(f'--no-signal', default=False, action='store_true', help='do not load any signal files')
        parser.add_argument(f'--use-signal', default='full_signal_list', help='which signal list to load')
        parser.add_argument(f'--no-bkg', default=False, action='store_true', help='do not load any background files')
        parser.add_argument(f'--no-data', default=False, action='store_true', help='do not load any data files')
        return parser

    @required
    def init_module(self, module):
        def _module(mod):
            local = dict()
            exec(f"module = {mod}", globals(), local)
            return local['module']
        self.module = _module(module)

    @required
    def init_files(self, altfile='{base}', module=None):
        use_signal = []
        if not self.no_signal:
            use_signal  = getattr(module, self.use_signal)
            self.signal = ObjIter([Tree(f, altfile=altfile, report=False) for f in tqdm(use_signal)])
            self.use_signal = [ use_signal.index(f) for f in module.signal_list ]
        else:
            self.use_signal = []

        if not self.no_bkg and not self.debug:
            self.bkg = ObjIter([Tree(module.Run2_UL18.QCD_B_List, altfile=altfile), Tree(module.Run2_UL18.TTJets, altfile=altfile)])

        if not self.no_data and not self.debug:
            self.data = ObjIter([ Tree(module.Run2_UL18.JetHT_Data_UL_List, altfile=altfile) ])

    @property
    def trees(self): return self.signal + self.bkg + self.data

if __name__ == '__main__': main()