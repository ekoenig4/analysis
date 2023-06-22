from tqdm import tqdm

import sys, git, os
sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils.notebookUtils import Notebook, required, dependency
from utils import Tree, ObjIter, fc

def main():
    notebook = RunReduction.from_parser()
    notebook.print_namespace()
    print(notebook)
    notebook.run()

class RunReduction(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument(f'--module', default='fc.eightb.preselection.t8btag_minmass', help='specify the file collection module to use for all samples')
        parser.add_argument("--dout", default='feynnet', help="output directory to write files to")
        parser.add_argument("--altfile", default='{base}',
                            help="output file pattern to write file with. Use {base} to substitute existing file")
        parser.add_argument("--merge", action='store_true', help="merge all files into one")
        parser.add_argument('files', nargs="*", help=f'files to run')
        return parser
    
    @required
    def init_dout(self, dout):
        self.dout = os.path.join(dout, self.model)

        if not os.path.exists(self.dout):
            os.makedirs(self.dout)

    @required
    def init_module(self, module):
        def _module(mod):
            local = dict()
            exec(f"module = {mod}", globals(), local)
            return local['module']
        self.module = _module(module)

    @required
    def init_files(self, files, altfile='{base}', module=None):
        def _file(f):
            from_glob = fc.fs.repo.glob(f)
            if from_glob: return from_glob
            if not module: return f
            
            local = dict()
            exec(f"f = module.{f}", dict(module=module), local)
            return local['f']
        def iter_files(fs):
            if isinstance(fs, list): return fs
            else: return [fs]

        files = [ _file(f) for f in files ]
        files = [ f for fs in files for f in iter_files(fs) ]

        if self.merge:
            trees = [ Tree(files, altfile=altfile, report=True) ]
        else:
            trees = [ Tree(f, altfile=altfile, report=False) for f in tqdm(files) ]

        self.signal = ObjIter([ tree for tree in trees if tree.is_signal ])
        self.bkg = ObjIter([ tree for tree in trees if (not tree.is_data and not tree.is_signal) ])
        self.data = ObjIter([ tree for tree in trees if tree.is_data ])

    @property
    def trees(self): return self.signal + self.bkg + self.data


if __name__ == '__main__': main()