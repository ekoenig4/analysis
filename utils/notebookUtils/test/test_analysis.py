from tqdm import tqdm

import sys, git
sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils.notebookUtils import Notebook, required, dependency
from utils import Tree, ObjIter, fc, study

from utils.notebookUtils.driver.run_skim import RunSkim

def main():
    notebook = Analysis.from_parser()
    notebook.print_namespace()
    print(notebook)
    notebook.run()

class Analysis(RunSkim):
    def plot_variables(self, signal):
        study.quick(
            signal,
            varlist=['X_m'],
            saveas='test/X_m'
        )

if __name__ == '__main__': main()

