from tqdm import tqdm

import sys, git
sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils import *

from utils.notebookUtils import required, dependency
from utils.notebookUtils.driver.run_reduction import RunReduction

def main():
    notebook = SignalReduction.from_parser()
    notebook.print_namespace()
    print(notebook)
    notebook.run()

class SignalReduction(RunReduction):
    @staticmethod
    def add_parser(parser):
        parser.add_argument("--model", default='feynnet_bkg_33sig', help="model to use for loading feynnet")
        parser.set_defaults(
            altfile='test_{base}',
            module='fc.eightb.feynnet',
        )
        return parser
    
    @required
    def init_model(self, model):
        self.model = eightb.models.get_model(model)

    @required
    def load_feynnet(self, signal, bkg, model):
        (signal+bkg).apply(lambda tree : eightb.load_feynnet_assignment(tree, model=model.storage), report=True, parallel=True, thread_order=lambda thread : len(thread.obj) )

    def signal_masks(self, signal):
        signal.apply(lambda tree : tree.extend(all_eightb=tree.nfound_select==8))        

    @dependency(signal_masks)
    def eightb_efficiency(self, signal):
        def _efficiency(tree):
            tree.reductions['eightb_h_eff'] = ak.mean( tree.h_signalId[tree.all_eightb]>-1, axis=0 )
            tree.reductions['eightb_y_eff'] = ak.mean( tree.y_signalId[tree.all_eightb]>-1, axis=0 )
            tree.reductions['eightb_x_eff'] = ak.mean( tree.x_signalId[tree.all_eightb]>-1, axis=0 )

        signal.apply(_efficiency, report=True)

    def get_obj_dr(self, signal, bkg):
        (signal+bkg).apply(
            lambda tree : (
                add_h_j_dr(tree),
                add_y_h_dr(tree),
            )
        )  

    @dependency(get_obj_dr)
    def eightb_dr(self, signal):
        def _dr(tree):
            tree.reductions['eightb_h_dr'] = ak.mean( tree.h_j_dr[tree.all_eightb], axis=0 )
            tree.reductions['eightb_y_dr'] = ak.mean( tree.y_h_dr[tree.all_eightb], axis=0 )

            tree.reductions['eightb_corr_h_dr'] = ak.mean( tree.h_j_dr[tree.all_eightb & (tree.x_signalId==0)], axis=0 )
            tree.reductions['eightb_corr_y_dr'] = ak.mean( tree.y_h_dr[tree.all_eightb & (tree.x_signalId==0)], axis=0 )
            
            tree.reductions['eightb_incorr_h_dr'] = ak.mean( tree.h_j_dr[tree.all_eightb & (tree.x_signalId!=0)], axis=0 )
            tree.reductions['eightb_incorr_y_dr'] = ak.mean( tree.y_h_dr[tree.all_eightb & (tree.x_signalId!=0)], axis=0 )

        signal.apply(_dr, report=True)

    @dependency(signal_masks)
    def eightb_efficiency(self, signal):
        def _efficiency(tree):
            tree.reductions['eightb_h_eff'] = ak.mean( tree.h_signalId[tree.all_eightb]>-1, axis=0 )
            tree.reductions['eightb_y_eff'] = ak.mean( tree.y_signalId[tree.all_eightb]>-1, axis=0 )
            tree.reductions['eightb_x_eff'] = ak.mean( tree.x_signalId[tree.all_eightb]>-1, axis=0 )

        signal.apply(_efficiency, report=True)

    @dependency(signal_masks)
    def load_true(self, signal):
        signal.apply(lambda tree : eightb.load_true_assignment(tree), report=True)
        
    @dependency(load_true)
    def eightb_resolution(self, signal):
        def _resolution(tree):
            tree.reductions['eightb_h_res'] = ak.mean( (tree.h_m/tree.true_h_m)[tree.all_eightb], axis=0 )
            tree.reductions['eightb_y_res'] = ak.mean( (tree.y_m/tree.true_y_m)[tree.all_eightb], axis=0 )

        signal.apply(_resolution, report=True)

    
    def write_reductions(self, signal):
        import uproot as ut

        def _write_reductions(tree):
            reduction = getattr(tree, "reductions", None)
            if reduction is None: return

            newtree = dict(
                mx = tree.mx,
                my = tree.my,
                **reduction,
            )
            
            with ut.recreate(f'{self.dout}/MX_{tree.mx}_MY_{tree.my}.root') as f:
                f['tree'] = {
                    k : [v] for k, v in newtree.items()
                }

        signal.apply(_write_reductions, report=True)



if __name__ == "__main__": main()