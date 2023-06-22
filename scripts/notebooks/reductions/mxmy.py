from tqdm import tqdm

import sys, git
sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils import *

from utils.notebookUtils import required, dependency
from utils.notebookUtils.driver.run_reduction import RunReduction

def main():
    notebook = Reduction.from_parser()
    notebook.print_namespace()
    print(notebook)
    notebook.run()

class Reduction(RunReduction):
    @staticmethod
    def add_parser(parser):
        parser.add_argument("--model", default='feynnet_trgkin_mx_my_reweight_v2', help="model to use for loading feynnet")
        parser.set_defaults(
            # altfile='test_{base}',
            # module='fc.eightb.feynnet',
            module='fc.eightb.preselection.t8btag_minmass',
        )
        return parser
    
    @required
    def init_model(self, model):
        self.model = eightb.models.get_model(model)

    def trigger_kinematics(self, signal):
        pt_filter = eightb.selected_jet_pt('trigger')

        def pfht(t):
            return ak.sum(t.jet_pt[(t.jet_pt > 30)], axis=-1)
        pfht_filter = EventFilter('pfht330', filter=lambda t : pfht(t) > 330)

        event_filter = FilterSequence(pfht_filter, pt_filter)
        self.signal = signal.apply(event_filter)

    def fully_resolved_efficiency(self, signal):
        def _efficiency(tree):
            tree.reductions['fully_resolved_efficiency'] = ak.mean( tree.nfound_select==8, axis=0 )

        signal.apply(_efficiency,report=True)

    # @required
    def load_feynnet(self, signal, bkg, model):
        
        load_feynnet = eightb.f_load_feynnet_assignment(self.model.analysis)
        (signal+bkg).apply(load_feynnet, report=True)

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

    def mass(self, signal, bkg):
        def _mass(tree):
            weights = tree.scale
            tree.reductions['h1y1_m'] = np.histogram( tree.h_m[:,0], bins=np.linspace(0,300,30), weights=weights )
            tree.reductions['h2y1_m'] = np.histogram( tree.h_m[:,1], bins=np.linspace(0,300,30), weights=weights )
            tree.reductions['h1y2_m'] = np.histogram( tree.h_m[:,2], bins=np.linspace(0,300,30), weights=weights )
            tree.reductions['h2y2_m'] = np.histogram( tree.h_m[:,3], bins=np.linspace(0,300,30), weights=weights )
            
            tree.reductions['y1_m'] = np.histogram( tree.y_m[:,0], bins=np.linspace(0,1000,30), weights=weights )
            tree.reductions['y2_m'] = np.histogram( tree.y_m[:,1], bins=np.linspace(0,1000,30), weights=weights )

        (signal+bkg).apply(_mass, report=True)

    def chi2_mass(self, signal, bkg):
        def _mass(tree):
            weights = tree.scale
            chi = np.sqrt( ak.sum( (tree.h_m-125)**2, axis=1 ) )
            mask = chi < 50

            tree.reductions['hm_chi'] = np.histogram( chi, bins=np.linspace(0,300,30), weights=tree.scale )

            tree.reductions['h1y1_m_chi50'] = np.histogram( tree.h_m[:,0][mask], bins=np.linspace(0,300,30), weights=weights[mask] )
            tree.reductions['h2y1_m_chi50'] = np.histogram( tree.h_m[:,1][mask], bins=np.linspace(0,300,30), weights=weights[mask] )
            tree.reductions['h1y2_m_chi50'] = np.histogram( tree.h_m[:,2][mask], bins=np.linspace(0,300,30), weights=weights[mask] )
            tree.reductions['h2y2_m_chi50'] = np.histogram( tree.h_m[:,3][mask], bins=np.linspace(0,300,30), weights=weights[mask] )
            
            tree.reductions['y1_m_chi50'] = np.histogram( tree.y_m[:,0][mask], bins=np.linspace(0,1000,30), weights=weights[mask] )
            tree.reductions['y2_m_chi50'] = np.histogram( tree.y_m[:,1][mask], bins=np.linspace(0,1000,30), weights=weights[mask] )
            tree.reductions['x_m_chi50'] = np.histogram( tree.x_m[mask], bins=np.linspace(0,2000,30), weights=weights[mask] )

        (signal+bkg).apply(_mass, report=True)
    
    @dependency(load_true)
    def eightb_mass(self, signal):
        def _mass(tree):
            weights = tree.scale[tree.all_eightb]
            tree.reductions['eightb_h1y1_m'] = np.histogram( tree.h_m[:,0][tree.all_eightb], bins=np.linspace(0,300,30), weights=weights )
            tree.reductions['eightb_h2y1_m'] = np.histogram( tree.h_m[:,1][tree.all_eightb], bins=np.linspace(0,300,30), weights=weights )
            tree.reductions['eightb_h1y2_m'] = np.histogram( tree.h_m[:,2][tree.all_eightb], bins=np.linspace(0,300,30), weights=weights )
            tree.reductions['eightb_h2y2_m'] = np.histogram( tree.h_m[:,3][tree.all_eightb], bins=np.linspace(0,300,30), weights=weights )
            
            tree.reductions['eightb_y1_m'] = np.histogram( tree.y_m[:,0][tree.all_eightb], bins=np.linspace(0,1000,30), weights=weights )
            tree.reductions['eightb_y2_m'] = np.histogram( tree.y_m[:,1][tree.all_eightb], bins=np.linspace(0,1000,30), weights=weights )

        signal.apply(_mass, report=True)

    
    def write_signal(self, signal):
        import uproot as ut

        def _write(tree):
            reduction = getattr(tree, "reductions", None)
            if reduction is None: return

            newtree = dict(
                mx = tree.mx,
                my = tree.my,
                **{
                    k : [v] for k, v in reduction.items() if not isinstance(v, tuple)
                }
            )

            newhistos = {
                k : v for k, v in reduction.items() if isinstance(v, tuple)
            }
            
            with ut.recreate(f'{self.dout}/MX_{tree.mx}_MY_{tree.my}.root') as f:
                f['tree'] = {
                    k : [v] for k, v in newtree.items()
                }
                for key, histo in newhistos.items():
                    f[f"MX_{tree.mx}_MY_{tree.my}_{key}"] = histo

        signal.apply(_write, report=True)

    def write_bkg(self, bkg):
        import uproot as ut
        def _write(tree):
            reduction = getattr(tree, "reductions", None)
            if reduction is None: return

            newtree = dict(
                **{
                    k : [v] for k, v in reduction.items() if not isinstance(v, tuple)
                }
            )

            newhistos = {
                k : v for k, v in reduction.items() if isinstance(v, tuple)
            }
            
            with ut.recreate(f'{self.dout}/{tree.sample}.root') as f:

                if any(newtree):
                    f['tree'] = {
                        k : [v] for k, v in newtree.items()
                    }

                for key, histo in newhistos.items():
                    f[f"{tree.sample}_{key}"] = histo

        bkg.apply(_write, report=True)




if __name__ == "__main__": main()