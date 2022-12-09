# %%
import os
os.environ['KMP_WARNINGS'] = 'off'
import sys
import git

import matplotlib as mpl
mpl.use('Agg')

import uproot as ut
import awkward as ak
import numpy as np
import math
import vector
import sympy as sp

import re
from tqdm import tqdm
import timeit
import re

sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils import *

# %%
from argparse import ArgumentParser, RawTextHelpFormatter
_template = eightb.analysis.TestAnalysis()

parser = ArgumentParser(description=repr(_template), formatter_class=RawTextHelpFormatter)
parser.add_argument('--altfile', default='ranked_quadh_{base}',
                    help='altfile pattern to use instead of just ntuple.root, you can use {base} to sub in existing file')

if hasattr(_template, '_add_parser'):
    parser = _template._add_parser(parser)

method_list = _template.methods.keys
parser.add_argument(f'--only', nargs="*", help=f'Disable all other methods from run list', default=method_list)
parser.add_argument(f'--disable', nargs="*", help=f'Disable method from run list', default=[])

args, _ = parser.parse_known_args()

print("---Arguments---")
for key, value in vars(args).items():
    print(f"{key} = {value}")
print()
# %%
module = fc.eightb.preselection.t8btag_minmass
altfile = args.altfile

# %%
use_signal = [ module.full_signal_list.index(f) for f in module.signal_list ]
signal = ObjIter([Tree(f, altfile=altfile, report=False) for f in tqdm(module.full_signal_list)])
bkg = ObjIter([Tree(module.Run2_UL18.QCD_B_List, altfile=altfile), Tree(module.Run2_UL18.TTJets, altfile=altfile)])
data = ObjIter([ Tree(module.Run2_UL18.JetHT_Data_UL_List, altfile=altfile) ])

if not args.ptcuts: args.ptcuts = args.ptwp
if not args.btagcuts: args.btagcuts = args.btagwp


runlist = [method for method in method_list if ( method in args.only and not method in args.disable )]

analysis = _template.__class__(
    signal=signal, 
    bkg=bkg, 
    data=data,
    use_signal=use_signal,
    dout=args.dout,
    ptcuts=args.ptcuts, btagcuts=args.btagcuts,
    ar_center = args.ar_center,
    vr_center = args.vr_center,
    sr_r = args.sr_r, 
    cr_r =  args.cr_r,
    bdt_features = [
        'jet_ht','min_jet_deta','max_jet_deta','min_jet_dr','max_jet_dr'
    ] + [
        f'h{i+1}_{var}'
        for var in ('pt','jet_dr')
        for i in range(4)
    ] + [
        f'h{i+1}{j+1}_{var}'
        for var in ('dphi','deta')
        for i in range(4)
        for j in range(i+1, 4)
    ],
)

print( repr(analysis) )
analysis.run(runlist=runlist)