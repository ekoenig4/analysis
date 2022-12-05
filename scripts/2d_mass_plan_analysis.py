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
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dout', default='ranked_quadh',
                    help='specify directory to save plots into')
parser.add_argument('--altfile', default='ranked_quadh_{base}',
                    help='altfile pattern to use instead of just ntuple.root, you can use {base} to sub in existing file')

parser.add_argument('--ptwp', default='loose',
                    help='Specify preset working point for pt cuts')
parser.add_argument('--ptcuts', type=float, nargs="*",
                    help='List of jet pt cuts to apply to selected jets')
parser.add_argument('--use_regressed', default=False, action='store_true',
                    help='Use ptRegressed for pt cuts instead of regular pt')
parser.add_argument('--btagwp', default='loose',
                    help='Specify preset working point for btag cuts')
parser.add_argument('--btagcuts', type=int, nargs="*",
                    help='List of jet btag wps cuts to apply to selected jets')
parser.add_argument('--ar-center', default=[125,125], type=float, nargs="*",
                    help='Specify the higgs mass centers to use for the analysis region')
parser.add_argument('--vr-center', default=[210,210], type=float, nargs="*",
                    help='Specify the higgs mass centers to use for the validation region')
parser.add_argument('--sr-r', default=50, type=float,
                    help='Specify the radius to in higgs mass space for signal region')
parser.add_argument('--cr-r', default=70, type=float,
                    help='Specify the radius to in higgs mass space for control region')

_template = eightb.analysis.TestAnalysis()
method_list = _template.methods.keys
parser.add_argument(f'--only', nargs="*", choices=method_list, help=f'Disable all other methods from run list', default=method_list)
parser.add_argument(f'--disable', nargs="*", choices=method_list, help=f'Disable method from run list', default=[])

args = parser.parse_args()

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
    signal=signal, bkg=bkg, data=data,
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
        for var in ('pt','dr')
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