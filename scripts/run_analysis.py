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

templates = { subclass.__name__:subclass() for subclass in Analysis.__subclasses__() }

driver = ArgumentParser(
    description='General Analysis Driver',
    formatter_class=RawTextHelpFormatter
)

subparser = driver.add_subparsers()

for name, template in templates.items():
    parser = subparser.add_parser(
        name=name,
        description=str(repr(template)),
        formatter_class=RawTextHelpFormatter
    )
    template._add_parser(parser)
    method_list = template.methods.keys
    parser.add_argument('--template', default=template, dest='_class_template_')
    parser.add_argument(f'--only', nargs="*", help=f'Disable all other methods from run list', default=method_list)
    parser.add_argument(f'--disable', nargs="*", help=f'Disable method from run list', default=[])
    parser.add_argument(f'--module', default='fc.eightb.preselection.t8btag_minmass', help='specify the file collection module to use for all samples')
    parser.add_argument(f'--no-signal', default=False, action='store_true', help='do not load any signal files')
    parser.add_argument(f'--no-bkg', default=False, action='store_true', help='do not load any background files')
    parser.add_argument(f'--no-data', default=False, action='store_true', help='do not load any data files')

args, unk = driver.parse_known_args()

print("---Arguments---")
for key, value in vars(args).items():
    if key.startswith('_'): continue
    print(f"{key} = {value}")
if any(unk):
    print(f"uknown = {unk}")
print()

template = args._class_template_
method_list = template.methods.keys

# %%

def _module(mod):
    local = dict()
    exec(f"module = {mod}", globals(), local)
    return local['module']
module = _module(args.module)
altfile = getattr(args, 'altfile', None)

signal, bkg, data = ObjIter([]), ObjIter([]), ObjIter([])
use_signal = []

if not args.no_signal:
    use_signal  = module.full_signal_list
    signal = ObjIter([Tree(f, altfile=altfile, report=False) for f in tqdm(use_signal)])
    use_signal = [ use_signal.index(f) for f in module.signal_list ]

if not args.no_bkg:
    bkg = ObjIter([Tree(module.Run2_UL18.QCD_B_List, altfile=altfile), Tree(module.Run2_UL18.TTJets, altfile=altfile)])

if not args.no_data:
    data = ObjIter([ Tree(module.Run2_UL18.JetHT_Data_UL_List, altfile=altfile) ])

def add_to_list(method):
    import re 

    for disable in args.disable:
        if any( re.findall(disable, method) ): return False 

    for only in args.only:
        if any( re.findall(only, method) ): return True 

    return False
    

runlist = [method for method in method_list if add_to_list(method)]

analysis = template.__class__(
    signal=signal, bkg=bkg, data=data,
    use_signal=use_signal,
    **vars(args)
)

print( repr(analysis) )
analysis.run(runlist=runlist)