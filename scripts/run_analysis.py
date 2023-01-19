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
    method_list = template.methods.keys
    parser.add_argument('--template', default=template, dest='_class_template_')
    parser.add_argument(f'--only', nargs="*", help=f'Disable all other methods from run list', default=method_list)
    parser.add_argument(f'--disable', nargs="*", help=f'Disable method from run list', default=[])
    parser.add_argument(f'--module', default='fc.eightb.preselection.t8btag_minmass', help='specify the file collection module to use for all samples')
    parser.add_argument(f'--no-signal', default=False, action='store_true', help='do not load any signal files')
    parser.add_argument(f'--use-signal', default='full_signal_list', help='which signal list to load')
    parser.add_argument(f'--no-bkg', default=False, action='store_true', help='do not load any background files')
    parser.add_argument(f'--no-data', default=False, action='store_true', help='do not load any data files')
    parser.add_argument(f'--debug', default=False, action='store_true', help='disable running analysis for debug')

    group = parser.add_argument_group(name)
    template._add_parser(group)

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
altfile = getattr(args, 'altfile', '{base}')

signal, bkg, data = ObjIter([]), ObjIter([]), ObjIter([])
use_signal = []

if not args.no_signal and not args.debug:
    use_signal  = getattr(module, args.use_signal)
    signal = ObjIter([Tree(f, altfile=altfile, report=False) for f in tqdm(use_signal)])
    args.use_signal = [ use_signal.index(f) for f in module.signal_list ]
else:
    args.use_signal = []

if not args.no_bkg and not args.debug:
    bkg = ObjIter([Tree(module.Run2_UL18.QCD_B_List, altfile=altfile), Tree(module.Run2_UL18.TTJets, altfile=altfile)])

if not args.no_data and not args.debug:
    data = ObjIter([ Tree(module.Run2_UL18.JetHT_Data_UL_List, altfile=altfile) ])

def add_to_list(i, method):
    import re 

    for disable in args.disable:
        if disable.isdigit() and int(disable) == i: return False
        if re.match(f'^{disable}$', method): return False 

    for only in args.only:
        if only.isdigit() and int(only) == i: return True
        if re.match(f'^{only}$', method): return True 

    return False
    
runlist = [method for i, method in enumerate(method_list) if add_to_list(i, method)]

analysis = template.__class__(
    signal=signal, bkg=bkg, data=data,
    runlist=runlist,
    **vars(args)
)

print( repr(analysis) )

if args.debug: exit()

analysis.run()