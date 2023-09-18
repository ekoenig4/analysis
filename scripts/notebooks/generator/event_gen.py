from functools import partial
import os
os.environ['KMP_WARNINGS'] = 'off'
import sys
import git

import uproot as ut
import awkward as ak
import numpy as np
import math
import vector
import sympy as sp
import tabulate


import re
from tqdm import tqdm
import timeit
import re

sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils import *

from utils.notebookUtils import Notebook, required, dependency

def main():
    notebook = EventGenerator.from_parser()
    notebook.hello()
    notebook.run()

class EventGenerator(Notebook):
    @staticmethod
    def add_parser(parser):
        parser.add_argument('--module', type=str, default='x_hh_4b', help='generator to use')
        parser.add_argument('--gen', type=str, default='x_hh_4b', help='generator to use')
        parser.add_argument('-n', type=int, default=1000, help='number of events to generate')
        parser.add_argument('-o', '--output', type=str, default='{gen}.root', help='output file name', dest='fname')

    def init(self):
        import importlib
        self.module = importlib.import_module(f'utils.genprodUtils.generators.{self.module}')
        generators = { key.lower(): getattr(self.module, key) for key in dir(self.module) if key[0] != '_' } 
        self.generator = generators[self.gen.lower()]( self.module.gen_info )
        self.fname = self.fname.format(gen=self.gen)

    def generate(self, n):
        print(f'Generating {n} events')
        self.tree = self.generator(n)

        print(f'Generation Efficiency: {len(self.tree)/n:.2%}')

    def save(self, fname, tree):
        print('Saving')

        def _prep_to_write_(tree):
            fields = tree.fields

            newtree = dict()
            # remove any with_name attributes
            for field in fields:

                if any(tree[field].fields):
                    for subfield in tree[field].fields:
                        newtree[f"{field}_{subfield}"] = tree[field][subfield]
                else:
                    newtree[field] = tree[field]
                
            return newtree

        tree = _prep_to_write_(tree)

        for i in range(2):
            try:
                with ut.recreate(fname) as f:
                    f['Events'] = tree
                break
            except ValueError:
                ...


if __name__ == '__main__':
    main()