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

import re
from tqdm import tqdm
import timeit
import re

sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils import *

from utils.notebookUtils import Notebook, required, dependency
from utils.notebookUtils.driver.run_skim import RunSkim

import json

# %%
def main():
    notebook = Notebook.from_parser()
    notebook.hello()
    notebook.run()

class Notebook(RunSkim):
    
    @staticmethod
    def add_parser(parser):
        parser.add_argument("--dout", default="reweight-info-6b",
                            help="directory to load/save cached reweighting values. Default reweight-info/")

        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--cache", action='store_true', help="running reweighting caching. Should be done first")
        group.add_argument("--apply", action='store_true', help="apply reweighting from cache")
        return parser
    
    def __init__(self, cache=False, apply=False, only=None, **kwargs):
        if cache:
            only = ['cache_reweight_info']
        elif apply:
            only = ['apply_max_sample_norm','is_bkg','write_trees']

        super().__init__(only=only, cache=cache, apply=apply, **kwargs)
    
    def init_cache(self):
        if not os.path.exists(self.dout):
            print(f" ... Making reweightning cache: {self.dout}")
            os.mkdir(self.dout)


    def init_apply(self):
        self.dout = f'{self.dout}'
        self.reweight_info = dict()
        for f_info in os.listdir(self.dout):
            if not f_info.endswith('.json'): continue
            print(f' ... loading {self.dout}/{f_info}')
            with open(f'{self.dout}/{f_info}', 'r') as f_info:
                info = json.load(f_info)
                self.reweight_info.update(**info)
        self.max_sample_abs_norm = min(
            float(info['max_sample_abs_norm'])
            for info in self.reweight_info.values()
        )

        for tree in self.bkg:
            fn = fc.cleanpath(tree.filelist[0].fname)
            tree.reweight_info = self.reweight_info[fn]
            print(fn)
            print(tree.reweight_info)

        for tree in self.signal:
            fn = fc.cleanpath(tree.filelist[0].fname)
            tree.reweight_info = {
                nb:self.reweight_info[f'{nb}:{fn}']
                for nb in ('full',)
            }
            print(fn)
            print(tree.reweight_info)

    #################################################
    # Define any selections on signal or background

    @required
    def skim_fully_resolved(self, signal):
        sixb_filter = EventFilter('signal_sixb', filter=lambda t: t.nfound_select==6)

        self.full_signal = signal
        # self.sixb_signal = signal.apply(sixb_filter)

        if self.apply:
            # for s in self.sixb_signal:
            #     s.reweight_info = s.reweight_info[6]
            for s in self.full_signal:
                s.reweight_info = s.reweight_info['full']
            
        self.signal = self.full_signal

    
    #################################################
    @dependency(init_cache)
    def cache_sample(self, signal, bkg):
        from collections import defaultdict

        samples = defaultdict(lambda:0)
        def _fill_samples_from_tree(t):
            samples[t.sample] += sum(f.total_events for f in t.filelist)

        (signal + bkg).apply(_fill_samples_from_tree, report=True)
        def get_sample(t):
            sample_total = samples[t.sample]
            file_total = sum(f.total_events for f in t.filelist)
            sample_total_norm = sample_total/file_total
            t.sample_total_norm = sample_total_norm
            t.extend(scale=sample_total_norm*t.scale)

        (signal + bkg).apply(get_sample, report=True)

    @dependency(init_apply)
    def apply_sample(self, signal, bkg):
        def get_sample(t):
            sample_total_norm = float(t.reweight_info['sample_total_norm'])
            t.extend(scale=sample_total_norm*t.scale)

        (signal + bkg).apply(get_sample, report=True)

    #################################################
    @dependency(cache_sample)
    def cache_abs_norm(self, signal, bkg):
        def get_abs_scale(t):
            scale = t.scale 
            abs_scale = np.abs(scale)
            norm = np.sum(scale)/np.sum(abs_scale)
            t.abs_norm = norm
            t.extend(abs_scale=norm*abs_scale)

        (signal + bkg).apply(get_abs_scale, report=True)

    
    @dependency(apply_sample)
    def apply_abs_norm(self, signal, bkg):
        def get_abs_scale(t):
            scale = t.scale 
            abs_scale = np.abs(scale)
            abs_norm = float(t.reweight_info['abs_norm'])
            t.extend(abs_scale=abs_norm*abs_scale)

        (signal + bkg).apply(get_abs_scale, report=True)

    #################################################
    @dependency(cache_abs_norm)
    def cache_sample_norm(self, signal, bkg):
        def get_sample_norm(sample):
            if not isinstance(sample, ObjIter): sample = ObjIter([sample])
            if not any(sample): return

            abs_scale = sample.abs_scale.cat
            sample_abs_norm = 1/np.sum(abs_scale)
            
            for tree in sample:
                tree.sample_abs_norm = sample_abs_norm
                tree.extend(norm_abs_scale= sample_abs_norm * tree.abs_scale)

        
        for tree in signal:
            get_sample_norm(ObjIter([tree]))

        get_sample_norm(bkg)

    @dependency(apply_abs_norm)
    def apply_sample_norm(self, signal, bkg):
        def get_sample_norm(tree):
            sample_abs_norm = float(tree.reweight_info['sample_abs_norm'])
            tree.extend(norm_abs_scale=sample_abs_norm * tree.abs_scale)

        (signal+bkg).apply(get_sample_norm)

    #################################################
    @dependency(cache_sample_norm)
    def cache_max_sample_norm(self, signal, bkg):
        def get_max_sample_scale(tree):
            norm_abs_scale = tree.norm_abs_scale
            max_sample_abs_norm = 1/ak.max(norm_abs_scale)
            tree.max_sample_abs_norm = max_sample_abs_norm

        (signal + bkg).apply(get_max_sample_scale)            

    @dependency(apply_sample_norm)
    def apply_max_sample_norm(self, signal, bkg):
        def get_max_sample_scale(tree):
                tree.extend(dataset_norm_abs_scale= self.max_sample_abs_norm*tree.norm_abs_scale)
        (signal + bkg).apply(get_max_sample_scale)

    #################################################
    @dependency(cache_max_sample_norm)
    def cache_reweight_info(self, full_signal, bkg):
        def cache_trees(trees, fname, tag=''):
            info = {
                f'{tag}{fc.cleanpath(t.filelist[0].fname)}':dict(
                    sample_total_norm = t.sample_total_norm,
                    abs_norm = t.abs_norm,
                    sample_abs_norm = t.sample_abs_norm,
                    max_sample_abs_norm = t.max_sample_abs_norm
                )
                for t in trees
            }
            print(info)
            with open(f"{self.dout}/{fname}.json", "w") as f:
                json.dump(info, f, indent=4)

        for nb, trees in zip(['full'],[full_signal,]):
            for tree in trees:
                cache_trees([tree], f'{tree.sample}_{nb}b-info', f'{nb}:')
        
        if any(bkg.objs):
            cache_trees(bkg, 'bkg-info')

    #################################################
    def is_bkg(self, signal, bkg):
        signal.apply(lambda t : t.extend(is_bkg=ak.zeros_like(t.Run)))
        bkg.apply(lambda t : t.extend(is_bkg=ak.ones_like(t.Run)))

    def write_trees(self, full_signal, sixb_signal, bkg):
        include=['^jet','.*scale$','is_bkg','nfound_select', '^X', 'gen_X_m','gen_Y_m']

        def suzs_to_evan(f):
            return f.replace('srosenzw','ekoenig')

        if sixb_signal and any(sixb_signal.objs):
            sixb_signal.write(
                f'reweight_sixb_{{base}}',
                include=include,
            )

        if full_signal and any(full_signal.objs):

            def rename(f):
                f = suzs_to_evan(f)
                path = os.path.dirname(f)
                base = os.path.basename(f)
                return f'{path}/reweight_{base}'
            
            full_signal.write(
                rename,
                include=include,
            )

        if bkg and any(bkg.objs):
            def move_to_local(f):
                # f = fc.cleanpath(f)
                # marina = 'mkolosov/MultiHiggs/DiHiggs/RunII/FeynNetTraining/ForFeynNet_UL18_Background_23May2023'
                # evan = 'ekoenig/6bAnalysis/NTuples/feynnet'
                # f =  f.replace(marina, evan)
                path = os.path.dirname(f)
                base = os.path.basename(f)
                # fc.mkdir_eos(path, recursive=True)
                return f'{path}/reweight_{base}'

            bkg.write(
                move_to_local,
                include=include,
            )


if __name__ == '__main__':
    main()