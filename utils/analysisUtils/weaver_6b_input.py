from .. import *
from .. import sixbUtils as sixb

from collections import defaultdict

import json

class weaver_6b_input(Analysis):
    """Analysis skim used to cache and apply reweighting procedure for training 

    Caching: should be done FIRST
        usage: ./run_files weaver_input --cache /path/to/file1/ /path/to/file2/ ...

    Caching will consider all files reconginzed as signal as their own separate samples and cache them
    each file with have values cached, background will be group together and should be ran in a single call

    Applying: should be done SECOND
        usage: ./parallel_files.sh weaver_input --apply

    *NOTE* parallel_files.sh should be modified to give the filelist of training files

    """
    @staticmethod
    def _add_parser(parser):
        parser.add_argument("--dout", default="reweight-6b-info",
                            help="directory to load/save cached reweighting values. Default reweight-info/")

        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--cache", action='store_true', help="running reweighting caching. Should be done first")
        group.add_argument("--apply", action='store_true', help="apply reweighting from cache")
        return parser

    def __init__(self, cache=False, apply=False, runlist=None, **kwargs):
        if cache:
            runlist = ['cache_reweight_info']
        elif apply:
            runlist = ['apply_max_sample_norm','is_bkg','write_trees']

        super().__init__(runlist=runlist, cache=cache, apply=apply, **kwargs)
    
    def init_cache(self):
        if not os.path.exists(self.dout):
            print(f" ... Making reweightning cache: {self.dout}")
            os.mkdir(self.dout)


    def init_apply(self):
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
                for nb in (6,5,4)
            }
            print(fn)
            print(tree.reweight_info)

    #################################################
    # @dependency(init_cache)
    # def cache_sample(self, trees):
    #     samples = defaultdict(lambda:0)
    #     for tree in trees:
    #         for f in tree.filelist:
    #             samples[f.sample] += f.total_events

    #     def reweight_sample(tree, samples=samples):
    #         xsec = tree.filelist[0].xsec
    #         total = samples[tree.sample]
    #         tree.sample_total = total

    #         scale = tree.genWeight if 'genWeight' in tree.fields else ak.ones(len(tree))
    #         scale = xsec * scale / total
    #         tree.extend(scale=scale)

    #     trees.apply(reweight_sample)    

    # @dependency(init_apply)
    # def apply_sample(self, signal, bkg):
    #     def reweight_sample(t):
    #         xsec = t.filelist[0].xsec
    #         sample_total = t.reweight_info['sample_total']
    #         scale = t.genWeight if 'genWeight' in t.fields else ak.ones(len(t))
    #         scale = xsec * scale / sample_total
    #         t.extend(scale=scale)

    #     (signal + bkg).apply(reweight_sample, report=True)
    ###################################################
    # Define any selections on signal or background

    @required
    def skim_fully_resolved(self, signal):
        sixb_filter = EventFilter('signal_sixb', filter=lambda t: t.nfound_select==6)
        fiveb_filter = EventFilter('signal_fiveb', filter=lambda t: t.nfound_select==5)
        fourb_filter = EventFilter('signal_fourb', filter=lambda t: t.nfound_select==4)

        self.sixb_signal = signal.apply(sixb_filter)
        self.fiveb_signal = signal.apply(fiveb_filter)
        self.fourb_signal = signal.apply(fourb_filter)

        if self.apply:
            for s in self.sixb_signal:
                s.reweight_info = s.reweight_info[6]
            
            for s in self.fiveb_signal:
                s.reweight_info = s.reweight_info[5]

            for s in self.fourb_signal:
                s.reweight_info = s.reweight_info[4]

        self.signal = self.sixb_signal + self.fiveb_signal + self.fourb_signal

    #################################################
    # @dependency(cache_sample)
    @dependency(init_cache)
    def cache_abs_norm(self, signal, bkg):
        def get_abs_scale(t):
            scale = t.scale 
            abs_scale = np.abs(scale)
            norm = np.sum(scale)/np.sum(abs_scale)
            t.abs_norm = norm
            t.extend(abs_scale=norm*abs_scale)

        (signal + bkg).apply(get_abs_scale, report=True)

    
    # @dependency(apply_sample)
    @dependency(init_apply)
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
    def cache_reweight_info(self, sixb_signal, fiveb_signal, fourb_signal, bkg):
        def cache_trees(trees, fname, tag=''):
            info = {
                f'{tag}{fc.cleanpath(t.filelist[0].fname)}':dict(
                    # sample_total = t.sample_total,
                    abs_norm = t.abs_norm,
                    sample_abs_norm = t.sample_abs_norm,
                    max_sample_abs_norm = t.max_sample_abs_norm
                )
                for t in trees
            }
            print(info)
            with open(f"{self.dout}/{fname}.json", "w") as f:
                json.dump(info, f, indent=4)

        for nb, trees in zip([6,5,4],[sixb_signal, fiveb_signal, fourb_signal]):
            for tree in trees:
                cache_trees([tree], f'{tree.sample}_{nb}b-info', f'{nb}:')
        
        if any(bkg.objs):
            cache_trees(bkg, 'bkg-info')

    #################################################
    def is_bkg(self, signal, bkg):
        signal.apply(lambda t : t.extend(is_bkg=ak.zeros_like(t.Run)))
        bkg.apply(lambda t : t.extend(is_bkg=ak.ones_like(t.Run)))

    def write_trees(self, sixb_signal, fiveb_signal, fourb_signal, bkg):
        include=['^jet','^X','.*scale$','is_bkg','gen_X_m','gen_Y1_m','gen_Y_m','nfound_select']

        if any(sixb_signal.objs):
            sixb_signal.write(
                'reweight_sixb_{base}',
                include=include,
            )

        if any(fiveb_signal.objs):
            fiveb_signal.write(
                'reweight_fiveb_{base}',
                include=include,
            )

        if any(fourb_signal.objs):
            fourb_signal.write(
                'reweight_fourb_{base}',
                include=include,
            )

        if any(bkg.objs):
            bkg.write(
                'reweight_{base}',
                include=include,
            )