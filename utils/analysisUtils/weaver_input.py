from .. import *
from .. import eightbUtils as eightb

import json

class weaver_input(Analysis):
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
        parser.add_argument("--dout", default="reweight-info",
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

        super().__init__(runlist=runlist, **kwargs)
    
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
            info['max_sample_abs_norm']
            for info in self.reweight_info.values()
        )

        for tree in self.trees:
            fn = fc.cleanpath(tree.filelist[0].fname)
            tree.reweight_info = self.reweight_info[fn]
            print(fn)
            print(tree.reweight_info)

    ###################################################
    # Define any selections on signal or background

    @required
    def skim_fully_resolved(self, signal):
        fully_resolved = EventFilter('signal_fully_resolved', filter=lambda t: t.nfound_select==8)

        filter = FilterSequence(
            fully_resolved
        )

        self.signal = signal.apply(filter)

    #################################################
    @dependency(init_cache)
    def cache_abs_norm(self, signal, bkg):
        def get_abs_scale(t):
            scale = t.scale 
            abs_scale = np.abs(scale)
            norm = np.sum(scale)/np.sum(abs_scale)
            t.abs_norm = norm
            t.extend(abs_scale=norm*abs_scale)

        (signal + bkg).apply(get_abs_scale, report=True)

    
    @dependency(init_apply)
    def apply_abs_norm(self, signal, bkg):
        def get_abs_scale(t):
            scale = t.scale 
            abs_scale = np.abs(scale)
            abs_norm = t.reweight_info['abs_norm']
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

        signal.apply(get_sample_norm)            
        get_sample_norm(bkg)

    @dependency(apply_abs_norm)
    def apply_sample_norm(self, signal, bkg):
        def get_sample_norm(tree):
            sample_abs_norm = tree.reweight_info['sample_abs_norm']
            tree.extend(norm_abs_scale=sample_abs_norm * tree.abs_scale)

        (signal+bkg).apply(get_sample_norm)

    #################################################
    @dependency(cache_sample_norm)
    def cache_max_sample_norm(self, signal, bkg):
        def get_max_sample_scale(sample):
            if not isinstance(sample, ObjIter): sample = ObjIter([sample])
            if not any(sample): return

            norm_abs_scale = sample.norm_abs_scale.cat
            max_sample_abs_norm = 1/ak.max(norm_abs_scale)

            for tree in sample:
                tree.max_sample_abs_norm = max_sample_abs_norm

        signal.apply(get_max_sample_scale)            
        get_max_sample_scale(bkg)

    @dependency(apply_sample_norm)
    def apply_max_sample_norm(self, signal, bkg):
        def get_max_sample_scale(tree):
                tree.extend(dataset_norm_abs_scale= self.max_sample_abs_norm*tree.norm_abs_scale)
        (signal + bkg).apply(get_max_sample_scale)

    #################################################
    @dependency(cache_max_sample_norm)
    def cache_reweight_info(self, signal, bkg):
        def cache_signal(t):
            info = {
                fc.cleanpath(t.filelist[0].fname):dict(
                    abs_norm = t.abs_norm,
                    sample_abs_norm = t.sample_abs_norm,
                    max_sample_abs_norm = t.max_sample_abs_norm
                )
            }

            with open(f"{self.dout}/{t.sample}-info.json", "w") as f:
                json.dump(info, f, indent=4)
        signal.apply(cache_signal)

        if any(bkg.objs):
            bkg_info = {
                fc.cleanpath(t.filelist[0].fname):dict(
                        abs_norm = t.abs_norm,
                        sample_abs_norm = t.sample_abs_norm,
                        max_sample_abs_norm = t.max_sample_abs_norm
                )
                for t in bkg
            }

            with open(f"{self.dout}/bkg-info.json", "w") as f:
                json.dump(bkg_info, f, indent=4)

    #################################################
    def is_bkg(self, signal, bkg):
        signal.apply(lambda t : t.extend(is_bkg=ak.zeros_like(t.Run)))
        bkg.apply(lambda t : t.extend(is_bkg=ak.ones_like(t.Run)))

    def write_trees(self, signal, bkg):
        include=['^jet','^X','.*scale$','is_bkg','gen_X_m','gen_Y1_m','gen_Y_m']

        if any(signal.objs):
            signal.write(
                'fully_res_{base}',
                include=include,
            )

        if any(bkg.objs):
            bkg.write(
                'training_{base}',
                include=include,
            )