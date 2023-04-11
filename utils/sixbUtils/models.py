import os
import yaml


basepath = '/uscms_data/d3/ekoenig/8BAnalysis/studies/weaver-multiH/weaver/models'
class WeaverModel:
    def __init__(self, path, desc=None):
        if not os.path.exists(path):
            path = os.path.join(basepath, path)
            if not os.path.exists(path):
                path = None
        self.path = path
        self.load = None
        if self.path is not None:
            loadlist = ['feynnet_6b']
            self.load = next( (load for load in loadlist if load in self.path), None )
        self.desc = desc

    @property
    def cfg(self):
        cfg = getattr(self, '_cfg', None)
        if cfg is None:
            with open(f'{self.path}/lightning_logs/tb/hparams.yaml', 'r') as f:
                self._cfg = yaml.safe_load(f)
        return self._cfg

        

    def __repr__(self):
        return yaml.dump(self.cfg)

    def __str__(self):
        if self.desc is None:
            return repr(self)
        return self.desc


feynnet_sig = WeaverModel('exp_feynnet_paper/feynnet_6b/20230410_ranger_lr0.0047_batch1024')
feynnet_bkg = WeaverModel('exp_feynnet_paper/feynnet_6b/20230410_ranger_lr0.0047_batch1024_withbkg')
 
def get_model_path(model, locals=locals()):
    if model not in locals: return model 
    return locals[model].path

def get_model(model, locals=locals()):
    if model not in locals: return model 
    return locals[model]
