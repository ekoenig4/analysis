import os
import yaml


basepath = '/uscms_data/d3/ekoenig/8BAnalysis/studies/weaver-multiH/weaver/models'
storage = '/eos/uscms/store/user/ekoenig/weaver/models/'


from ..weaver_tools import WeaverModel as WeaverModelBase
class WeaverModel(WeaverModelBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, basepath=basepath, storage=storage, **kwargs)


feynnet_sig = WeaverModel('exp_feynnet_paper/feynnet_6b/20230410_ranger_lr0.0047_batch1024')
feynnet_bkg = WeaverModel('exp_feynnet_paper/feynnet_6b/20230410_ranger_lr0.0047_batch1024_withbkg')
feynnet_sig_mass_square = WeaverModel('exp_feynnet_paper/feynnet_6b/20230413_ranger_lr0.0047_batch1024_mass_square')
 
def get_model_path(model, locals=locals()):
    if model not in locals: return model 
    return locals[model].path

def get_model(model, locals=locals()):
    if model not in locals: return model 
    return locals[model]
