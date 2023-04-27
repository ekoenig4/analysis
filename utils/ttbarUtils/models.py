import os
import yaml


basepath = '/uscms_data/d3/ekoenig/8BAnalysis/studies/weaver-multiH/weaver/models'
storage = '/eos/uscms/store/user/ekoenig/weaver/models/'

from ..weaver_tools import WeaverModel as WeaverModelBase
class WeaverModel(WeaverModelBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, basepath=basepath, storage=storage, **kwargs)


feynnet = WeaverModel('exp_ttbar/template_feynnet/20230321_ranger_lr0.0047_batch1024/')
feynnet_btag = WeaverModel('exp_ttbar/template_feynnet/20230330_ranger_lr0.0047_batch1024/')
feynnet_bool_btag = WeaverModel('exp_ttbar/template_feynnet/20230401_ranger_lr0.0047_batch1024/')
feynnet_bool_btag_bkg = WeaverModel('exp_ttbar/template_feynnet/20230401_ranger_lr0.0047_batch1024_withbkg/')

 
def get_model_path(model, locals=locals()):
    if model not in locals: return model 
    return locals[model].path

def get_model(model, locals=locals()):
    if model not in locals: return model 
    return locals[model]
