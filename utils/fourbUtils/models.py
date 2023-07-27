import os
import yaml


basepath = '/uscms_data/d3/ekoenig/8BAnalysis/studies/weaver-multiH/weaver/models'
storage = '/eos/uscms/store/user/ekoenig/weaver/models/'

from ..weaver_tools import WeaverModel as WeaverModelBase
class WeaverModel(WeaverModelBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, basepath=basepath, storage=storage, **kwargs)


feynnet_4b = WeaverModel('exp_fourb/feynnet_4b/20230522_ranger_lr0.0047_batch1024/')
feynnet_4b_5jet = WeaverModel('exp_fourb/feynnet_4b/20230523_ranger_lr0.0047_batch1024_5jet/')
feynnet_4b_6jet_bkg = WeaverModel('exp_fourb/feynnet_4b/20230523_ranger_lr0.0047_batch1024_6jet_withbkg/')
feynnet_4b_6jet_bkg05_hm10 = WeaverModel('exp_fourb/feynnet_4b/20230524_b5b957b4ed1f2a286c340f07c2ff1fe4_ranger_lr0.0047_batch1024_bkg0.5_hm10_withbkg/')

feynnet_etpiece = WeaverModel('exp_fourb/feynnet_4b/20230623_7e4a73e376c8020ebacb48d37ed34444_ranger_lr0.0047_batch1024_withbkg/')











def get_model_path(model, locals=locals()):
    if model not in locals: return model 
    return locals[model].path

def get_model(model, locals=locals()):
    if model not in locals: return model 
    return locals[model]