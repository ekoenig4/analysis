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
            loadlist = ['quadh_ranker','yy_4h_reco_ranker']
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


# pn = WeaverModel(f'quadh_ranker/20221115_ranger_lr0.0047_batch512_m7m10m12/')

# mp = WeaverModel(f'quadh_ranker_mp/20221124_ranger_lr0.0047_batch512_m7m10m12/')
# mp300k = WeaverModel(f'quadh_ranker_mp/20221205_ranger_lr0.0047_batch512_m7m10m12_300k/')
# mp500k = WeaverModel(f'quadh_ranker_mp/20221205_ranger_lr0.0047_batch512_m7m10m12_500k/')

# mpbkg00 = WeaverModel(f'quadh_ranker_mp/20221209_b72001172c5d04183ed7bb294252320b_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
# mpbkg005 = WeaverModel(f'quadh_ranker_mp/20221212_293790a7fbfb752ded05771058bf5a25_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
# mpbkg01 = WeaverModel(f'quadh_ranker_mp/20221209_be9efb5b61eb1c42aeb209728eec84d7_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')

# mpbkg01_hard25 = WeaverModel(f'quadh_ranker_mp/20221214_d595a9703289900d701416bb7274ab71_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
# mpbkg35_hard25 = WeaverModel(f'quadh_ranker_mp/20221215_8d087d23e1f72729bdcdd043b3d693e6_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')

# mpbkg01_hard50 = WeaverModel(f'quadh_ranker_mp/20221214_13676d884fa50cdaffb748fc057f180a_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
# mpbkg01_hard50 = WeaverModel(f'quadh_ranker_mp/20221215_13676d884fa50cdaffb748fc057f180a_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')

# mpbkg01_exp = WeaverModel(f'quadh_ranker_mp/20221214_2f889467cb0f6c7a9269c92e93c25c1d_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
# mpbkg05_exp = WeaverModel(f'quadh_ranker_mp/20221214_34452fc51690ae1d20a150a10c0bafa7_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')

# experiment hardcut
mpbkg00 = WeaverModel('exp_hardcut/quadh_ranker_mp/20221221_52ed083e1aa8def4475c5a7f90c66743_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')

mpbkg01_hard25 = WeaverModel('exp_hardcut/quadh_ranker_mp/20221220_d595a9703289900d701416bb7274ab71_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
mpbkg05_hard25 = WeaverModel('exp_hardcut/quadh_ranker_mp/20221220_5a4e703b954d41932dda3000c09be636_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
mpbkg10_hard25 = WeaverModel('exp_hardcut/quadh_ranker_mp/20221220_c3be05f5c45f34a18606ca114e24856d_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
mpbkg20_hard25 = WeaverModel('exp_hardcut/quadh_ranker_mp/20221220_d6c051425d9d07f0e59ff85efdd1e8a5_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')

mpbkg01_hard50 = WeaverModel('exp_hardcut/quadh_ranker_mp/20221220_13676d884fa50cdaffb748fc057f180a_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
mpbkg05_hard50 = WeaverModel('exp_hardcut/quadh_ranker_mp/20221220_0512a8d68083c0eb2c5875d5d670c042_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
mpbkg10_hard50 = WeaverModel('exp_hardcut/quadh_ranker_mp/20221220_dbe056a55e82ce1d89e004942c741bb3_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
mpbkg20_hard50 = WeaverModel('exp_hardcut/quadh_ranker_mp/20221220_dbefbf74d757e3a883244b2ab22b0aa4_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')

# experiment softcut multi
mpbkg10_soft100_md = WeaverModel('exp_multi/quadh_ranker_mp/20221221_91ef478db7c19ab72ebfb2c7763831d0_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
mpbkg20_soft100_md = WeaverModel('exp_multi/quadh_ranker_mp/20221222_9a858186424e04f4ea7534a371509ae1_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')

# YY QuadH
yy_quadh_bkg00 = WeaverModel('exp_yy/yy_4h_reco_ranker/20221223_b72001172c5d04183ed7bb294252320b_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
yy_quadh_bkg10_soft100_md = WeaverModel('exp_yy/yy_4h_reco_ranker/20221227_7a4d53b9cfb89697dd40324814f7aa4f_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
yy_quadh_bkg10_soft025_md = WeaverModel('exp_yy/yy_4h_reco_ranker/20230103_7c14f31f22b318c56b1b4621b416c4a6_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')
yy_quadh_bkg10_asym002_md = WeaverModel('exp_yy/yy_4h_reco_ranker/20230105_7f6d562c1ae48b9fd380ed2ffc27c347_ranger_lr0.0047_batch1024_m7m10m12_withbkg/')

yy_quadh_bkg10_soft025_md_allsig = WeaverModel('exp_yy/yy_4h_reco_ranker/20230125_7c14f31f22b318c56b1b4621b416c4a6_ranger_lr0.0047_batch1024_withbkg/')

yy_quadh_bkg10_allsig = WeaverModel('exp_yy/yy_4h_reco_ranker/20230125_7c14f31f22b318c56b1b4621b416c4a6_ranger_lr0.0047_batch1024_withbkg/')
yy_quadh_bkg10_MX_700_MY_300  =  WeaverModel('exp_mass/yy_4h_reco_ranker/20230216_ranger_lr0.0047_batch1024_MX_700_MY_300_withbkg/')
yy_quadh_bkg10_MX_800_MY_300  =  WeaverModel('exp_mass/yy_4h_reco_ranker/20230216_ranger_lr0.0047_batch1024_MX_800_MY_300_withbkg/')
yy_quadh_bkg10_MX_800_MY_350  =  WeaverModel('exp_mass/yy_4h_reco_ranker/20230216_ranger_lr0.0047_batch1024_MX_800_MY_350_withbkg/')
yy_quadh_bkg10_MX_900_MY_300  =  WeaverModel('exp_mass/yy_4h_reco_ranker/20230216_ranger_lr0.0047_batch1024_MX_900_MY_300_withbkg/')
yy_quadh_bkg10_MX_900_MY_400  =  WeaverModel('exp_mass/yy_4h_reco_ranker/20230216_ranger_lr0.0047_batch1024_MX_900_MY_400_withbkg/')
yy_quadh_bkg10_MX_1000_MY_300 =  WeaverModel('exp_mass/yy_4h_reco_ranker/20230216_ranger_lr0.0047_batch1024_MX_1000_MY_300_withbkg/')
yy_quadh_bkg10_MX_1000_MY_450 =  WeaverModel('exp_mass/yy_4h_reco_ranker/20230216_ranger_lr0.0047_batch1024_MX_1000_MY_450_withbkg/')
yy_quadh_bkg10_MX_1200_MY_500 =  WeaverModel('exp_mass/yy_4h_reco_ranker/20230216_ranger_lr0.0047_batch1024_MX_1200_MY_500_withbkg/')

 
def get_model_path(model, locals=locals()):
    if model not in locals: return model 
    return locals[model].path

def get_model(model, locals=locals()):
    if model not in locals: return model 
    return locals[model]
