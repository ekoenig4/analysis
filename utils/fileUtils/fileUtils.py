import os

def sample_files(path):
    NMSSM_XYY_YToHH_8b_MX_1200_MY_500 = path + \
        "/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1200_MY_500_accstudies.root"
    NMSSM_XYY_YToHH_8b_MX_1000_MY_300 = path + \
        "/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1000_MY_300_accstudies.root"
    NMSSM_XYY_YToHH_8b_MX_1000_MY_450 = path + \
        "/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1000_MY_450_accstudies.root"
    NMSSM_XYY_YToHH_8b_MX_700_MY_300 = path + \
        "/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_700_MY_300_accstudies.root"
    NMSSM_XYY_YToHH_8b_MX_800_MY_300 = path + \
        "/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_800_MY_300_accstudies.root"
    NMSSM_XYY_YToHH_8b_MX_800_MY_350 = path + \
        "/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_800_MY_350_accstudies.root"
    NMSSM_XYY_YToHH_8b_MX_900_MY_300 = path + \
        "/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_900_MY_300_accstudies.root"
    NMSSM_XYY_YToHH_8b_MX_900_MY_400 = path + \
        "/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_900_MY_400_accstudies.root"

    signal_list = [
        NMSSM_XYY_YToHH_8b_MX_700_MY_300,
        NMSSM_XYY_YToHH_8b_MX_800_MY_300,
        NMSSM_XYY_YToHH_8b_MX_800_MY_350,
        NMSSM_XYY_YToHH_8b_MX_900_MY_300,
        NMSSM_XYY_YToHH_8b_MX_900_MY_400,
        NMSSM_XYY_YToHH_8b_MX_1000_MY_300,
        NMSSM_XYY_YToHH_8b_MX_1000_MY_450,
        NMSSM_XYY_YToHH_8b_MX_1200_MY_500
    ]

    QCD_bEn_Ht_100to200 = f"{path}/QCD/QCD_bEnriched_HT100to200_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bEn_Ht_200to300 = f"{path}/QCD/QCD_bEnriched_HT200to300_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bEn_Ht_300to500 = f"{path}/QCD/QCD_bEnriched_HT300to500_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bEn_Ht_500to700 = f"{path}/QCD/QCD_bEnriched_HT500to700_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bEn_Ht_700to1000 = f"{path}/QCD/QCD_bEnriched_HT700to1000_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bEn_Ht_1000to1500 = f"{path}/QCD/QCD_bEnriched_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bEn_Ht_1500to2000 = f"{path}/QCD/QCD_bEnriched_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bEn_Ht_2000toInf = f"{path}/QCD/QCD_bEnriched_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"

    QCD_bEn_List = [QCD_bEn_Ht_100to200, QCD_bEn_Ht_200to300, QCD_bEn_Ht_300to500, QCD_bEn_Ht_500to700,
                    QCD_bEn_Ht_700to1000, QCD_bEn_Ht_1000to1500, QCD_bEn_Ht_1500to2000, QCD_bEn_Ht_2000toInf]

    QCD_bGf_Ht_100to200 = f"{path}/QCD/QCD_HT100to200_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bGf_Ht_200to300 = f"{path}/QCD/QCD_HT200to300_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bGf_Ht_300to500 = f"{path}/QCD/QCD_HT300to500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bGf_Ht_500to700 = f"{path}/QCD/QCD_HT500to700_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bGf_Ht_700to1000 = f"{path}/QCD/QCD_HT700to1000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bGf_Ht_1000to1500 = f"{path}/QCD/QCD_HT1000to1500_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bGf_Ht_1500to2000 = f"{path}/QCD/QCD_HT1500to2000_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    QCD_bGf_Ht_2000toInf = f"{path}/QCD/QCD_HT2000toInf_BGenFilter_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"

    QCD_bGf_List = [QCD_bGf_Ht_100to200, QCD_bGf_Ht_200to300, QCD_bGf_Ht_300to500, QCD_bGf_Ht_500to700,
                    QCD_bGf_Ht_700to1000, QCD_bGf_Ht_1000to1500, QCD_bGf_Ht_1500to2000, QCD_bGf_Ht_2000toInf]

    QCD_B_List = QCD_bEn_List + QCD_bGf_List

    TTJets = f"{path}/TTJets/TTJets/ntuple.root"

    Bkg_MC_List = QCD_B_List + [TTJets]


    JetHT_Run2018A_UL = f"{path}/JetHT_Data/JetHT_Run2018A/ntuple.root"
    JetHT_Run2018B_UL = f"{path}/JetHT_Data/JetHT_Run2018B/ntuple.root"
    JetHT_Run2018C_UL = f"{path}/JetHT_Data/JetHT_Run2018C/ntuple.root"
    JetHT_Run2018D_UL = f"{path}/JetHT_Data/JetHT_Run2018D/ntuple.root"

    JetHT_Data_UL_List = [JetHT_Run2018A_UL,
                        JetHT_Run2018B_UL, JetHT_Run2018C_UL, JetHT_Run2018D_UL]

    return locals()

class FileCollection:

  def __init__(self, path='/eos/uscms/store/user/ekoenig/8BAnalysis/NTuples/2018/'):
    self.path = os.path.abspath(path)

    for key, value in sample_files(self.path).items():
      setattr(self, key, value)

  def __getattr__(self, f):
    path = f'{self.path}/{f}'
    if not os.path.exists(path): raise AttributeError(f'{path} could not be found')
    return FileCollection(path)

  @property
  def ls(self):
    return os.listdir(self.path)

  def __str__(self): return self.path
  def __repr__(self): 
    ls='\n'.join([ f'   {f}' for f in self.ls])
    return f"{self.path}\n{ls}"


eightb = FileCollection()