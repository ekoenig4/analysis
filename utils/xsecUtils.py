import matplotlib.pyplot as plt
from itertools import cycle
lumiMap = {
    None: [1, None],
    2016: [35900, "(13 TeV,2016)"],
    2017: [41500, "(13 TeV,2017)"],
    2018: [59740, "(13 TeV,2018)"],
    20180: [14300, "(13 TeV,2018 A)"],
    20181: [7070, "(13 TeV,2018 B)"],
    20182: [6900, "(13 TeV,2018 C)"],
    20183: [13540, "(13 TeV,2018 D)"],
    "Run2": [101000, "13 TeV,Run 2)"],
}

tagMap = {
    "QCD": "QCD",
    "NMSSM_XYH_YToHH_6b": "X$\\rightarrow$ YH$\\rightarrow$3H$\\rightarrow$6b",
    "NMSSM_XYY_YToHH_8b": "X$\\rightarrow$ YY$\\rightarrow$4H$\\rightarrow$8b",
    "TT": "TTJets",
    "Data": "Data",
}


colorMap = {    
    "X$\\rightarrow$ YH$\\rightarrow$3H$\\rightarrow$6b": "orange",
    "X$\\rightarrow$ YY$\\rightarrow$4H$\\rightarrow$8b": iter(cycle(['tomato', 'royalblue', 'mediumorchid'])),
    "MX_700_MY_300": 'tomato',
    "MX_1000_MY_450": 'royalblue',
    "MX_1200_MY_500": 'mediumorchid',
    "QCD": "lightblue",
    "TTJets": "darkseagreen",
    "Data": "black",
    "MC-Bkg": "grey"
}

xsecMap = {
    "JetHT_Run2018A": 'N/A',
    "JetHT_Run2018B": 'N/A',
    "JetHT_Run2018C": 'N/A',
    "JetHT_Run2018D": 'N/A',

    "NMSSM_XYH_YToHH_6b": 1.0 * (5.824E-01)**3,
    # "NMSSM_XYY_YToHH_8b": 0.0016,
    # "NMSSM_XYY_YToHH_8b": 0.3,
    "NMSSM_XYY_YToHH_8b": 1.0 * (5.824E-01)**4,

    "GluGluToHHTo4B" : 1.0 * (5.824E-01)**2,

    "QCD_Pt_15to30": 1246000000.0,
    "QCD_Pt_30to50": 106500000.0,
    "QCD_Pt_50to80": 15700000.0,
    "QCD_Pt_80to120": 2346000.0,
    "QCD_Pt_120to170": 407300.0,
    "QCD_Pt_170to300": 103500.0,
    "QCD_Pt_300to470": 6826.0,
    "QCD_Pt_470to600": 552.1,
    "QCD_Pt_600to800": 156.5,
    "QCD_Pt_800to1000": 26.28,
    "QCD_Pt_1000to1400": 7.465,
    "QCD_Pt_1400to1800": 0.6484,
    "QCD_Pt_1800to2400": 0.08734,
    "QCD_Pt_2400to3200": 0.005237,
    "QCD_Pt_3200toInf": 0.000135,

    # previous 
    # "QCD_bEnriched_HT100to200"  : 1127000.0,
    # "QCD_bEnriched_HT200to300"  : 80430.0  ,
    # "QCD_bEnriched_HT300to500"  : 16620.0  ,
    # "QCD_bEnriched_HT500to700"  : 1487.0   ,
    # "QCD_bEnriched_HT700to1000" : 296.5    ,
    # "QCD_bEnriched_HT1000to1500": 46.61    ,
    # "QCD_bEnriched_HT1500to2000": 3.72     ,
    # "QCD_bEnriched_HT2000toInf" : 0.6462   ,

    "QCD_bEnriched_HT100to200"  : 1122000.00,
    "QCD_bEnriched_HT200to300"  : 79760.00,
    "QCD_bEnriched_HT300to500"  : 16600.00,
    "QCD_bEnriched_HT500to700"  : 1503.000,
    "QCD_bEnriched_HT700to1000" : 297.400,
    "QCD_bEnriched_HT1000to1500": 48.0800,
    "QCD_bEnriched_HT1500to2000": 3.95100,
    "QCD_bEnriched_HT2000toInf" : 0.695700,

    # previous 
    # "QCD_HT100to200_BGenFilter"  : 1275000.0,
    # "QCD_HT200to300_BGenFilter"  : 111700.0 ,
    # "QCD_HT300to500_BGenFilter"  : 27960.0  ,
    # "QCD_HT500to700_BGenFilter"  : 3078.0   ,
    # "QCD_HT700to1000_BGenFilter" : 721.8    ,
    # "QCD_HT1000to1500_BGenFilter": 138.2    ,
    # "QCD_HT1500to2000_BGenFilter": 13.61    ,
    # "QCD_HT2000toInf_BGenFilter" : 2.92     ,

    "QCD_HT100to200_BGenFilter"  : 1266000.00,
    "QCD_HT200to300_BGenFilter"  : 109900.00,
    "QCD_HT300to500_BGenFilter"  : 27360.00,
    "QCD_HT500to700_BGenFilter"  : 2991.00,
    "QCD_HT700to1000_BGenFilter" : 731.80,
    "QCD_HT1000to1500_BGenFilter": 139.300,
    "QCD_HT1500to2000_BGenFilter": 14.7400,
    "QCD_HT2000toInf_BGenFilter" : 3.0900,
 
    "TTTo2L2Nu":831.76*0.3259*0.3259,
    "TTToSemiLeptonic":831.76*2*0.6741*0.3259,
    "TTToHadronic":831.76*0.6741*0.6741,
    
    "TTJets": 734.60,
    # "TTJets": 831.76,
    # "TTJets": 750.5,
}
