from . import *


tagMap = {
    "QCD":"QCD",
    "NMSSM":"Signal",
    "TTJets":"TTJets",
    "Data":"Data",
}

colorMap = {
    "QCD":"tab:blue",
    "Signal":"tab:orange",
    "TTJets":"tab:green",
    "Data":"black",
    "Bkg":"grey"
}

xsecMap = {
    "JetHT_Run2018A":'N/A',
    "JetHT_Run2018B":'N/A',
    "JetHT_Run2018C":'N/A',
    "JetHT_Run2018D":'N/A',
    
    "NMSSM":0.3,
    
    "QCD_Pt_15to30"    :1246000000.0,
    "QCD_Pt_30to50"    :106500000.0,
    "QCD_Pt_50to80"    :15700000.0,
    "QCD_Pt_80to120"   :2346000.0,
    "QCD_Pt_120to170"  :407300.0,
    "QCD_Pt_170to300"  :103500.0,
    "QCD_Pt_300to470"  :6826.0,
    "QCD_Pt_470to600"  :552.1,
    "QCD_Pt_600to800"  :156.5,
    "QCD_Pt_800to1000" :26.28,
    "QCD_Pt_1000to1400":7.465,
    "QCD_Pt_1400to1800":0.6484,
    "QCD_Pt_1800to2400":0.08734,
    "QCD_Pt_2400to3200":0.005237,
    "QCD_Pt_3200toInf" :0.000135,
    
    "QCD_bEnriched_HT100to200"  :1127000.0,
    "QCD_bEnriched_HT200to300"  :80430.0,
    "QCD_bEnriched_HT300to500"  :16620.0,
    "QCD_bEnriched_HT500to700"  :1487.0,
    "QCD_bEnriched_HT700to1000" :296.5,
    "QCD_bEnriched_HT1000to1500":46.61,
    "QCD_bEnriched_HT1500to2000":3.72,
    "QCD_bEnriched_HT2000toInf" :0.6462,
    
    "QCD_HT100to200_BGenFilter"  :1275000.0,
    "QCD_HT200to300_BGenFilter"  :111700.0,
    "QCD_HT300to500_BGenFilter"  :27960.0,
    "QCD_HT500to700_BGenFilter"  :3078.0,
    "QCD_HT700to1000_BGenFilter" :721.8,
    "QCD_HT1000to1500_BGenFilter":138.2,
    "QCD_HT1500to2000_BGenFilter":13.61,
    "QCD_HT2000toInf_BGenFilter" :2.92,

    "TTJets":831.76,
}
