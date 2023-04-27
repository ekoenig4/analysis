import os, subprocess

from .eos import *

def check_accstudies(fn):
    if eos.exists(fn): return fn
    fn = fn.replace("_accstudies.root","/ntuple.root")
    return fn

def check_ttjets(fn):
  if eos.exists(fn): return [fn]
  return [ fn.replace('ntuple.root',f'ntuple_{i}.root') for i in range(1) ]
  # return [ fn.replace('ntuple.root',f'ntuple_{i}.root') for i in range(2) ]
  # return [ fn.replace('ntuple.root',f'ntuple_{i}.root') for i in range(8) ]


def sample_files(path):
    NMSSM_XYY_YToHH_8b_MX_1200_MY_500 = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1200_MY_500_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_1000_MY_300 = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1000_MY_300_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_1000_MY_450 = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1000_MY_450_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_700_MY_300  =  check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_700_MY_300_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_800_MY_300  =  check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_800_MY_300_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_800_MY_350  =  check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_800_MY_350_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_900_MY_300  =  check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_900_MY_300_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_900_MY_400  =  check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_900_MY_400_accstudies.root")

    # 12_4_8_500k
    NMSSM_XYY_YToHH_8b_MX_500_MY_250   = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_500_MY_250_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_1400_MY_250  = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1400_MY_250_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_1400_MY_450  = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1400_MY_450_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_1400_MY_700  = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1400_MY_700_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_2000_MY_250  = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_2000_MY_250_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_2000_MY_600  = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_2000_MY_600_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_2000_MY_1000 = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_2000_MY_1000_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_2800_MY_250  = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_2800_MY_250_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_2800_MY_800  = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_2800_MY_800_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_2800_MY_1400 = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_2800_MY_1400_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_4000_MY_250  = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_4000_MY_250_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_4000_MY_1100 = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_4000_MY_1100_accstudies.root")
    NMSSM_XYY_YToHH_8b_MX_4000_MY_2000 = check_accstudies(f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_4000_MY_2000_accstudies.root")

    NMSSM_XYY_YToHH_8b_MX_600_MY_250 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_600_MY_250/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_600_MY_300 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_600_MY_300/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_700_MY_250 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_700_MY_250/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_700_MY_350 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_700_MY_350/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_800_MY_250 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_800_MY_250/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_800_MY_400 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_800_MY_400/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_900_MY_250 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_900_MY_250/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_900_MY_350 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_900_MY_350/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_900_MY_450 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_900_MY_450/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1000_MY_250 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1000_MY_250/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1000_MY_350 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1000_MY_350/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1000_MY_400 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1000_MY_400/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1000_MY_500 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1000_MY_500/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1100_MY_250 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1100_MY_250/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1100_MY_300 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1100_MY_300/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1100_MY_350 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1100_MY_350/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1100_MY_400 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1100_MY_400/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1100_MY_450 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1100_MY_450/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1100_MY_500 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1100_MY_500/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1200_MY_250 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1200_MY_250/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1200_MY_300 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1200_MY_300/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1200_MY_350 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1200_MY_350/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1200_MY_400 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1200_MY_400/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1200_MY_450 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1200_MY_450/ntuple.root"
    NMSSM_XYY_YToHH_8b_MX_1200_MY_600 = f"{path}/NMSSM_XYY_YToHH_8b/NMSSM_XYY_YToHH_8b_MX_1200_MY_600/ntuple.root"

    full_signal_list = [
        NMSSM_XYY_YToHH_8b_MX_700_MY_300 ,
        NMSSM_XYY_YToHH_8b_MX_800_MY_300 ,
        NMSSM_XYY_YToHH_8b_MX_800_MY_350 ,
        NMSSM_XYY_YToHH_8b_MX_900_MY_300 ,
        NMSSM_XYY_YToHH_8b_MX_900_MY_400 ,
        NMSSM_XYY_YToHH_8b_MX_1000_MY_300,
        NMSSM_XYY_YToHH_8b_MX_1000_MY_450,
        NMSSM_XYY_YToHH_8b_MX_1200_MY_500
    ]

    feynnet_signal_list = [
      NMSSM_XYY_YToHH_8b_MX_600_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_600_MY_300 ,
      NMSSM_XYY_YToHH_8b_MX_700_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_700_MY_300  ,
      NMSSM_XYY_YToHH_8b_MX_700_MY_350 ,
      NMSSM_XYY_YToHH_8b_MX_800_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_800_MY_300  ,
      NMSSM_XYY_YToHH_8b_MX_800_MY_350  ,
      NMSSM_XYY_YToHH_8b_MX_800_MY_400 ,
      NMSSM_XYY_YToHH_8b_MX_900_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_900_MY_300  ,
      NMSSM_XYY_YToHH_8b_MX_900_MY_350 ,
      NMSSM_XYY_YToHH_8b_MX_900_MY_400  ,
      NMSSM_XYY_YToHH_8b_MX_900_MY_450 ,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_250,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_300 ,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_350,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_400,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_450 ,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_500,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_250,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_300,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_350,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_400,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_450,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_500,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_250,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_300,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_350,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_400,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_450,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_500 ,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_600,
    ]

    extended_signal_list = [
      NMSSM_XYY_YToHH_8b_MX_500_MY_250  ,
      NMSSM_XYY_YToHH_8b_MX_600_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_600_MY_300 ,
      NMSSM_XYY_YToHH_8b_MX_700_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_700_MY_300  ,
      NMSSM_XYY_YToHH_8b_MX_700_MY_350 ,
      NMSSM_XYY_YToHH_8b_MX_800_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_800_MY_300  ,
      NMSSM_XYY_YToHH_8b_MX_800_MY_350  ,
      NMSSM_XYY_YToHH_8b_MX_800_MY_400 ,
      NMSSM_XYY_YToHH_8b_MX_900_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_900_MY_300  ,
      NMSSM_XYY_YToHH_8b_MX_900_MY_350 ,
      NMSSM_XYY_YToHH_8b_MX_900_MY_400  ,
      NMSSM_XYY_YToHH_8b_MX_900_MY_450 ,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_250,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_300 ,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_350,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_400,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_450 ,
      NMSSM_XYY_YToHH_8b_MX_1000_MY_500,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_250,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_300,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_350,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_400,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_450,
      NMSSM_XYY_YToHH_8b_MX_1100_MY_500,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_250,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_300,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_350,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_400,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_450,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_500 ,
      NMSSM_XYY_YToHH_8b_MX_1200_MY_600,
      NMSSM_XYY_YToHH_8b_MX_1400_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_1400_MY_450 ,
      NMSSM_XYY_YToHH_8b_MX_1400_MY_700 ,
      NMSSM_XYY_YToHH_8b_MX_2000_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_2000_MY_600 ,
      NMSSM_XYY_YToHH_8b_MX_2000_MY_1000,
      NMSSM_XYY_YToHH_8b_MX_2800_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_2800_MY_800 ,
      NMSSM_XYY_YToHH_8b_MX_2800_MY_1400,
      NMSSM_XYY_YToHH_8b_MX_4000_MY_250 ,
      NMSSM_XYY_YToHH_8b_MX_4000_MY_1100,
      NMSSM_XYY_YToHH_8b_MX_4000_MY_2000,
    ]

    signal_list = [
        NMSSM_XYY_YToHH_8b_MX_700_MY_300,
        NMSSM_XYY_YToHH_8b_MX_1000_MY_450,
        NMSSM_XYY_YToHH_8b_MX_1200_MY_500
    ]

    
    NMSSM_XYH_YToHH_6b_MX_700_MY_300  =  f"{path}/NMSSM_XYH_YToHH_6b/NMSSM_XToYHTo6B_MX-700_MY-300_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    NMSSM_XYH_YToHH_6b_MX_800_MY_300  =  f"{path}/NMSSM_XYH_YToHH_6b/NMSSM_XToYHTo6B_MX-800_MY-300_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    NMSSM_XYH_YToHH_6b_MX_800_MY_350  =  f"{path}/NMSSM_XYH_YToHH_6b/NMSSM_XToYHTo6B_MX-800_MY-350_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    NMSSM_XYH_YToHH_6b_MX_900_MY_300  =  f"{path}/NMSSM_XYH_YToHH_6b/NMSSM_XToYHTo6B_MX-900_MY-300_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    NMSSM_XYH_YToHH_6b_MX_900_MY_400  =  f"{path}/NMSSM_XYH_YToHH_6b/NMSSM_XToYHTo6B_MX-900_MY-400_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    NMSSM_XYH_YToHH_6b_MX_1000_MY_350 = f"{path}/NMSSM_XYH_YToHH_6b/NMSSM_XToYHTo6B_MX-1000_MY-350_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    NMSSM_XYH_YToHH_6b_MX_1000_MY_450 = f"{path}/NMSSM_XYH_YToHH_6b/NMSSM_XToYHTo6B_MX-1000_MY-450_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"
    NMSSM_XYH_YToHH_6b_MX_1200_MY_450 = f"{path}/NMSSM_XYH_YToHH_6b/NMSSM_XToYHTo6B_MX-1200_MY-450_TuneCP5_13TeV-madgraph-pythia8/ntuple.root"

    signal_6b_list = [
      NMSSM_XYH_YToHH_6b_MX_700_MY_300, 
      NMSSM_XYH_YToHH_6b_MX_1000_MY_350,
      NMSSM_XYH_YToHH_6b_MX_1200_MY_450,
    ]

    full_signal_6b_list = [
      NMSSM_XYH_YToHH_6b_MX_700_MY_300, 
      NMSSM_XYH_YToHH_6b_MX_800_MY_300, 
      NMSSM_XYH_YToHH_6b_MX_800_MY_350, 
      NMSSM_XYH_YToHH_6b_MX_900_MY_300, 
      NMSSM_XYH_YToHH_6b_MX_900_MY_400, 
      NMSSM_XYH_YToHH_6b_MX_1000_MY_350,
      NMSSM_XYH_YToHH_6b_MX_1000_MY_450,
      NMSSM_XYH_YToHH_6b_MX_1200_MY_450,
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

    TTJets_old = check_ttjets(f"{path}/TTJets/TTJets/ntuple.root")
    TTJets = check_ttjets(f"{path}/TTJets/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/ntuple.root")
    TTTo2L2Nu = check_ttjets(f"{path}/TTJets/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/ntuple.root")
    TTToHadronic = check_ttjets(f"{path}/TTJets/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/ntuple.root")
    TTToSemiLeptonic = check_ttjets(f"{path}/TTJets/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/ntuple.root")

    TT = TTTo2L2Nu + TTToSemiLeptonic

    Bkg_MC_List = QCD_B_List + TTJets


    JetHT_Run2018A_UL = f"{path}/JetHT_Data/JetHT_Run2018A/ntuple.root"
    JetHT_Run2018B_UL = f"{path}/JetHT_Data/JetHT_Run2018B/ntuple.root"
    JetHT_Run2018C_UL = f"{path}/JetHT_Data/JetHT_Run2018C/ntuple.root"
    JetHT_Run2018D_UL = f"{path}/JetHT_Data/JetHT_Run2018D/ntuple.root"

    JetHT_Data_UL_List = [JetHT_Run2018A_UL,
                        JetHT_Run2018B_UL, JetHT_Run2018C_UL, JetHT_Run2018D_UL]

    return locals()



class FileCollection:

  def __init__(self, path='.'):
    self.path = os.path.abspath(path)
    self.contents = None

    self.__dict__.update( sample_files(self.path) )

  def get(self, fname):
    return os.path.join(self.path, fname)

  @property
  def eos(self):
    if self.path.startswith('/store/user/'):
      path = os.path.join('/eos/uscms/', self.path)
      return FileCollection(path)
    return self

  @property
  def store(self):
    if self.path.startswith('/eos/uscms/'):
      path = self.path.replace('/eos/uscms/','/store/user/')
      return FileCollection(path)
    return self

  def __getattr__(self, f): 
    path = f'{self.path}/{f}'
    if not eos.exists(path): raise AttributeError(f'{path} could not be found')
    return FileCollection(path)

  @property
  def ls(self):
    if self.contents is None:
      self.contents = eos.ls(self.path)
    return self.contents

  @property
  def Run2_UL18(self):
    path = os.path.join(self.path, 'Run2_UL', 'RunIISummer20UL18NanoAODv9')
    return FileCollection(path)
    
  @property
  def Run2_UL17(self):
    path = os.path.join(self.path, 'Run2_UL', 'RunIISummer20UL17NanoAODv9')
    return FileCollection(path)

  @property
  def Run2_UL16(self):
    path = os.path.join(self.path, 'Run2_UL', 'RunIISummer20UL16NanoAODv9')
    return FileCollection(path)
    
  @property
  def Run2_Au18(self):
    path = os.path.join(self.path, 'Run2_Autumn18')
    return FileCollection(path)

  def __str__(self): return self.path
  def __repr__(self): 
    ls='\n'.join([ f'   {f}' for f in self.ls])
    return f"{self.path}\n{ls}"