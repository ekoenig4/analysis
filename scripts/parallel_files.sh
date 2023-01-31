#!/bin/sh

get_files() {
    path=/eos/uscms/store/user/ekoenig/8BAnalysis/NTuples/2018/preselection/t8btag_minmass/

    for f in $(ls $path/NMSSM_XYY_YToHH_8b/*/ntuple.root); do
        echo ${f/\/eos\/uscms/}
    done
    
    for f in $(ls $path/Run2_UL/RunIISummer20UL18NanoAODv9/QCD/*/ntuple.root); do
        echo ${f/\/eos\/uscms/}
    done
    
    for f in $(ls $path/Run2_UL/RunIISummer20UL18NanoAODv9/TTJets/TTJets*/ntuple_{0,1}.root); do
        echo ${f/\/eos\/uscms/}
    done
    
    for f in $(ls $path/Run2_UL/RunIISummer20UL18NanoAODv9/JetHT_Data/*/ntuple.root); do
        echo ${f/\/eos\/uscms/}
    done
}

get_training() {
    path=/eos/uscms/store/user/ekoenig/8BAnalysis/NTuples/2018/training/training_5M/

    # for f in $(ls $path/NMSSM_XYY_YToHH_8b/*{MX_700_MY_300,MX_1000_MY_450,MX_1200_MY_500}*/ntuple*.root); do
    #     echo ${f/\/eos\/uscms/}
    # done
    
    for f in $(ls $path/NMSSM_XYY_YToHH_8b/*{MX_700_MY_300,MX_800_MY_300,MX_800_MY_350,MX_900_MY_300,MX_900_MY_400,MX_1000_MY_300,MX_1000_MY_450,MX_1200_MY_500}*/ntuple*.root); do
        echo ${f/\/eos\/uscms/}
    done


    path=/eos/uscms/store/user/ekoenig/8BAnalysis/NTuples/2018/preselection/t8btag_minmass/

    # for f in $(ls $path/Run2_Autumn18//QCD/*/ntuple.root); do
    #     echo ${f/\/eos\/uscms/}
    # done

    # for f in $(ls $path/Run2_UL/RunIISummer20UL18NanoAODv9/TTJets/TTJets*/ntuple_{6,7}.root); do
    #     echo ${f/\/eos\/uscms/}
    # done
}


run_function() {
    ./run_files.py $@
}

export -f run_function

time get_files | parallel -j 4 -k run_function $@
# time get_training | parallel -j 6 -k run_function $@
