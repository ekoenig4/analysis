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
    
    for f in $(ls $path/NMSSM_XYY_YToHH_8b/*/ntuple.root); do
        echo ${f/\/eos\/uscms/}
    done


    # path=/eos/uscms/store/user/ekoenig/8BAnalysis/NTuples/2018/preselection/t8btag_minmass/

    # for f in $(ls $path/Run2_Autumn18//QCD/*/ntuple.root); do
    #     echo ${f/\/eos\/uscms/}
    # done

    # for f in $(ls $path/Run2_UL/RunIISummer20UL18NanoAODv9/TTJets/TTJets*/ntuple_training.root); do
    #     echo ${f/\/eos\/uscms/}
    # done
}

split_training() {
    path=/eos/uscms/store/user/ekoenig/8BAnalysis/NTuples/2018/training/

    for f in $(find $path -name training_ntuple.root); do
        echo ${f/\/eos\/uscms/}
    done

}

split_sixb_training() {
    for f in $(find /eos/uscms/store/user/ekoenig/sixb/ntuples/Summer2018UL/maxbtag/NMSSM -name fully_res_ntuple.root); do 
        echo ${f/\/eos\/uscms/}
    done
}

ttbar_training() {
    path=/eos/uscms/store/user/ekoenig/TTAnalysis/NTuples/2018/preselection/Run2_UL/RunIISummer20UL18NanoAODv9/TTJets/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/

    for f in $(ls $path/ntuple_0.root); do
        echo ${f/\/eos\/uscms/}
    done
    
    path=/eos/uscms/store/user/ekoenig/TTAnalysis/NTuples/2018/preselection/Run2_UL/RunIISummer20UL18NanoAODv9/QCD/

    for f in $(ls $path/*/ntuple_0.root); do
        echo ${f/\/eos\/uscms/}
    done
}

get_feynnet_training() {
    file="reweight*train_ntuple.root"
    path=/eos/uscms/store/user/ekoenig/8BAnalysis/NTuples/2018/feynnet

    for f in $(ls $path/NMSSM_XYY_YToHH_8b/*/$file); do
        echo ${f/\/eos\/uscms/}
    done

    for f in $(ls $path/Run2_UL/RunIISummer20UL18NanoAODv9/QCD/*/$file); do
        echo ${f/\/eos\/uscms/}
    done
}


run_function() {
    echo ./run_files.py $@
    ./run_files.py $@
}

export -f run_function

filelist=$1
shift 1

time ./fetch_files.sh $filelist | parallel --eta -j 12 run_function $@

