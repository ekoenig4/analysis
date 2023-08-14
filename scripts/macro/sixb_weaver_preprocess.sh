set -e

ntuples=filelist/6b/feynnet_2018.txt
signal_ntuples=$(cat $ntuples | grep NMSSM)
# background_ntuples=$(cat $ntuples | grep -v NMSSM )

script=notebooks/skims/sixb_weaver_input.py

dout=reweight-6b-info

echo Generating reweighting information
# python $script --dout ${dout} --cache ${background_ntuples} 
# python $script --dout ${dout} --cache ${signal_ntuples}
parallel --jobs 8 --bar --verbose --line-buffer python $script --dout ${dout} --cache {} ::: $(echo $signal_ntuples)

echo Applying reweighting information
parallel --jobs 8 --bar --verbose --line-buffer python $script --dout ${dout} --apply {} ::: $(echo $signal_ntuples)
