set -e

ntuples=filelist/8b/feynnet_trainlist.txt
signal_ntuples=$(cat $ntuples | grep NMSSM | sed -e 's|/eos/uscms||g')
background_ntuples=$(cat $ntuples | grep -v NMSSM | sed -e 's|/eos/uscms||g')
dout=reweight-8b-info

echo Generating reweighting information for 8b
# python notebooks/skims/eightb_weaver_input.py --dout ${dout} --cache ${background_ntuples}
# echo $signal_ntuples | xargs -n 1 | parallel -j 8 --eta python notebooks/skims/eightb_weaver_input.py --dout ${dout} --cache


echo Applying reweighting information for 8b
echo $signal_ntuples $background_ntuples | xargs -n 1 | parallel -j 8 --eta python notebooks/skims/eightb_weaver_input.py --dout ${dout} --apply 

echo Splitting reweighted files into 0.8/0.2 train/validation sets
reweight_ntuples="$(echo ${signal_ntuples} | sed -e 's|ntuple|reweight_eightb_ntuple|g') $(echo ${background_ntuples} | sed -e 's|ntuple|reweight_ntuple|g')"

split() {
	python notebooks/skims/randomize_split.py $1 --frac 0.8 0.2
}
export -f split

echo $reweight_ntuples | xargs -n 1 | parallel -j 8 --eta split


