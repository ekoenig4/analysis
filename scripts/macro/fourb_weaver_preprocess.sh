set -e

ntuples=filelist/4b/ForFeynNet_UL18_SignalPlusBackground_27June2023_training.txt
signal_ntuples=$(cat $ntuples | grep GluGlu )
background_ntuples=$(cat $ntuples | grep -v GluGlu )

script=notebooks/skims/fourb_weaver_input.py
njet=6

dout=reweight-${njet}jet-4b-info

echo Generating reweighting information for ${njet} jets with 4b
python $script --dout ${dout} --cache ${background_ntuples} --njet ${njet}
python $script --dout ${dout} --cache ${signal_ntuples} --njet ${njet}
# echo $signal_ntuples | xargs -n 1 | parallel -j 8 --eta python $script --dout ${dout} --cache --njet 6


echo Applying reweighting information for ${njet} jets with 4b
# echo $signal_ntuples $background_ntuples | xargs -n 1 | parallel -j 8 --eta python $script --dout ${dout} --apply  --njet 6

for f in $(echo $signal_ntuples $background_ntuples); do
	python $script --dout ${dout} --apply --njet ${njet} $f
done
