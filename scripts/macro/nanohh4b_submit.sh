CMD="sbatch submit.sh python notebooks/analysis/nanohh4b.py"

ARGS=" --no-bkg --only 21 22 23 --load-reweighter /home/ekoenig/analysis/studies/models/nanoHH4b/private/mindiag/medium/bdt_reweighter.pkl"

# $CMD $ARGS

# # submit analysis chain with mindiag
# $CMD --dout bdt-50-estimators --n-classifier-estimators 50 $ARGS 

# $CMD --model /home/ekoenig/analysis/weaver-multiH/models/feynnet_lightning/nanoHH4b/sixjet/training-data-scan/lightning_logs/version_11881856 --dout training-20230925 $ARGS

# # # submit analysis chain with 6jet FeynNet 
$CMD --model /home/ekoenig/analysis/weaver-multiH/models/feynnet_lightning/nanoHH4b/sixjet/training-data-scan/lightning_logs/version_11881857 --only "write_features"

# # # submit analysis chain with 6jet FeynNet with btag L variable
# $CMD --model /home/ekoenig/analysis/weaver-multiH/models/feynnet_lightning/nanoHH4b/sixjet/lightning_logs/version_12177187 --dout training-20231004-btagL $ARGS

# # # submit analysis chain with 6jet FeynNet with btag L variable using leading btagging info ARGS
# $CMD --model /home/ekoenig/analysis/weaver-multiH/models/feynnet_lightning/nanoHH4b/sixjet/lightning_logs/version_12177187 --dout training-20231004-btagL --leading-btag $ARGS