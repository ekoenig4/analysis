set -e

python notebooks/skims/sixb_feynnet_bdt.py

export total=119
export njobs=119
export nfiles=$((total/njobs))

run() {
    python notebooks/skims/sixb_feynnet_limits.py --input $1
}

export -f run

# seq 0 $((total-1)) | xargs -n 1 | parallel --jobs 4 --bar --verbose run

# run jobs in parallel with 8 jobs at a time
seq 0 $((njobs-1)) | xargs -n 1 | parallel --jobs 8 --bar --verbose --ungroup run

