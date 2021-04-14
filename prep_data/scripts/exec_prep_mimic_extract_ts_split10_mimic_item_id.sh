#!/usr/bin/env bash

source activate py27

if [[ $1 = "testmode" ]]; then
    test_opt="--testmode"
else
    test_opt=""
fi

cd ../
python prep_mimic_extract_ts_adhoc.py \
        --excl-lab-abnormal \
        --not-allow-load \
        --multi-split 10 \
        --max-span-day 20 \
        --min-span-day 2 \
        ${test_opt} \
        --skip-valid
cd -