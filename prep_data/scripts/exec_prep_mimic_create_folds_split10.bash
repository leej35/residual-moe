#!/usr/bin/env bash

cd ../

if [[ $2 = "testmode" ]]; then
    test_str="_TEST"
else
    test_str=""
fi

for hr in $1 ; do
    for split_id in "1"; do # 
        for num_folds in "5"; do
            python prep_mimic_create_folds.py \
                --data-path data_path/mimic.sequence/mimic_train_xhr_${hr}_yhr_${hr}_ytype_multi_event_exclablab${test_str}_singleseq_mimicid_elapsedt_minsd_2_maxsd_20_sv/split_${split_id} --num-folds ${num_folds} \
                --remapped-data &
        done &
    done
done

cd -