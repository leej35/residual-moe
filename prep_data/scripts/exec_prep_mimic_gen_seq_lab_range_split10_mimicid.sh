#!/usr/bin/env bash

if [[ $2 = "testmode" ]]; then
    test_str="_TEST"
    test_opt="--testmode"
else
    test_str=""
    test_opt=""
fi

cd ../

for hr in "$1" ; do 
    for split_id in "1" ; do #  
        for set in "train" "test"; do # "valid" -- Skipped!
            python prep_mimic_gen_seq.py \
                --single-seq \
                --align-y-seq \
                --window-x ${hr} \
                --window-y ${hr} \
                --y-type 'multi_event' \
                --step-size ${hr} \
                --medts-file medTS_MV_${set}_instances_exclablab${test_str}_split_${split_id}_minsd_2.0_sv_mimicid.npy \
                --set-type ${set} \
                --excl-lab-abnormal \
                --split-id ${split_id} \
                --use-mimicid \
                ${test_opt} \
                --force-create \
                --data-save-path "data_path/mimic.sequence" \
                --base-path "data_path/mimic.events" \
                --elapsed-time \
                --opt-str "_minsd_2_maxsd_20_sv" &
        done
    done
done

cd -