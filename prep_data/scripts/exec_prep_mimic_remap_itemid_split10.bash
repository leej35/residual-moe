#!/usr/bin/env bash

cd ../

if [[ $2 = "testmode" ]]; then
    test_str="_TEST"
    test_opt="--testmode"
else
    test_str=""
    test_opt=""
fi

for split_id in "1"; do #   
    python prep_mimic_remap_itemid_split10.py \
        --data-type "mimic" \
        --excl-lab-abnormal \
        --single-seq \
        --align-y-seq \
        --y-type "multi_event" \
        --base-path "data_path/mimic.sequence/" \
        --step-size "$1" \
        --window-hr-x "$1" \
        --window-hr-y "$1" \
        --split-id "${split_id}" \
        --itemdic "data_path/mimic.events/data/itemdic_exclablab_maxsd_20.0_minsd_2.0${test_str}.npy" \
        --opt-str "_minsd_2_maxsd_20_sv"\
        --elapsed-time \
        ${test_opt} \
        --use-mimicid &
done

cd -