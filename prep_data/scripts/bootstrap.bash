
# step1. extract TS
bash exec_prep_mimic_extract_ts_split10_mimic_item_id.sh


# step2. generate sequence
bash exec_prep_mimic_gen_seq_lab_range_split10_mimicid.sh 24;


# step3. remap mimic itemid to vec idx (and remove non-overlap items)
bash exec_prep_mimic_remap_itemid_split10.bash 24;


# step4. create internal cross validation
bash exec_prep_mimic_create_folds_split10.bash 24;
