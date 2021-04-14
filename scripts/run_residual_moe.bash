#!/usr/bin/env bash
cd ../
python_path="/PATH_TO_PYTHON/anaconda3/envs/torch17/bin/python"
base_path="/PATH_TO_DATASET/ts_dataset/"

mkdir -p "logs"
mkdir -p "trained"

window_hr="$1"
model_type="$2"
model_name=${model_type}
split_id="$3"

opt_trg_auprc="--target-auprc"

randid=$(cat /dev/urandom | tr -dc 'A-Z0-9' | fold -w 6 | head -n 1)

num_folds="5"
epoch="5000"
valid_every="5"
patient_stop="5"
filename="${model_name}_w${window_hr}_s${split_id}"
curtime=$(date +"%Y_%m_%d_%H_%M_%S")
outdir="trained/fml_${curtime}_${randid}_${model_name}_w${window_hr}_s${split_id}"

source scripts/model_opts.sh


mkdir -p ${outdir}
CUDA_VISIBLE_DEVICES=0 "${python_path}" main_residual_moe.py \
    --epoch "${epoch}" \
    --validate-every "${valid_every}" \
    --save-every -1 \
    --model-prefix "${outdir}/${filename}" \
    --num-workers 0 \
    --print-every 1 \
    --n-layers 1 \
    --patient-stop "${patient_stop}" \
    --data-name "mimic3" \
    --base-path "${base_path}"\
    --window-hr-x "${window_hr}" \
    --window-hr-y "${window_hr}" \
    ${model_opt} \
    --model-name "${model_name}" \
    --split-id "${split_id}" \
    --num-folds "${num_folds}" \
    --remapped-data \
    --use-mimicid \
    --cuda \
    --labrange \
    --pred-normal-labchart \
    --opt-str "_minsd_2_maxsd_20_sv" \
    --force-auroc \
    --prior-from-mimic \
    --code-name "m2x20_predall_v5_hyper" \
    --hidden-dim 128 \
    --embedding-dim 64 \
    --learning-rate 0.005 \
    --batch-size 128 \
    --elapsed-time \
    --force-epoch \
    ${opt_trg_auprc} \
    $4 \
    | tee "logs/log_${filename}.txt" "${outdir}/log_${filename}.txt" --ignore-interrupts


cd -

