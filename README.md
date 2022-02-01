# Learning to Adapt Dynamic Clinical Event Sequences with Residual Mixture of Experts

This repository is the official implementation of Learning to Adapt Dynamic Clinical Event Sequences with Residual Mixture of Experts. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data Preparation
This paper experiments with MIMIC-3 Clinical Database, provided by physionet.org. 
As they prohibit re-distribution of the MIMIC-3 database, we cannot provide preprocessed dataset we use for our experiment.

Instead, we prepare the data extraction, event-time-series generation, and featurization code in this repository.

**Prerequsite**
(1) Obtain MIMIC-3 access at https://mimic.physionet.org/gettingstarted/access/
(2) Obtain the database (csv files) and install them at your local database (e.g., MySQL)
(3) Setting database access info (account, password) at `prep_data/DBConnectInfo.py`.

**Training/Test Data Generation**
(1) Set proper `data path` according to your local computer setting in following files:
* `prep_data/prep_mimic_extract_ts_adhoc.py`
* `prep_data/scripts/exec_prep_mimic_gen_seq_lab_range_split10_mimicid.sh`
* `prep_data/scripts/exec_prep_mimic_remap_itemid_split10.bash`

(2) Run scripts according to `prep_data/scripts/bootstrap.bash`. I recommend to run each step in the file one by one.


## Training and Evaluation

We recommend to train `GRU` model first. Then, provide the path of the trained model of the GRU at `--load-model-from` argument.

When one experiment is run, it first do training and then it automatically run evaluation.

In order to run evaluation only, provide this argument: `--eval-only --load-model-from [pretrained model]`.

To train and get the evaluation the models in the paper, run following commands in `scripts` folder:

### (Base) GRU
```
hidden_dim=512
bash run_sci.bash 24 GRU 1 "--fast-folds 1 --hyper-weight-decay 1e-04 --hyper-weight-decay 1e-05 --hyper-weight-decay 1e-06 --hyper-weight-decay 1e-07 --hyper-hidden-dim ${hidden_dim} --multiproc 10 --bptt 0 --eval-on-cpu"
```   

### RETAIN
```
bash run_sci.bash 24 RETAIN 1 "--fast-folds 1 --hyper-weight-decay 1e-04 --hyper-weight-decay 1e-05 --hyper-weight-decay 1e-06 --hyper-weight-decay 1e-07 --hidden-dim 512 --multiproc 5 --hyper-num-layer 1 --bptt 0 --eval-on-cpu"
```  

### CNN
```
bash run_sci_nips20.bash 24 CNN 1 "--fast-folds 1 --hyper-weight-decay 1e-04 --hyper-weight-decay 1e-05 --hyper-weight-decay 1e-06 --hyper-weight-decay 1e-07 --hidden-dim 512 --multiproc 5 --hyper-num-layer 1 --bptt 0 --eval-on-cpu"
```  

### LOGISTIC REGRESSION 
```
bash run_sci.bash 24 logistic_binary 1 "--fast-folds 1 --hyper-weight-decay 1e-04 --hyper-weight-decay 1e-05 --hyper-weight-decay 1e-06 --hyper-weight-decay 1e-07 --hyper-weight-decay 1e-07 --hidden-dim 512 --multiproc 5 --bptt 0 --eval-on-cpu"

``` 

### Residual-Mixture-of-Experts (R-MoE) 
```
moe_num_experts=50
hidden_dim=512
moe_hidden_dim=8
multiproc=5
bash run_residual_moe.bash 24 GRU 1 "--fast-folds 1 --hyper-weight-decay 1.5 --hyper-weight-decay 1.25 --hyper-weight-decay 1.0 --hyper-weight-decay 0.75 --hyper-weight-decay 0.5 --hidden-dim ${hidden_dim} --validate-every 2 --multiproc ${multiproc} --bptt 0 --eval-on-cpu --moe --moe_residual --moe_hidden_dim ${moe_hidden_dim} --moe_gate_type gru --moe_skip_train_basemodel --moe_zero_expert --moe_load_gru_model_from pretrained_models/GRU_final_model.model --moe_num_experts ${moe_num_experts} --learning-rate 0.0005"
``` 

## Trained (Output) Models

The trained models that used to generate experiment reports are located under `pretrained_models` folder.
