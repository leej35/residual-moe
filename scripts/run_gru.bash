hidden_dim=512
bash run_sci.bash 24 GRU 1 "\
    --fast-folds 1 \
    --hyper-weight-decay 1e-04 \
    --hyper-weight-decay 1e-05 \
    --hyper-weight-decay 1e-06 \
    --hyper-weight-decay 1e-07 \
    --hyper-hidden-dim ${hidden_dim} \
    --multiproc 10 \
    --bptt 0 --eval-on-cpu "
