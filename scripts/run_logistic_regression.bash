bash run_sci.bash 24 logistic_binary 1 "\
    --fast-folds 1 \
    --hyper-weight-decay 1e-04 \
    --hyper-weight-decay 1e-05 \
    --hyper-weight-decay 1e-06 \
    --hyper-weight-decay 1e-07 \
    --hyper-weight-decay 1e-07 \
    --hidden-dim 512 \
    --multiproc 5 \
    --bptt 0 --eval-on-cpu"

