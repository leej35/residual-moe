bash run_sci.bash 24 RETAIN 1 "\
    --fast-folds 1 \
    --hyper-weight-decay 1e-04 \
    --hyper-weight-decay 1e-05 \
    --hyper-weight-decay 1e-06 \
    --hyper-weight-decay 1e-07 \
    --hidden-dim 512 \
    --multiproc 5 \
    --hyper-num-layer 1 \
    --bptt 0 --eval-on-cpu"
