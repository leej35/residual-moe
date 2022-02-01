import logging
import sys
from argparse import ArgumentParser
logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y/%m/%d %I:%M:%S %p'
)

logger.setLevel(logging.DEBUG)


def get_parser():

    parser = ArgumentParser(
        description='To run RNN Timeseries Prediction Model')

    # data
    parser.add_argument('--data-name', dest='data_name',
                        default='None',
                        help='data name')
    parser.add_argument('--data-filter', dest='data_filter',
                        default='None')

    parser.add_argument('--window-hr-x', dest='window_hr_x', type=int, default=6,
                        help='window x size in hour')
    parser.add_argument('--window-hr-y', dest='window_hr_y', type=int, default=48,
                        help='window y size in hour')
    parser.add_argument('--load-model-from', dest='load_model_from',
                        help='Loads the model parameters from the given path')
    parser.add_argument('--num-workers', type=int, default=2, dest='num_workers',
                        help='Number of workers for data loader class')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128,
                        help='The batch size. Default to 128')
    parser.add_argument('--base-path', dest='base_path', type=str,
                        default='../../',
                        help='base path for the dataset')
    parser.add_argument('--testmode', action='store_true',
                        dest='testmode', default=False)
    parser.add_argument('--midnight', action='store_true',
                        dest='midnight', default=False)
    parser.add_argument('--labrange', action='store_true',
                        dest='labrange', default=False)

    # attention mechanism
    parser.add_argument('--attn', action='store_true', default=False,
                        dest='attn', help='use attention mechanism')
    parser.add_argument('--attn-type', dest='attn_type', type=str,
                        default='dot', help='one of [dot, additive]')
    parser.add_argument('--attn-channel', dest='attn_channel', type=str,
                        default='single', help='one of [single, event]')

    # parser.add_argument('--attn-pe', action='store_true', default=False,
    #                     help='Positional Embedding')
    parser.add_argument('--attn-pe-c', action='store_true', default=False,
                        help='Positional Embedding, Concat version')
    # parser.add_argument('--attn-bpe', action='store_true', default=False,
    #                     help='Back Positional Embedding')
    parser.add_argument('--attn-pe', type=str, default='None',
                        help='Type of Positional Embedding. One of '
                             '[sinusoid, pos, reverse-pos, back]')

    parser.add_argument('--attn-ch-size', dest='attn_channel_size',
                        type=int, default=4)
    parser.add_argument('--do-svd', dest='do_svd', action='store_true',
                        default=False)
    parser.add_argument('--freeze-emb', dest='freeze_emb', action='store_true',
                        default=False)

    parser.add_argument('--attn-direct', action='store_true', default=False,
                        help='Apply embedding directly over events')

    parser.add_argument('--attn-q-type', type=str, default='None',
                        help='Attention query type [None, recent]')
    parser.add_argument('--attn-group-type', type=str, default='event_type',
                        help='Attention group type [event_type, coocur]')
    parser.add_argument('--group-size-method', type=str, default='fixed',
                        help='Attn group size computing method [fixed, renorm0.5]')
    parser.add_argument('--attn-inner-dim', type=int,
                        default=None, help='inner dimension of attention net')

    parser.add_argument('--attn-temporal-embed', action='store_true', default=False,
                        dest='attn_temp_embed', help='Temporal Embedding.')
    parser.add_argument('--attn-co-channel',
                        action='store_true', default=False)
    parser.add_argument('--cluster-only', action='store_true', default=False)
    parser.add_argument('--cluster-method', type=str, default='discretize',
                        help='Clustering label assign type '
                             '[discretize, kmeans]')

    parser.add_argument('--pred-period', action='store_true', default=False,
                        help='Apply periodic event prediction')
    parser.add_argument('--pred-period-type', type=str, default='default',
                        help='Type of periodic event prediction '
                             '[default, adaptive]')
    parser.add_argument('--pred-period-attn', action='store_true', default=False,
                        help='Insert statistics of pp module into attention')
    parser.add_argument('--pp-weight-scheme', type=str, default='default',
                        help='Type of periodic alpha weighting scheme'
                             ' [default, feed_nn]')

    parser.add_argument('--layer-norm', action='store_true', default=False,
                        help='Apply layer normalization')

    # model parameters
    parser.add_argument('--memorize-time', action='store_true', default=False,
                        dest='memorize_time',
                        help='Time memory for timing prediction.')

    parser.add_argument('--baseline-type', dest='baseline_type', type=str,
                        default='None', help='one of [logistic_binary, '
                                             'logistic_count, random,'
                                             'majority_all, majority_ts]')

    parser.add_argument('--hidden-dim', dest='hidden_dim',
                        type=int, default=128,
                        help=('The size of hidden activation and '
                              'memory cell of LSTM. Default is 128'))

    parser.add_argument('--tf-d-word', dest='tf_d_word',
                        type=int, default=16,
                        help=('word embedding dim for transformer'))
    parser.add_argument('--tf-d-model', dest='tf_d_model',
                        type=int, default=16,
                        help=('model dim for transformer'))

    parser.add_argument('--tf-d-inner', dest='tf_d_inner',
                        type=int, default=16,
                        help=('inner dim for transformer'))

    parser.add_argument('--tf-d-k', dest='tf_d_k',
                        type=int, default=16,
                        help=('key dim for transformer'))

    parser.add_argument('--tf-d-v', dest='tf_d_v',
                        type=int, default=16,
                        help=('value dim for transformer'))

    parser.add_argument('--tf-n-heads', dest='num_heads', type=int, default=1,
                        help=('Number of heads in transformer model.'))

    parser.add_argument('--embedding-dim', dest='embedding_dim',
                        type=int, default=128,
                        help=('The size of word embedding '
                              'Default is 128'))
    parser.add_argument('--pooling-type', dest='pooling_type', type=str,
                        default='mean',
                        help='Type of pooling: (mean, max) ')

    parser.add_argument('--bidirectional', action='store_true',
                        dest='bidirectional',
                        default=False, help=('Bidirectionality of LSTM.'))
    parser.add_argument('--n-layers', dest='num_layers', type=int, default=1,
                        help=('Number of layers in RNN model.'))

    parser.add_argument('--rnn-type', dest='rnn_type', type=str,
                        default='NoRNN',
                        help='Type of RNN: (RNN, LSTM, GRU) ')
    parser.add_argument('--out-func', dest='out_func', type=str,
                        default='sigmoid',
                        help='Type of last activation function of a network: '
                             '(sigmoid, softmax) default: sigmoid ')
    parser.add_argument('--cuda', action='store_true', default=False,
                        dest='use_cuda', help='Use GPU for computation.')
    parser.add_argument('--not-batch-first', dest='batch_first',
                        action='store_false', default=True,
                        help='Batch first')
    parser.add_argument('--pred-time', dest='pred_time', action='store_true',
                        default=False)
    parser.add_argument('--do-rank', dest='do_rank', action='store_true',
                        default=False)
    parser.add_argument('--time-loss-scale', dest='time_loss_scale',
                        type=float, default=0.5,
                        help=('scaling time loss w.r.t. event loss'))
    parser.add_argument('--hier-pred', dest='hier_pred', action='store_true',
                        default=False,
                        help='cs3750: hierarchical RNN prediction '
                        'for event and timing')
    parser.add_argument('--recent-bias', dest='recent_bias', action='store_true',
                        default=False, help='recent bias ')
    parser.add_argument('--simple-gate', dest='simple_gate', action='store_true',
                        default=False)
    parser.add_argument('--use-orig-params', action='store_true',
                        default=False)
    parser.add_argument('--eval-on-cpu', action='store_true',
                        default=False)
    parser.add_argument('--init-decay', action='store_true',
                        default=False,
                        help='pos (rev) embedding initialized '
                             'with decayed scaling')
    parser.add_argument('--f-exp', action='store_true',
                        default=False,
                        help='to use exp based function formular for periodicity'
                        ' modules distance computation')

    parser.add_argument('--dummy', action='store_true', default=False)

    # optimization and training

    parser.add_argument('--learning-rate', dest='learning_rate', type=float,
                        default=0.005,
                        help='The learning rate for the model.')
    parser.add_argument('--epoch', dest='epoch', type=int, default=20,
                        help='The number of epoch to train on.')
    parser.add_argument('--max-doc-len', dest='max_doc_len', type=int, default=0,
                        help='Limit the maximum document length')
    parser.add_argument('--bptt', type=int, default=False)
    parser.add_argument('--patient-stop', type=int, default=10,
                        help='Patient stopping criteria')
    parser.add_argument('--use-bce-logit', action='store_true',
                        dest='use_bce_logit', default=False,
                        help='Use BCEWithLogit loss function.')
    parser.add_argument('--use-bce-stable', action='store_true',
                        dest='use_bce_stable', default=False,
                        help='Add eps (1e-12) to pred on bce loss.')
    parser.add_argument('--optimizer', type=str, default="adam")

    # print, load and save options
    parser.add_argument('--validate-every', dest='valid_every', type=int,
                        metavar='n', default=10,
                        help='Validate on validation data every n epochs')
    parser.add_argument('--print-every', type=int, default=5,)
    parser.add_argument('--save-every', type=int, default=-1,
                        help=('Save the model every this number of epochs. '
                              'Default to be the same as --validate-every. '
                              'If set to 0, will save only the final model.'))
    parser.add_argument('--model-prefix', type=str, default="logs/_tmp_",
                        help='Binarized model file will have '
                             'this prefix on its name.')

    # curriculum learning
    parser.add_argument('--curriculum', action='store_true',
                        dest='curriculum_learning', default=False,
                        help='Do curriculum learning.')
    parser.add_argument('--curriculum-rate', type=float,
                        dest='curriculum_rate', default=1.0005,
                        help='The rate for curriculum learning')
    parser.add_argument('--curriculum-init', type=int,
                        dest='curriculum_init', default=1,
                        help='The initial curriculum max seq length')

    # learning rate scheduler
    parser.add_argument('--lr-scheduler', dest='lr_scheduler',
                        action='store_true', default=False,
                        help='Learning Rate Opitmization Scheduler')
    parser.add_argument('--lr-scheduler-multistep',
                        dest='lr_scheduler_multistep', action='store_true',
                        default=False,
                        help='Use multi-step learning rate scheduler. '
                             'Specify the milestones using '
                             '--lr-scheduler-epochs.')
    parser.add_argument('--lr-scheduler-ror',
                        dest='lr_scheduler_ror', action='store_true',
                        default=False,
                        help='Use ReduceLROnPlateau learning rate scheduler. ')

    parser.add_argument('--lr-scheduler-epochs', dest='lr_scheduler_epochs',
                        type=int, nargs='+',
                        default=[45, 90, 135, 180, 225],
                        help='The epochs in which to reduce the learning rate.')
    parser.add_argument('--lr-scheduler-numiter', dest='lr_scheduler_numiter',
                        type=int, default=15,
                        help='Number of epochs before reducing learning rate.')
    parser.add_argument('--lr-scheduler-mult', dest='lr_scheduler_mult',
                        type=float, default=0.5,
                        help='The multiplier used to reduce the learning rate.')
    parser.add_argument('--gpu-id', type=int, dest='gpu_id', default=0)

    parser.add_argument('--aime-eval', action='store_true', default=False,
                        help='for evaluation code, run like AIME-2019 paper')
    parser.add_argument('--aime-eval-macro', dest='aime_eval_macro',
                        action='store_true', default=False,
                        help='macro auprc, auroc')
    parser.add_argument('--jit', action='store_true', default=False,
                        help='for pyro HMM implementation, use jit mode for '
                             'faster computation')

    parser.add_argument('--skip-hypertuning', action='store_true', default=False,
                        dest='skip_hypertuning',
                        help='skip hypertuning for Split10 setting')
    parser.add_argument('--not-eval-multithread', action='store_false', default=True,
                        dest='eval_multithread',
                        help='Multithread eval')

    # split10 & cross validation
    parser.add_argument('--num-folds', type=int, dest='num_folds', default=5)
    parser.add_argument('--split-id', type=int, dest='split_id', default=1)
    parser.add_argument('--model-name', type=str, dest='model_name')
    parser.add_argument('--code-name', type=str, dest='code_name', default='')
    parser.add_argument('--remapped-data', action='store_true', default=False,
                        dest='remapped_data')

    # regularizations (https://github.com/salesforce/awd-lstm-lm)
    parser.add_argument('--weight-decay', type=float, dest='weight_decay',
                        default=0,
                        help=('L2 Regularization coefficient'))
    parser.add_argument('--dropout', type=float, default=0,
                        help=('Dropout rate (0:none)'))
    parser.add_argument('--dropouth', type=float, default=0,
                        help=('Dropout rate (0:none) for RNN hidden states'))
    parser.add_argument('--dropouti', type=float, default=0,
                        help=('Dropout rate (0:none) for input embedding'))
    parser.add_argument('--wdrop', type=float, default=0,
                        help=('weight dropout to apply to the RNN hidden to hidden matrix'))
    parser.add_argument('--tie-weights', action='store_true', default=False,
                        help='share input embedding & output embedding')

    parser.add_argument('--single-run-cv', action='store_true', default=False,
                        help='Run cross validation with only one run')
    parser.add_argument('--force-epoch', action='store_true', default=False,
                        help='Turn off the ')
    parser.add_argument('--skip-hidden-state', action='store_true', default=False,
                        dest='skip_hidden_state')

    parser.add_argument('--use-mimicid', dest='use_mimicid', action="store_true",
                        default=False,
                        help='instead of mapped vec_index, use mimic itemid with'
                        'value string (abnormal/normal/etc.)')
    parser.add_argument('--opt-str', dest='opt_str', type=str, default="none",
                        help='optional string attached to output path name')
    parser.add_argument('--use-valid', dest='use_valid', action="store_true",
                        default=False)

    parser.add_argument('--clamp-prob', dest='clamp_prob', type=float, default=None,
                        help=('clamp output probability with this minimum bound'))

    # debug LR-last (oct-02-2019)
    parser.add_argument('--target-event', type=int, default=-1,
                        help=('only use this target id for predict & evaluation'))
    parser.add_argument('--force-checkpoint', dest='force_checkpointing', action="store_true",
                        default=False)
    parser.add_argument('--force-auroc', dest='force_auroc', action="store_true",
                        default=False)
    parser.add_argument('--force-comet', dest='force_comet', action="store_true",
                        default=False)
    parser.add_argument('--force-plot-auroc', dest='force_plot_auroc', action="store_true",
                        default=False)

    parser.add_argument('--multiproc', type=int, default=1,
                        help='multiprocessing (number of cores)')

    parser.add_argument('--weight-change', dest='weight_change', action="store_true",
                        default=False)

    parser.add_argument('--loss-tol', type=float,
                        default=1, help='loss tolerance')

    parser.add_argument('--weight-decay-range', nargs='+', type=float,
                        help='weight decay range for hyperparam tuning')

    parser.add_argument('--excl-ablab', dest='excl_ablab', action="store_true",
                        default=False)

    parser.add_argument('--excl-abchart', dest='excl_abchart', action="store_true",
                        default=False)
    parser.add_argument('--hyper-weight-decay', dest='hyper_weight_decay',
                        action='append', default=None)

    parser.add_argument('--hyper-bptt', dest='hyper_bptt',
                        action='append', default=None)
    parser.add_argument('--hyper-num-layer', dest='hyper_num_layer',
                        action='append', default=None)

    parser.add_argument('--hyper-hidden-dim', dest='hyper_hidden_dim',
                        action='append', default=None)
    parser.add_argument('--hyper-batch-size', dest='hyper_batch_size',
                        action='append', default=None)
    parser.add_argument('--hyper-learning-rate', dest='hyper_learning_rate',
                        action='append', default=None)
    parser.add_argument('--rb-init', dest='rb_init', type=str, default="None",
                        help="xavier-prior-asweight or asbias")
    parser.add_argument('--manual-alpha', type=float, default=-1,
                        help='manual alpha (importance weight) for periodicity module')

    parser.add_argument('--eval-only', dest='eval_only', action="store_true",
                        default=False, help="run evaluation only (skip training)")

    parser.add_argument('--freeze-loaded-model', dest='freeze_loaded_model', action="store_true",
                        default=False, help="freeze loaded model for finetuning")

    parser.add_argument('--fast-folds', type=int, default=None,)

    parser.add_argument('--skip-train-eval', dest='skip_train_eval',
                        action="store_true", default=False,)

    parser.add_argument('--vbv', action="store_true", default=False)
    parser.add_argument('--target-auprc', action="store_true", default=False)
    parser.add_argument('--f-window', action="store_true", default=False)
    parser.add_argument('--pred-labs', action="store_true", default=False)
    parser.add_argument('--clock-gate', action="store_true", default=False)
    parser.add_argument('--lab-pp-proc', action="store_true", default=False)
    parser.add_argument('--elapsed-time', action="store_true", default=False)
    parser.add_argument('--pp-merge-signal',
                        dest='pp_merge_signal', type=str, default="")
    parser.add_argument('--simulated-data', action="store_true", default=False)
    parser.add_argument('--simulated-data-name', type=str,
                        default="None")
    parser.add_argument('--pp-concat', action="store_true", default=False)
    parser.add_argument('--rb-concat', action="store_true", default=False)
    parser.add_argument('--pp-ascounts', action="store_true", default=False)

    parser.add_argument('--prior-from-mimic',
                        action="store_true", default=False)

    parser.add_argument('--warmup', dest='n_warmup_steps', type=int,
                        default=1, help='number of warmup steps for Transformer')
    parser.add_argument('--testmode-by-onefold',
                        action="store_true", default=False)

    parser.add_argument('--x-as-list', action="store_true", default=False)
    parser.add_argument('--force-comet-off',
                        action="store_true", default=False)
    parser.add_argument('--use-pos-enc', action="store_true", default=False)

    parser.add_argument('--grad-accum-steps', default=1, type=int)

    # AIME 20 - multi-scale memory
    parser.add_argument('--past-mem', action="store_true", default=False)
    parser.add_argument('--past-dist', default=0, type=int,
                        )

    parser.add_argument('--past-dist-lt', default=0, type=int)
    parser.add_argument('--past-dist-st', default=0, type=int,
                        help="unit by hour. for a sequence, from beginning to current minus"
                             "this threshold will be part of distant memory.")
    parser.add_argument('--past-as-count', action="store_true", default=False)
    parser.add_argument('--pm-softmax', action="store_true", default=False)

    # Transformer
    parser.add_argument('--tf-by-step', action="store_true", default=False)
    parser.add_argument('--tf-pooling', action="store_true", default=False,
                        help="pooling events in a single time step before put in to self-attention"
                        )
    parser.add_argument('--tf-type', dest='tf_type', type=str, default="full",
                        help="transformer type, [full, thin]")
    parser.add_argument('--pred-future-steps', default=0, type=int)

    # Adaptive Prediction Module
    parser.add_argument('--adapt-lstm', dest='adapt_lstm',
                        action="store_true", default=False)
    parser.add_argument('--adapt-bandwidth', dest='adapt_bandwidth', type=int, default=3,
                        help="kernel bandwidth size.")
    parser.add_argument('--adapt-loss', dest='adapt_loss',
                        type=str, default="bce", help="[bce, mse]")
    parser.add_argument('--adapt-lr', dest='adapt_lr',
                        type=float, default=0.005)
    parser.add_argument('--adapt-pop-based', dest='adapt_pop_based',
                        action="store_true", default=False)
    parser.add_argument('--adapt-residual', dest='adapt_residual',
                        action="store_true", default=False)
    parser.add_argument('--adapt-residual-wdecay',
                        dest='adapt_residual_wdecay', type=float, default=1e-06)
    parser.add_argument('--adapt-switch', dest='adapt_switch',
                        action="store_true", default=False)
    parser.add_argument('--adapt-lstm-only', dest='adapt_lstm_only',
                        action="store_true", default=False)
    parser.add_argument('--adapt-fc-only', dest='adapt_fc_only',
                        action="store_true", default=False)
    parser.add_argument('--adapt-sw-pop', dest='adapt_sw_pop', action="store_true", default=False,
                        help="when start new sequence, compare previous step's model and population "
                        "based on performance and choose one that gives lower error.")

    parser.add_argument('--pred-normal-labchart',
                        dest='pred_normal_labchart', action="store_true", default=False)
    parser.add_argument('--verbose', dest='verbose',
                        action="store_true", default=False)

    parser.add_argument('--ptn-event-cnt', dest='ptn_event_cnt',
                        action="store_true", default=False)

    parser.add_argument('--event-weight-loss',
                        dest='event_weight_loss', type=str, default=None)
    parser.add_argument('--event-weight-loss-importance',
                        dest='event_weight_loss_importance', type=float, default=2)

    # Adaptive Memory Prediction Module
    parser.add_argument('--neural-caching', dest='neural_caching',
                        action="store_true", default=False)

    parser.add_argument('--ncache-window',
                        dest='ncache_window', type=int, default=10)
    parser.add_argument('--ncache-theta', dest='ncache_theta',
                        type=float, default=0.6625523432485668)
    parser.add_argument('--ncache-lambdah', dest='ncache_lambdah',
                        type=float, default=0.12785920428335693)

    parser.add_argument('--hyper-ncache-window', dest='hyper_ncache_window',
                        action='append', default=None)
    parser.add_argument('--hyper-ncache-theta', dest='hyper_ncache_theta',
                        action='append', default=None)
    parser.add_argument('--hyper-ncache-lambdah', dest='hyper_ncache_lambdah',
                        action='append', default=None)
    parser.add_argument('--early_terminate_inference',
                        dest='early_terminate_inference', action="store_true", default=False)

    parser.add_argument('--adapt-mem', dest='adapt_mem',
                        action="store_true", default=False)
    parser.add_argument('--mem-size',
                        dest='mem_size', type=int, default=1000)
    parser.add_argument('--use-mem-gpu', dest='use_mem_gpu',
                        action="store_true", default=False)
    parser.add_argument('--read-mode',
                        dest='read_mode', type=str, default='nn1',
                        help="[nn1 or softmax]")
    parser.add_argument('--mem-policy', dest='mem_policy',
                        type=str, default='age',
                        help="[age or error]")

    parser.add_argument('--mem-merge', dest='mem_merge',
                        type=str, default='lambdah',
                        help="[lambdah, concat, add, mem_only, \
                             gating_hidden_and_mem, attention, gating_hidden, \
                              mem_only_gating]")
    parser.add_argument('--hyper-mem-read-threshold', dest='hyper_mem_read_error_threshold',
                        action='append', default=None)
    parser.add_argument('--mem-read-error-threshold', dest='mem_read_error_threshold',
                        type=float, default=0.1)
    parser.add_argument('--mem-read-similarity-threshold', dest='mem_read_similarity_threshold',
                        type=float, default=None)
    parser.add_argument('--mem-key', dest='mem_key',
                        type=str, default='hidden',
                        help="[hidden or input]")

    parser.add_argument('--mem-content', dest='mem_content',
                        type=str, default='target',
                        help="[target or prob_adjust]")
    parser.add_argument('--log-mem-file', dest='log_mem_file',
                        action="store_true", default=False)
    parser.add_argument('--log-hidden-target-error-file',
                        dest='log_hidden_target_error_file', action="store_true", default=False)

    # train another model that predicts LSTM's performance per event type
    parser.add_argument('--train-error-pred', dest='train_error_pred',
                        action="store_true", default=False)
    parser.add_argument('--pred_error_train_stop_gap',
                        type=float, default=1e-10)
    parser.add_argument('--pred_error_train_min_epoch',
                        type=int, default=4)
    parser.add_argument('--pred_error_learning_rate',
                        type=float, default=1e-03)
    parser.add_argument('--pred_error_hidden_dim',
                        type=int, default=1024)
    parser.add_argument('--pred_error_act_func',
                        type=str, default="gelu")
    parser.add_argument('--mem_is_unbounded',
                        action="store_true", default=False)
    parser.add_argument('--train_learn_to_use_mem',
                        action="store_true", default=False)
    parser.add_argument('--ltam_num_layer',
                        type=int, default=4)
    parser.add_argument('--ltam_dropout',
                        type=float, default=0.2)
    parser.add_argument('--ltam_icv',
                        action="store_true", default=False)
    parser.add_argument('--use_context_target_stats',
                        action="store_true", default=False)
    parser.add_argument('--nn_mlp',
                        action="store_true", default=False)
    parser.add_argument('--gaussian_width',
                        type=float, default=0.2)
    parser.add_argument('--use_mapped_key',
                        action="store_true", default=False)
    parser.add_argument('--shrink_event_dim',
                        action="store_true", default=False)

    parser.add_argument('--fp16',
                        action="store_true", default=False)
    parser.add_argument('--fp16_opt_level',
                        type=str, default='O0')
    parser.add_argument('--max_grad_norm',
                        type=float, default=1.0)
    parser.add_argument('--tf_use_torch',
                        action="store_true", default=False)

    # self-correct LSTM
    parser.add_argument('--self_correct',
                    action="store_true", default=False)

    parser.add_argument('--correct_mode',
                        type=str, default='train_predictor',
                        help="""one of [train_predictor, train_corrector], 
                                default: train_corrector""")
    parser.add_argument('--corrector_feed_input',
                        action="store_true", default=False)
    parser.add_argument('--corrector_feed_input_and_hidden',
                        action="store_true", default=False)
    parser.add_argument('--correct_loss_type',
                        type=str, default='bce', help="[bce, mse]")
    parser.add_argument('--corrector_f_act',
                        type=str, default='tanh', 
                        help='[tanh, relu, leakyrelu, none]')
    parser.add_argument('--corrector_init_bias_zero',
                        action="store_true", default=False)
    parser.add_argument('--corrector_init_weight_small',
                        action="store_true", default=False)
    parser.add_argument('--corrector_init_val',
                        type=float, default=1.0)
    parser.add_argument('--corrector_no_clamp',
                        action="store_true", default=False)
    parser.add_argument('--use_corrector_control',
                        action="store_true", default=False)
    parser.add_argument('--corrector_control_act',
                        type=str, default="sigmoid", 
                        help="[sigmoid, hard]")
    parser.add_argument('--corrector_control_inp',
                        type=str, default="hidden", 
                        help="[hidden_and_input, input, hidden]")
    parser.add_argument('--corrector_control_arch',
                        type=str, default="mlp")

    # Domain Adaptation
    parser.add_argument('--da',
                        action="store_true", default=False,
                        help="domain adaptation")
    parser.add_argument('--da_percentile',
                        type=float, default=0.90,
                        help="instances above this percentile are of "
                              "target domain (loss: lower the better)")
    parser.add_argument('--da_lambda',
                        type=float, default=0.1,
                        help="weight for adversarial loss term")
    parser.add_argument('--da_input',
                        type=str, default='hidden',
                        help="what will be input for domain pred layer"
                             "[hidden or proj]")
    parser.add_argument('--da_pooling',
                        type=str, default='sum', help="[sum, mean, max]")

    parser.add_argument('--hyper_da_pooling', dest='hyper_da_pooling',
                        action='append', default=None)
    parser.add_argument('--hyper_da_lambda', dest='hyper_da_lambda',
                        action='append', default=None)
    parser.add_argument('--hyper_da_input', dest='hyper_da_input',
                        action='append', default=None)
    parser.add_argument('--da_change_every',
                        type=int, default=1, help="change domain split every")

    # GRU-adaption-clustering (Subgroup)
    parser.add_argument('--subgroup_adaptation',
                        action="store_true", default=False,
                        help="adapt GRU with clustering")
    parser.add_argument('--sg_skip_train_basemodel',
                        action="store_true", default=False,)
    parser.add_argument('--num_clusters',
                        type=int, default=8, help="num of subgroups")
    parser.add_argument('--num_svd_components',
                        type=int, default=64, help="num of svd output dim")
    parser.add_argument('--decompose',
                        type=str, default='svd', help="[pca, svd]")
    parser.add_argument('--way_output',
                        type=str, default='W0')
    parser.add_argument('--sg_init_gru_from_base',
                        action="store_true", default=False,)
    parser.add_argument('--do_plot',
                        action="store_true", default=False,)
    parser.add_argument('--plot_tsne',
                        action="store_true", default=False,)
    parser.add_argument('--threshold_errors',
                        action="store_true", default=False,)
    parser.add_argument('--threshold_errors_value',
                        type=float, default=0.1,)
    parser.add_argument('--seq_err_pooling',
                        type=str, default='mean', help="[mean, sum, max]")
    parser.add_argument('--bypass_svd',
                        action="store_true", default=False,)
    parser.add_argument('--kmeans_max_iter',
                        type=int, default=100,)
    parser.add_argument('--kmeans_tol',
                        type=float, default=0.00001,)
    parser.add_argument('--sg_sub_hidden_dim',
                        type=int, default=512,)
    parser.add_argument('--sg_add_base_hidden',
                        action="store_true", default=False,)
    parser.add_argument('--sg_concat_base_hidden',
                        action="store_true", default=False,)
    parser.add_argument('--sg_loss_type',
                        type=str, default='bce', help="[bce, mse, l1]")
    parser.add_argument('--sg_adapt_on_error',
                        action="store_true", default=False,)
    parser.add_argument('--sg_topk',
                        type=int, default=0,)

    # Mixture of Experts (MoE)
    parser.add_argument('--moe',
                        action="store_true", default=False,)
    parser.add_argument('--moe_num_experts',
                        type=int, default=10,)
    parser.add_argument('--moe_topk',
                        type=int, default=10,)
    parser.add_argument('--moe_noisy_gating',
                        action="store_true", default=False,)
    parser.add_argument('--moe_gate_type',
                        type=str, default='mlp',)
    parser.add_argument('--moe_residual',
                        action="store_true", default=False,)
    parser.add_argument('--moe_hidden_dim',
                        type=int, default=512,)
    parser.add_argument('--moe_skip_train_basemodel',
                        action="store_true", default=False,)
    parser.add_argument('--moe_zero_expert',
                        action="store_true", default=False,)
    parser.add_argument('--moe_load_gru_model_from',
                        type=str, default=None,)
    parser.add_argument('--moe_feed_error',
                        action="store_true", default=False,)
    parser.add_argument('--moe_incl_base_to_expert',
                        action="store_true", default=False,)



    args = parser.parse_args()

    return args


def print_args(args):
    for arg in sorted(vars(args)):  # print all args
        itm = str(getattr(args, arg))
        if (itm != 'None' and itm != 'False'
                and arg != 'vecidx2label' and arg != 'event_dic'):
            logger.info('{0: <20}: {1}'.format(arg, itm))  #
