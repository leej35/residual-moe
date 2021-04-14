import os
import copy
import itertools
import collections
import csv
import logging
import sys
import pickle

import numpy as np
from utils.project_utils import Timing
from utils.args import print_args

from models.base_seq_model import masked_bce_loss
from trainer import Trainer, load_simulated_data, load_multitarget_data, load_multitarget_dic
from pathos.multiprocessing import ProcessingPool as Pool

from main_utils import (create_model, get_weblog, load_model)
import traceback
import time
from AFSLogin import API_KEY

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)


def get_hyperparam_setttings(args):

    hyperparam_settings = []

    HyperArgs = collections.namedtuple('HyperArgs',
        'batch_size learning_rate embedding_dim '
        'hidden_dim bptt dropout dropout_emb '
        'wdrop weight_decay num_layers '
        'ncache_theta ncache_lambdah ncache_window '
        'mem_read_error_threshold da_pooling da_input da_lambda '
    )

    if args.baseline_type.startswith('hmm'):
        embedding_dims = [0]
        dropouts = [0]
    else:
        # dropout on hidden states as output of LSTM/RNN
        if args.rnn_type != 'NoRNN':
            dropouts = [0] # , 0.5
            embedding_dims = [args.embedding_dim]
        else:
            dropouts = [0]
            embedding_dims = [0]

    # dropout on word embedding output
    dropout_embs = [0]
    # dropout on hidden-to-hidden parameter of LSTM/RNN
    wdrops = [0]

    if args.hyper_batch_size is not None:
        batch_sizes = [int(x) for x in args.hyper_batch_size]
        logger.info("batch_sizes: {}".format(batch_sizes))
    else:
        batch_sizes = [args.batch_size]

    if args.hyper_learning_rate is not None:
        learning_rates = [float(x) for x in args.hyper_learning_rate]
        logger.info("learning_rates: {}".format(learning_rates))
    else:
        learning_rates = [args.learning_rate]

    if args.hyper_num_layer is not None:
        num_layers_list = [int(x) for x in args.hyper_num_layer]
        logger.info("num_layers_list: {}".format(num_layers_list))
    else:
        num_layers_list = [args.num_layers]

    if args.hyper_weight_decay is not None:
        weight_decays = [float(x) for x in args.hyper_weight_decay]
        logger.info("weight_decays: {}".format(weight_decays))
    else:
        weight_decays = [1e-06, 1e-07, 1e-08]

    if args.hyper_bptt is not None:
        bptts = [int(x) for x in args.hyper_bptt]
        logger.info("bptts: {}".format(bptts))
    else:
        bptts = [args.bptt]

    if args.hyper_ncache_theta is not None:
        ncache_thetas = [float(x) for x in args.hyper_ncache_theta]
        logger.info("ncache_thetas: {}".format(ncache_thetas))
    else:
        ncache_thetas = [args.ncache_theta]

    if args.hyper_ncache_window is not None:
        ncache_windows = [int(x) for x in args.hyper_ncache_window]
        logger.info("ncache_windows: {}".format(ncache_windows))
    else:
        ncache_windows = [args.ncache_window]

    if args.hyper_ncache_lambdah is not None:
        ncache_lambdahs = [float(x) for x in args.hyper_ncache_lambdah]
        logger.info("ncache_lambdahs: {}".format(ncache_lambdahs))
    else:
        ncache_lambdahs = [args.ncache_lambdah]

    if args.hyper_mem_read_error_threshold is not None:
        mem_read_error_thresholds = [float(x) for x in args.hyper_mem_read_error_threshold]
        logger.info("mem_read_error_thresholds: {}".format(ncache_lambdahs))
    else:
        mem_read_error_thresholds = [args.mem_read_error_threshold]

    if (args.baseline_type.startswith('logistic_')
            or args.baseline_type in ['copylast', 'majority_ts',
                                      'timing_random', 'random']):
        hidden_dims = [0]
    else:
        if args.hyper_hidden_dim is not None:
            hidden_dims = [int(x) for x in args.hyper_hidden_dim]
            logger.info("hidden_dims: {}".format(hidden_dims))
        else:
            hidden_dims = [args.hidden_dim]  # [64, 128, 256]

    if args.hyper_da_pooling is not None:
        da_poolings = [
            str(x) for x in args.hyper_da_pooling]
        logger.info("da_poolings: {}".format(da_poolings))
    else:
        da_poolings = [args.da_pooling]

    def process_hyper_args(name, arg_hyper, arg_var, arg_type):
        if arg_hyper is not None:
            return_list = [
                arg_type(x) for x in arg_hyper
            ]
            logger.info("{}: {}".format(name, return_list))
        else:
            return_list = [arg_var]
        return return_list

    # da_poolings = process_hyper_args(
    #     "da_poolings",
    #     args.hyper_da_pooling, 
    #     args.da_pooling,
    #     str,
    # )

    da_inputs = process_hyper_args(
        "da_inputs",
        args.hyper_da_input,
        args.da_input,
        str,
    )

    da_lambdas = process_hyper_args(
        "da_lambdas",
        args.hyper_da_lambda,
        args.da_lambda,
        float,
    )

    logger.info(f"hypertune - len dropout_embs : {len(batch_sizes)}")
    logger.info(f"hypertune - len learning_rates : {len(learning_rates)}")
    logger.info(f"hypertune - len embedding_dims : {len(embedding_dims)}")
    logger.info(f"hypertune - len hidden_dims : {len(hidden_dims)}")
    logger.info(f"hypertune - len bptts : {len(bptts)}")
    logger.info(f"hypertune - len dropouts : {len(dropouts)}")
    logger.info(f"hypertune - len dropout_embs : {len(dropout_embs)}")
    logger.info(f"hypertune - len wdrops : {len(wdrops)}")
    logger.info(f"hypertune - len num_layers_list : {len(num_layers_list)}")
    logger.info(f"hypertune - len weight_decays : {len(weight_decays)}")
    logger.info(f"hypertune - len ncache_lambdahs : {len(ncache_lambdahs)}")
    logger.info(f"hypertune - len ncache_windows : {len(ncache_windows)}")
    logger.info(f"hypertune - len ncache_thetas : {len(ncache_thetas)}")
    logger.info(f"hypertune - len mem_read_error_thresholds : {len(mem_read_error_thresholds)}")
    logger.info(f"hypertune - len da_lambdas : {len(da_lambdas)}")
    logger.info(f"hypertune - len da_inputs : {len(da_inputs)}")
    logger.info(f"hypertune - len da_poolings : {len(da_poolings)}")

    for batch_size in batch_sizes:
        for lr in learning_rates:
            for e_dim in embedding_dims:
                for h_dim in hidden_dims:
                    for bptt in bptts:
                        for dropout in dropouts:
                            for dropout_emb in dropout_embs:
                                for wdrop in wdrops:
                                    for num_layers in num_layers_list:
                                        for weight_decay in weight_decays:
                                            for ncache_theta in ncache_thetas:
                                                for ncache_lambdah in ncache_lambdahs:
                                                    for ncache_window in ncache_windows:
                                                        for mem_read_error_threshold in mem_read_error_thresholds:
                                                            for da_lambda in da_lambdas:
                                                                for da_input in da_inputs:
                                                                    for da_pooling in da_poolings:
                                                                        hyperparam_settings.append(
                                                                            HyperArgs(
                                                                                batch_size, lr, e_dim, h_dim,
                                                                                bptt, dropout, dropout_emb,
                                                                                wdrop,
                                                                                weight_decay,
                                                                                num_layers,
                                                                                ncache_theta, 
                                                                                ncache_lambdah, 
                                                                                ncache_window,
                                                                                mem_read_error_threshold,
                                                                                da_pooling,
                                                                                da_input,
                                                                                da_lambda
                                                                            )
                                                                        )
    
    logger.info(f"len(hyperparam_settings): {len(hyperparam_settings)}")

    return hyperparam_settings


def _runner(args):
    try:
        hyper_idx, package, fold_id, args, n_settings = args
        args = copy.deepcopy(args)
        hyper_args = package['hyper_args']
        args.model_prefix += '/h{}_f{}/'.format(hyper_idx, fold_id)
        os.system("mkdir -p {}".format(args.model_prefix))

        # merge original args and this hyper_args object together
        args = combine_namedtuple(args, hyper_args)
        if hasattr(args, 'web_logger'):
            args.web_logger.end()
            del args.web_logger
        web_logger = get_weblog(API_KEY, args)
        web_logger.log_other('model_name', args.model_name)
        web_logger.log_other('run_mode', 'hypertune')
        web_logger.log_other('split_id', args.split_id)
        web_logger.log_other('hyper_idx', hyper_idx)
        web_logger.log_other('fold_id', fold_id)
        for name, value in hyper_args._asdict().items():
            if name:
                web_logger.log_parameter(name, value)

        logger.info('\n{}\nhyper parameter tuning {}/{}  '
                    '\nsplit:{} '
                    '\nhyper parameter trying: {} '
                    '\ncv-fold:{}'
                    '\n{}'
                    ''.format('-' * 64, hyper_idx,
                            n_settings,
                            args.split_id, hyper_args, fold_id,
                            '-' * 64))

        print_args(args)

        with Timing('Loading data files...\n', logger=logger):
            train_dataset_cv, valid_dataset_cv, seqlen_idx, train_data_path,\
                target_size, inv_id_mapping \
                = get_dataset(args, args.split_id, args.event_size,
                            args.base_path, fold_id, args.num_folds, 
                            simulated_data=args.simulated_data)

        args.target_size = target_size
        args.inv_id_mapping = inv_id_mapping
        with Timing('Creating model and trainer...', logger=logger):
            model = create_model(args, args.event_size, args.device,
                                args.target_type, args.vecidx2label,
                                train_dataset_cv,
                                train_data_path,
                                web_logger=web_logger,
            )

            trainer = Trainer(
                model, args=args, d_path=train_data_path, web_logger=web_logger,
            )

        if args.load_model_from is not None:
            trainer = load_model(args, trainer, args.device, args.login,
                                args.renew_token)
        
        if not args.freeze_loaded_model:
            trainer.train_model(train_dataset_cv, valid_dataset_cv)
        
        if args.adapt_mem:
            trainer.infer_model(train_dataset_cv,
                                test_name='ltam_write',
                                final=True,
                                export_csv=False,
                                eval_multithread=args.eval_multithread,
                                return_metrics=True,
                                no_other_metrics_but_flat=True
                                )


        eval_stats = trainer.infer_model(valid_dataset_cv,
                                        test_name='hyper_valid',
                                        final=True,
                                        export_csv=False,
                                        eval_multithread=args.eval_multithread,
                                        return_metrics=True,
                                        no_other_metrics_but_flat=True
                                        )
        best_mac_auprc = eval_stats[-2]
        logger.info("best metric: {:.4f}".format(best_mac_auprc))
        package['stats'].append(best_mac_auprc)
        package['best_epochs'].append(trainer.best_epoch)
        del trainer
        del model
        del train_dataset_cv
        del valid_dataset_cv
        os.system('rm -rf {}epoch*.model'.format(args.model_prefix))
        web_logger.end()

    except Exception:
        logger.info("Exception in worker:")
        traceback.print_exc()
        raise

    return package


def hyperparam_tuning(args, hyperparam_settings):
    Box = collections.namedtuple('Box', 'stats param best_epochs')
    boxs = []

    homeworks = []
    for hyper_idx, hyper_args in enumerate(hyperparam_settings):

        package = {'hyper_idx': hyper_idx, 'stats': [],
                   'hyper_args': hyper_args, 'best_epochs': []}

        num_folds = args.num_folds if args.fast_folds is None else args.fast_folds

        for fold_id in range(num_folds):
            homeworks.append(
                (
                    hyper_idx, package, fold_id, args,
                    len(hyperparam_settings)
                )
            )


    # packages = [_runner(hw) for hw in homeworks] ## when debug
    # run workers asynchly

    # method : pathos 
    pool = Pool(nodes=args.multiproc)
    packages = pool.amap(_runner, homeworks)
    while not packages.ready():
        time.sleep(5)

    packages = packages.get()

    # method : py multiproc

    # with Pool(processes=args.multiproc) as pool:
    #     results = [pool.apply_async(_runner, args=(hw,)) for hw in homeworks]
    #     # packages = result.get()
    #     packages = [p.get() for p in results]

    logger.info('='*20)
    logger.info('hyper param tuning done!')
    logger.info('='*20)
    # packages = packages.get()

    # merge by same hyper_idxs (from different internal-cv sets)
    boxs = {}
    for pack in packages:
        hyper_idx = pack['hyper_idx']

        if hyper_idx not in boxs:
            boxs[hyper_idx] = {'stats': [], 'best_epochs': [],
                               'hyper_args': pack['hyper_args']}

        boxs[hyper_idx]['stats'] += pack['stats']
        boxs[hyper_idx]['best_epochs'] += pack['best_epochs']

    # get best hyperparam
    best_mean, best_std, best_param, best_epoch = 0, 0, None, 0

    with open(('{}hypertune_result.csv'.format(args.model_prefix)), 'w') as f_csv:
        writer = csv.writer(f_csv)
        fields = hyperparam_settings[0]._fields
        writer.writerow(list(fields) + ['avg', 'std', 'best_epoch'])
        for box in list(boxs.values()):
            writer.writerow(
                list(box['hyper_args']._asdict().values())
                + [
                    np.mean(box['stats']),
                    np.std(box['stats']),
                    int(np.mean(box['best_epochs']))
                ]
            )
            if np.mean(box['stats']) > best_mean:
                best_param = box['hyper_args']
                best_mean = np.mean(box['stats'])
                best_std = np.std(box['stats'])
                best_epoch = int(np.mean(box['best_epochs']))

    logger.info('\n{}'.format('=' * 64))
    logger.info('best hyperparam info \n param:{} \n mean:{} std: {}'
                 '\n best epoch: {}'.format(
                     best_param, best_mean, best_std, best_epoch
                 ))
    logger.info('\n{}'.format('-' * 64))

    return best_param, best_epoch



def combine_namedtuple(args, hyper_args):
    """
    Overwrite hyper_args into args
    """
    for name in hyper_args._fields:
        setattr(args, name, getattr(hyper_args, name))
    return args


def get_dict_and_info(args):
    target_type = 'multi'

    if args.data_name == 'mimic3':
        base_path = '{}/mimic_cs3750.sequence/'.format(
            args.base_path)
    else:
        base_path = args.base_path

    loss_fn = masked_bce_loss

    vecidx2label, event_size = load_multitarget_dic(base_path,
                                                    data_name=args.data_name,
                                                    data_filter=args.data_filter,
                                                    x_hr=args.window_hr_x,
                                                    y_hr=args.window_hr_y,
                                                    set_type=None,
                                                    midnight=args.midnight,
                                                    labrange=args.labrange,
                                                    excl_ablab=args.excl_ablab,
                                                    excl_abchart=args.excl_abchart,
                                                    test=args.testmode,
                                                    split_id=args.split_id,
                                                    remapped_data=args.remapped_data,
                                                    use_mimicid=args.use_mimicid,
                                                    option_str=args.opt_str,
                                                    elapsed_time=args.elapsed_time,
                                                    get_vec2mimic=args.prior_from_mimic,
                                                    )

    if args.prior_from_mimic:
        vecidx2label, vecidx2mimic = vecidx2label
        args.vecidx2mimic = vecidx2mimic
    else:
        args.vecidx2mimic = None
        
    args.event_size = event_size
    args.vecidx2label = vecidx2label
    args.event_dic = vecidx2label
    args.loss_fn = loss_fn
    args.target_type = target_type
    args.base_path = base_path
    return args


def get_merged_id_for_abnormal_normal(args):
    assert args.event_dic is not None
    assert not args.excl_ablab
    assert args.labrange
    # support for lab range, non-excl_ablab and non-excl_abchart
    # merge item-id index for abnormal and normal ids into normal ones.
    ab2n_mapping = {} # args.event_dic
    non_labchart_items = []
    for orig_idx, item in args.event_dic.items():
        if item['category'] in ['lab', 'chart']:

            label = item['label']
            norm_label = label.replace('-NORMAL', '').replace('-ABNORMAL_LOW', '').replace('-ABNORMAL_HIGH', '').replace('-ABNORMAL', '')

            if norm_label not in ab2n_mapping:
                ab2n_mapping[norm_label] = {'normal':None,'abnormals':[]}

            if label.endswith('-ABNORMAL_LOW') or label.endswith('-ABNORMAL_HIGH') or label.endswith('-ABNORMAL'):
                ab2n_mapping[norm_label]['abnormals'].append(orig_idx)

            else: # label.endswith('-NORMAL'):
                ab2n_mapping[norm_label]['normal'] = orig_idx

        else:
            non_labchart_items.append(orig_idx)
    
    # check where no-normal item exist (if happens, assign itself for normal)
    for norm_label, entry in ab2n_mapping.items():
        if entry['normal'] == None:
            logger.info('None: {}'.format(norm_label))
            logger.info('entry: {}'.format(entry))
            entry['normal'] = entry['abnormals'][0]

    ab2n_mapping = {v['normal']: v['abnormals']
                    for k, v in ab2n_mapping.items()}  # discard name
    
    inv_ab2n_mapping = {}
    for normal_item, abnormal_items in ab2n_mapping.items():
        for ab_item in abnormal_items:
            inv_ab2n_mapping[ab_item] = normal_item

    normal_items = list(ab2n_mapping.keys())
    target_items = normal_items + non_labchart_items
    non_target_items = itertools.chain.from_iterable(ab2n_mapping.values())

    logger.info('!! target_items: {}'.format(len(target_items)))
    logger.info('!! non_target_items: {}'.format(len(list(non_target_items))))

    # create id mapping first (orig_id -> new_id) for target items

    id_mapping = {}
    inv_id_mapping = {}
    for orig_idx in target_items:

        # supports excl lab only!! 
        item = args.event_dic[orig_idx]
        new_idx = len(id_mapping) + 1
        id_mapping[new_idx] = {'orig_idx': orig_idx, 
                            'category': item['category'],
                            'label': item['label']}    
        inv_id_mapping[orig_idx] = new_idx

    # then, add lab and chart abnormal items to inv_id_mapping
    for orig_idx in non_target_items:
        normal_item_orig_idx = inv_ab2n_mapping[orig_idx]
        new_idx = inv_id_mapping[normal_item_orig_idx]
        inv_id_mapping[orig_idx] = new_idx

    return id_mapping, inv_id_mapping, target_items


def test_get_merged_id_for_abnormal_normal():
    event_dic = {
        1:{'category':'lab','label':'A'}, 
        2:{'category':'lab','label':'A-ABNORMAL_LOW'},
        3:{'category':'lab','label':'A-ABNORMAL'},
        4:{'category':'lab','label':'A-ABNORMAL_HIGH'},
        5:{'category':'lab','label':'B'},
    }
    id_mapping, inv_id_mapping = get_merged_id_for_abnormal_normal(event_dic)
    logger.info('id_mapping:\n{}'.format(id_mapping))
    logger.info('inv_id_mapping:\n{}'.format(inv_id_mapping))


def get_lab_target_id_mapping(args):
    
    assert args.event_dic is not None
    assert args.excl_ablab 
    assert not args.labrange
    id_mapping = {}
    inv_id_mapping = {}
    for orig_idx, item in args.event_dic.items():
        if item['category'] == 'lab':
            
            # supports excl lab only!! 
            
            new_idx = len(id_mapping) + 1
            id_mapping[new_idx] = {'orig_idx': orig_idx, 
                                'category': item['category'],
                                'label': item['label']}    
            inv_id_mapping[orig_idx] = new_idx

    return id_mapping, inv_id_mapping



def get_dataset(args, split_id, event_size, base_path, fold_id=None,
                num_folds=None, count_mode=False,
                use_valid=False, simulated_data=False):
    logger.info('event_size: {}'.format(event_size))
    
    if simulated_data:
        fname_base = args.simulated_data_name

    if args.pred_labs or args.pred_normal_labchart:
        if args.pred_labs:
            args.event_dic, inv_id_mapping = get_lab_target_id_mapping(args)
            target_items = None
        elif args.pred_normal_labchart:
            args.event_dic, inv_id_mapping, target_items = get_merged_id_for_abnormal_normal(
                args)
        target_size = len(args.event_dic)
        np.save('{}_event_dic_target_id.npy'.format(args.model_prefix), args.event_dic)
        np.save('{}_inv_id_mapping_target_id.npy'.format(
            args.model_prefix), inv_id_mapping)
        logger.info('target_size: {}'.format(target_size))
    else:
        target_size, inv_id_mapping = event_size, None

    if fold_id is not None:
        valid_fold_id = [fold_id]
        train_fold_ids = list(range(num_folds))
        train_fold_ids.remove(fold_id)

        if args.testmode_by_onefold:
            # NOTE: 2020/01 for fast testmode run, only use first icv set.
            train_fold_ids = [train_fold_ids[0]]

    else:
        valid_fold_id = train_fold_ids = None

    seqlen_idx = 2

    if simulated_data:
        train_dataset, train_data_path = load_simulated_data(
            fname_base, set_type='train', split_id=split_id, 
            icv_fold_ids=train_fold_ids, icv_numfolds=5)
    else:
        train_dataset, train_data_path = \
            load_multitarget_data(args.data_name,
                                'train',
                                event_size,
                                data_filter=args.data_filter,
                                base_path=base_path,
                                x_hr=args.window_hr_x,
                                y_hr=args.window_hr_y,
                                test=args.testmode,
                                midnight=args.midnight,
                                labrange=args.labrange,
                                excl_ablab=args.excl_ablab,
                                excl_abchart=args.excl_abchart,
                                split_id=split_id,
                                icv_fold_ids=train_fold_ids,
                                icv_numfolds=num_folds,
                                remapped_data=args.remapped_data,
                                use_mimicid=args.use_mimicid,
                                option_str=args.opt_str,
                                pred_labs=args.pred_labs,
                                  pred_normal_labchart=args.pred_normal_labchart,
                                inv_id_mapping=inv_id_mapping,
                                target_size=target_size,
                                elapsed_time=args.elapsed_time,
                                x_as_list=args.x_as_list,
                                )

    tr_len = len(train_dataset[seqlen_idx])
    logger.info("# train seq: {}".format(tr_len))

    if simulated_data:

        valid_dataset, _ = load_simulated_data(
            fname_base, set_type='train', split_id=split_id,
            icv_fold_ids=valid_fold_id, icv_numfolds=5)

        va_len = len(valid_dataset[seqlen_idx])
        logger.info("# valid seq: {}".format(va_len))

    else:

        if use_valid:
            valid_type_str = 'valid'
        elif ((count_mode or fold_id is None)):
            valid_type_str = 'test'
        else:
            valid_type_str = 'train'

        valid_dataset, _ = \
            load_multitarget_data(args.data_name,
                                valid_type_str,
                                event_size,
                                data_filter=args.data_filter,
                                base_path=base_path,
                                x_hr=args.window_hr_x,
                                y_hr=args.window_hr_y,
                                test=args.testmode,
                                midnight=args.midnight,
                                labrange=args.labrange,
                                excl_ablab=args.excl_ablab,
                                excl_abchart=args.excl_abchart,
                                split_id=split_id,
                                icv_fold_ids=valid_fold_id,
                                icv_numfolds=num_folds,
                                remapped_data=args.remapped_data,
                                use_mimicid=args.use_mimicid,
                                option_str=args.opt_str,
                                pred_labs=args.pred_labs,
                                pred_normal_labchart=args.pred_normal_labchart,
                                inv_id_mapping=inv_id_mapping,
                                target_size=target_size,
                                elapsed_time=args.elapsed_time,
                                x_as_list=args.x_as_list,
                                )

        va_len = len(valid_dataset[seqlen_idx])
        logger.info("# valid seq: {}".format(va_len))

        if use_valid:
            test_dataset, _ = \
                load_multitarget_data(args.data_name,
                                    'test',
                                    event_size,
                                    data_filter=args.data_filter,
                                    base_path=base_path,
                                    x_hr=args.window_hr_x,
                                    y_hr=args.window_hr_y,
                                    test=args.testmode,
                                    midnight=args.midnight,
                                    labrange=args.labrange,
                                    excl_ablab=args.excl_ablab,
                                    excl_abchart=args.excl_abchart,
                                    split_id=split_id,
                                    icv_fold_ids=valid_fold_id,
                                    icv_numfolds=num_folds,
                                    remapped_data=args.remapped_data,
                                    use_mimicid=args.use_mimicid,
                                    option_str=args.opt_str,
                                    pred_labs=args.pred_labs,
                                      pred_normal_labchart=args.pred_normal_labchart,
                                    inv_id_mapping=inv_id_mapping,
                                    target_size=target_size,
                                    elapsed_time=args.elapsed_time,
                                    x_as_list=args.x_as_list,
                                    )
            valid_dataset = (test_dataset, valid_dataset)

    return train_dataset, valid_dataset, seqlen_idx, train_data_path, \
        target_size, inv_id_mapping
