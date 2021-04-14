from comet_ml import Experiment

from AFSLogin import init_token, API_KEY
import logging
import numpy as np
from utils.args import get_parser, print_args
from utils.hypertune_utils import (
    combine_namedtuple, get_dict_and_info,
    get_dataset, hyperparam_tuning, get_hyperparam_setttings)

import multiprocessing
multiprocessing.set_start_method('spawn', True)

from models.base_seq_model import BaseMultiLabelLSTM, masked_bce_loss
from models.seq_moe import SeqMoE

# from multiprocessing import Pool
import os
import sys

# import pickle
import torch
from multiprocessing.reduction import ForkingPickler
from torch.multiprocessing import reductions
from torch.utils.data import dataloader
import torch.nn.functional as F

import copy
from utils.project_utils import Timing
from trainer import Trainer
from models.base_seq_model import masked_bce_loss
from main_utils import (create_model, add_main_setup, get_weblog,
                        load_model)


import matplotlib.pyplot as plt

SEED = 5

np.random.seed(SEED)
torch.manual_seed(SEED)
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)



def main():
    
    init_token()
    # args
    args = get_parser()
    args = add_main_setup(args)

    # let's assume each split is executed in different machines
    # hyperparam (grid search is happen in a single machine)
    # so I should be able to run each file for each split in each machine.
    # (later, a job manager can be built)

    # single machine from here
    logger.info('split: {}'.format(args.split_id))

    args = get_dict_and_info(args)

    # Step1. load/train base model

    if not args.skip_hypertuning:
        hyperparam_settings = get_hyperparam_setttings(args)
        best_hyper_param, best_epoch = hyperparam_tuning(copy.deepcopy(args),
                                                         hyperparam_settings)
        args = combine_namedtuple(args, best_hyper_param)

    # # training for post-hypertuning run
    logger.info('\n{}'.format('=' * 64))
    logger.info('start final run')
    logger.info('\n{}'.format('-' * 64))
    web_logger = get_weblog(API_KEY, args)
    web_logger.log_other('model_name', args.model_name)
    web_logger.log_other('run_mode', 'final_run')
    web_logger.log_other('split_id', args.split_id)
    web_logger.log_other('code_name', args.code_name)
    web_logger.log_other('fp16_opt_level', args.fp16_opt_level)
    web_logger.log_other('fp16', args.fp16)
    print_args(args)

    if not args.skip_hypertuning:
        for name, value in best_hyper_param._asdict().items():
            if name:
                web_logger.log_parameter(name, value)

    with Timing('Loading data files...\n', logger=logger):
        train_dataset, test_dataset, seqlen_idx, train_data_path, target_size, \
            inv_id_mapping \
            = get_dataset(args, args.split_id, args.event_size, args.base_path,
                          use_valid=args.use_valid,
                          simulated_data=args.simulated_data)
        
        args.target_size = target_size
        args.inv_id_mapping = inv_id_mapping

        if args.use_valid:
            (test_dataset, valid_dataset) = test_dataset
        elif args.vbv:
            valid_dataset = test_dataset
        else:
            valid_dataset = None  # means no early stopping

    # Create base Model

    with Timing('Creating model and trainer...', logger=logger):
        model = create_model(args, args.event_size, args.device,
                            args.target_type, args.vecidx2label,
                            train_dataset,
                            train_data_path,
                            web_logger=web_logger,
        )
        trainer = Trainer(
            model, args=args,
            d_path=train_data_path, web_logger=web_logger)

    if not args.skip_hypertuning:
        trainer.epoch = best_epoch
        trainer.force_epoch = True  # on final train, stop by best epoch
        web_logger.log_other('best_epoch', best_epoch)
        logger.info('best_epoch from hypertuning: {}'.format(best_epoch))

    logger.info('Start SeqMoE model Final Train')
    trainer.train_model(train_dataset, do_checkpoint=False,
                        valid_data=valid_dataset)


    # Step5. test set running


    if args.adapt_mem:
        logger.info('='*32)
        logger.info("[main] Run train set memory write process")
        trainer.infer_model(train_dataset, cur_epoch=0,
                    test_name='ltam_write', return_metrics=False)

    if args.train_error_pred:
        logger.info('='*32)
        logger.info("[main] Run train for predicting error model")
        trainer.run_train_error_pred()

    with Timing('Save final trained model\n', logger=logger):
        trainer.save('{}_final.model'.format(args.model_prefix))

    logger.info('='*32)
    logger.info("[main] Run quick test set")
    trainer.infer_model(test_dataset, cur_epoch=0,
                        test_name='ltam_read', return_metrics=False)

    if args.eval_on_cpu and not args.baseline_type.startswith('hmm'):
        logger.info('='*32)
        logger.info('Run evaluation on CPU')
        cpu = torch.device('cpu')
        trainer = trainer.to(cpu)
        trainer.device = cpu
        trainer.model.device = cpu
        trainer.use_cuda = False
        trainer.model.use_cuda = False
        trainer.model = trainer.model.to(cpu)

        if hasattr(trainer, 'ltam_model'):
            trainer.ltam_model = trainer.ltam_model.to(cpu)
            trainer.ltam_model.device = cpu
            trainer.ltam_model.use_mem_gpu = False

        if hasattr(trainer.model, 'pp'):
            trainer.model.pp = trainer.model.pp.to(cpu)
            trainer.model.pp.device = cpu
            trainer.model.pp.use_cuda = False
            if hasattr(trainer.model.pp, 'pp'):
                logger.info('move pp obj to cpu')
                # trainer.model.pp.pp._to(cpu)
                trainer.model.pp.pp.device = cpu
                trainer.model.pp.pp.use_cuda = False
                logger.info('trainer.model.pp.pp.device: {}'.format(trainer.model.pp.pp.device))

    logger.info('='*32)
    with Timing('Doing final evaluation...', logger=logger):
        if args.load_model_from is None and valid_dataset is not None:
            trainer.load_best_epoch_model()
            trainer.save_final_model()

        logger.info('{}'.format('-' * 64))
        logger.info('Eval stats')
        logger.info('{}'.format('-' * 64))
        
        last_slash = args.model_prefix.rfind('/')
        csvfile = args.model_prefix[:last_slash + 1] + 'metric'

        logger.info('='*32)
        logger.info("[main] Debug 2: Run quick test set")
        trainer.infer_model(test_dataset, cur_epoch=0,
                            test_name='ltam_read', return_metrics=False)

        logger.info('='*32)
        logger.info("[main] Final Test result:")

        eval_stats = trainer.infer_model(test_dataset,
                                         test_name='final_test', final=True,
                                         export_csv=True,
                                         csv_file=csvfile + '_test.csv',
                                         eval_multithread=args.eval_multithread)

        logger.info('\n{}'.format('-' * 64))
        if not args.skip_train_eval:
            logger.info('Train stats')
            logger.info('{}'.format('-' * 64))
            train_stats = trainer.infer_model(train_dataset,
                                            test_name='final_train', final=True,
                                            export_csv=True,
                                            csv_file=csvfile + '_train.csv',
                                            eval_multithread=args.eval_multithread
                                            )

    logger.info('eval stats: {}'.format(eval_stats))


if __name__ == '__main__':
    main()
