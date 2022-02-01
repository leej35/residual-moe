# -*- coding: utf-8 -*-

import multiprocessing
multiprocessing.set_start_method('spawn', True)

import os
import sys
import socket
import logging
import pickle

from comet_ml import Experiment
import torch
# import pyro

from utils.project_utils import (objectview, Timing, tprint,
                                 count_parameters)
from utils.tensor_utils import (to_multihot_sorted_vectors_jsb, get_dataset)
from models.self_correct_model import SelfCorrectLSTM

from models.base_seq_model import BaseMultiLabelLSTM, masked_bce_loss
from models.attn_seq_model import AttnMultiLabelLSTM
from models.LR import LogisticRegression
from models.MLP import MLP
from models.seq_moe import SeqMoE

# from models.hmm import (model_hmm, model_hmm_1 as model_hmm_fast, model_5,
#                         pred_model_5, get_preds)
from sklearn.linear_model import LinearRegression
from trainer import load_multitarget_data, load_multitarget_dic

# pyro stuff
# import pyro.distributions as dist
# from pyro import poutine
# from pyro.contrib.autoguide import AutoDelta
# from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
# import pyro.optim as optim
# from pyro.util import ignore_jit_warnings

# optimizer = Optimizer(API_KEY)
optimizer = None
do_multithreading = False

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)


def create_model(args, event_size, device, target_type, vecidx2label,
                 train_dataset=None, train_data_path=None, hidden_dim=None, embedding_dim=None, batch_size=None,
                 num_layers=None, dropout=None, web_logger=None):
    
    if args.self_correct:
        model = SelfCorrectLSTM(
            rnn_type=args.rnn_type,
            event_input_dim=event_size,
            target_dim=args.target_size,
            embed_dim=embedding_dim if embedding_dim else args.embedding_dim,
            hidden_dim=hidden_dim if hidden_dim else args.hidden_dim,
            use_cuda=args.use_cuda,
            device=device,
            num_layers=num_layers if num_layers else args.num_layers,
            dropout=dropout if dropout else args.dropout,
            feed_input=args.corrector_feed_input,
            feed_input_and_hidden=args.corrector_feed_input_and_hidden,
            correct_loss_type=args.correct_loss_type,
            activation=args.corrector_f_act,
            init_bias_zero=args.corrector_init_bias_zero,
            init_weight_small=args.corrector_init_weight_small,
            init_val=args.corrector_init_val,
            no_clamp=args.corrector_no_clamp,
            use_corrector_control=args.use_corrector_control,
            corrector_control_act=args.corrector_control_act,
            corrector_control_inp=args.corrector_control_inp,
            corrector_control_arch=args.corrector_control_arch,
        )

    elif args.moe:
        if args.moe_residual and args.moe_load_gru_model_from:

            base_model = BaseMultiLabelLSTM(
                event_size=args.event_size,
                window_size_y=args.window_hr_y,
                target_size=args.target_size,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embedding_dim,
                batch_size=args.batch_size,
                use_cuda=args.use_cuda,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                rnn_type=args.rnn_type,
                batch_first=args.batch_first,
                device=args.device,
                hier_pred=args.hier_pred,
                recent_bias=args.recent_bias,
                memorize_time=args.memorize_time,
                target_type=args.target_type,
                pooling_type=args.pooling_type,
                dropouth=args.dropouth,
                dropouti=args.dropouti,
                wdrop=args.wdrop,
                tie_weights=args.tie_weights,
                pred_period=args.pred_period,
                pp_type=args.pred_period_type,
                pp_weight_scheme=args.pp_weight_scheme,
                use_simple_gate=args.simple_gate,
                use_orig_params=args.use_orig_params,
                remap=args.remapped_data,
                inv_id_mapping=args.inv_id_mapping,
                lab_pp_proc=args.lab_pp_proc,
                elapsed_time=args.elapsed_time,
            )
            pretrained_dict = torch.load(
                args.moe_load_gru_model_from, map_location=torch.device('cpu'))

            base_model.load_state_dict(pretrained_dict)
        
        else:
            base_model = None

        model = SeqMoE(
            input_size=event_size, 
            embed_size=embedding_dim if embedding_dim else args.embedding_dim, 
            output_size=args.target_size,
            num_experts=args.moe_num_experts, 
            hidden_size=hidden_dim if hidden_dim else args.moe_hidden_dim,
            noisy_gating=args.moe_noisy_gating, k=args.moe_topk,
            dropout=dropout if dropout else args.dropout,
            gate_type=args.moe_gate_type,
            residual=args.moe_residual,
            use_zero_expert=args.moe_zero_expert,
            base_gru=base_model,
            feed_error=args.moe_feed_error,
            incl_base_to_expert=args.moe_incl_base_to_expert,
        )

    elif args.baseline_type in ['sk_timing_linear_reg']:
        model = LinearRegression()

    elif args.baseline_type in ['logistic_binary', 'logistic_count',
                                'logistic_last',
                                'timing_linear_reg']:
        if args.target_event > -1:
            trg_event_size = 1
        else:
            trg_event_size = args.target_size
            
        model = LogisticRegression(event_size, trg_event_size)

    elif args.baseline_type in ['logistic_binary_mlp',
                                'logistic_count_mlp',
                                'logistic_last_mlp']:
        model = MLP(event_size, args.target_size, hidden_dim)

    elif args.baseline_type in ['copylast', 'majority_ts',
                                'majority_all', 'random',
                                'timing_random', 'timing_mean']:
        model = None

    elif args.baseline_type.startswith('hmm'):
        pass
        """
        # args.batch_size = batch_size = tr_len
        if args.baseline_type == 'hmm_fast':
            model = {'model': model_hmm_fast}
        else:
            model = {'model': model_hmm}

        model['get_pred'] = get_preds

        model['guide'] = AutoDelta(poutine.block(model['model'],
             expose_fn=lambda msg: msg["name"].startswith("probs_")))

        # To help debug our tensor shapes, let's print the shape of each
        # site's distribution, value, and log_prob tensor.
        # Note this information is automatically printed on most
        # errors inside SVI.

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD
        elif args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop
        elif args.optimizer == 'clippedadam':
            optimizer = optim.ClippedAdam
        else:
            optimizer = torch.optim.Adam

        if args.lr_scheduler:
            
            optimizer_args = {'lr': args.learning_rate}

            if args.lr_scheduler_multistep:
                scheduler_args = {
                    'optimizer': optimizer,
                    'milestones': args.lr_scheduler_epochs,
                    'gamma': args.lr_scheduler_mult,
                    'optim_args': optimizer_args
                }

                pyro_scheduler = pyro.optim.MultiStepLR(scheduler_args)

            elif args.lr_scheduler_ror:
                scheduler_args = {
                    'optimizer': optimizer,
                    'verbose': True,
                    'optim_args': optimizer_args,
                    'factor': args.lr_scheduler_mult
                }
                pyro_scheduler = pyro.optim.ReduceLROnPlateau(scheduler_args)

            else:
                scheduler_args = {
                    'optimizer': optimizer,
                    'step_size': args.lr_scheduler_numiter, 
                    'gamma': args.lr_scheduler_mult,
                    'optim_args': optimizer_args
                }

                pyro_scheduler = pyro.optim.StepLR(scheduler_args)
 
            # scheduler = pyro_scheduler({'optimizer': optimizer, 
            #     'optim_args': {'lr': args.learning_rate}})

        # Enumeration requires a TraceEnum elbo and declaring the
        # max_plate_nesting.
        # All of our models have two plates: "data" and "tones".
        model['args'] = objectview({'hidden_dim': args.hidden_dim,
                                    'jit': args.jit,
                                    'clamp_prob': args.clamp_prob
                                    })

        Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
        model['elbo'] = Elbo(max_plate_nesting=2 \
            if args.baseline_type == 'hmm_fast' else 1)
        
        model['optim'] = pyro_scheduler if args.lr_scheduler \
            else optimizer(lr=args.learning_rate)

        model['svi'] = SVI(model=model['model'], guide=model['guide'],
                           optim=model['optim'], loss=model['elbo'])
        """
        # if args.lr_scheduler:
        #     model['scheduler'] = pyro_scheduler

    elif args.attn:
        model = AttnMultiLabelLSTM(event_size=event_size,
                                   window_size_y=args.window_hr_y,
                                   target_size=args.target_size,
                                   hidden_dim=hidden_dim \
                                       if hidden_dim else args.hidden_dim,
                                   embed_dim=embedding_dim if embedding_dim \
                                       else args.embedding_dim,
                                   batch_size=batch_size \
                                       if batch_size else args.batch_size,
                                   use_cuda=args.use_cuda,
                                   num_layers=num_layers \
                                       if num_layers else args.num_layers,
                                   dropout=dropout if dropout else args.dropout,
                                   rnn_type=args.rnn_type,
                                   batch_first=args.batch_first,
                                   device=device,
                                   hier_pred=args.hier_pred,
                                   recent_bias=args.recent_bias,
                                   memorize_time=args.memorize_time,
                                   attn_type=args.attn_type,
                                   attn_channel=args.attn_channel,
                                   channel_size=args.attn_channel_size,
                                   target_type=target_type,
                                   attn_q_type=args.attn_q_type,
                                   attn_direct=args.attn_direct,
                                   event_dic=vecidx2label,
                                   train_dataset=train_dataset,
                                   attn_group_type=args.attn_group_type,
                                   attn_temp_embed=args.attn_temp_embed,
                                   pooling_type=args.pooling_type,
                                   attn_co_channel=args.attn_co_channel,
                                   train_data_path=train_data_path,
                                   cluster_only=args.cluster_only,
                                   cluster_method=args.cluster_method,
                                   pred_period=args.pred_period,
                                   pp_type=args.pred_period_type,
                                   pp_attn=args.pred_period_attn,
                                   use_layer_norm=args.layer_norm,
                                   pp_weight_scheme=args.pp_weight_scheme,
                                   use_simple_gate=args.simple_gate,
                                   use_orig_params=args.use_orig_params,
                                   pe=args.attn_pe,
                                   init_decay=args.init_decay,
                                   group_size_method=args.group_size_method,
                                   attn_inner_dim=args.attn_inner_dim,
                                   do_svd=args.do_svd,
                                   freeze_emb=args.freeze_emb,
                                   dropouth=args.dropouth,
                                   dropouti=args.dropouti,
                                   wdrop=args.wdrop,
                                   tie_weights=args.tie_weights,
                                   remap=args.remapped_data,
                                   skip_hidden_state=args.skip_hidden_state,
                                   f_exp=args.f_exp,
                                   f_window=args.f_window,
                                   rb_init=args.rb_init,
                                   manual_alpha=args.manual_alpha,
                                   inv_id_mapping=args.inv_id_mapping,
                                   clock_gate=args.clock_gate,
                                   pp_merge_signal=args.pp_merge_signal,
                                   pp_concat=args.pp_concat,
                                   rb_concat=args.rb_concat,
                                   pp_ascounts=args.pp_ascounts,
                                   )
    else:
        model = BaseMultiLabelLSTM(event_size=event_size,
                                   window_size_y=args.window_hr_y,
                                   target_size=args.target_size,
                                   hidden_dim=hidden_dim \
                                       if hidden_dim else args.hidden_dim,
                                   embed_dim=embedding_dim if embedding_dim \
                                       else args.embedding_dim,
                                   batch_size=batch_size \
                                       if batch_size else args.batch_size,
                                   use_cuda=args.use_cuda,
                                   num_layers=num_layers \
                                       if num_layers else args.num_layers,
                                   num_heads=args.num_heads,
                                   dropout=dropout if dropout \
                                       else args.dropout,
                                   rnn_type=args.rnn_type,
                                   batch_first=args.batch_first,
                                   device=device,
                                   hier_pred=args.hier_pred,
                                   recent_bias=args.recent_bias,
                                   memorize_time=args.memorize_time,
                                   target_type=target_type,
                                   pooling_type=args.pooling_type,
                                   dropouth=args.dropouth,
                                   dropouti=args.dropouti,
                                   wdrop=args.wdrop,
                                   tie_weights=args.tie_weights,
                                   pred_period=args.pred_period,
                                   pp_type=args.pred_period_type,
                                   pp_weight_scheme=args.pp_weight_scheme,
                                   use_simple_gate=args.simple_gate,
                                   use_orig_params=args.use_orig_params,
                                   remap=args.remapped_data,
                                   skip_hidden_state=args.skip_hidden_state,
                                   f_exp=args.f_exp,
                                   f_window=args.f_window,
                                   rb_init=args.rb_init,
                                   manual_alpha=args.manual_alpha,
                                   inv_id_mapping=args.inv_id_mapping,
                                   clock_gate=args.clock_gate,
                                   lab_pp_proc=args.lab_pp_proc,
                                   elapsed_time=args.elapsed_time,
                                   pp_merge_signal=args.pp_merge_signal,
                                   pp_concat=args.pp_concat,
                                   rb_concat=args.rb_concat,
                                   pp_ascounts=args.pp_ascounts,
                                   use_pos_enc=args.use_pos_enc,
                                   tf_d_word=args.tf_d_word,
                                   tf_d_model=args.tf_d_model,
                                   tf_d_inner=args.tf_d_inner,
                                   tf_d_k=args.tf_d_k,
                                   tf_d_v=args.tf_d_v,
                                   tf_by_step=args.tf_by_step,
                                   past_mem=args.past_mem,
                                   past_dist_lt=args.past_dist_lt,
                                   past_dist_st=args.past_dist_st,
                                   past_as_count=args.past_as_count,
                                   tf_pooling=args.tf_pooling,
                                   tf_type=args.tf_type,
                                   pm_softmax=args.pm_softmax,
                                   pred_future_steps=args.pred_future_steps,
                                   tf_use_torch=args.tf_use_torch,
                                   )
      
    if args.use_cuda and model is not None and hasattr(model, 'parameters'):
        model = model.to(device)

    num_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('num_total_params: {}'.format(num_total_params))
    if web_logger:
        web_logger.log_other('num_total_params', num_total_params)

    return model


def prep_dataset(args):
    if args.data_name in ['mimic3', 'instacart', 'tipas']:
        """
        Dataset for Multitarget
        """
        target_type = 'multi'

        if args.data_name == 'mimic3':
            base_path = '{}/mimic_cs3750.sequence/'.format(args.base_path)
        elif args.data_name == 'instacart':
            base_path = '{}/{}.data/'.format(args.base_path, args.data_name)
        elif args.data_name == 'tipas':
            base_path = args.base_path
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
                                                        test=args.testmode)

        # if args.data_name == 'instacart':
        # NOTE: we don't have category in instacart. so it will not
        # be needed for evaluation computation and so on.

        # NOTE: (revise at May 29 2019) We will need  it to print in
        # cluster info.
        # vecidx2label = None

        train_dataset, train_data_path, train_hadmids = \
            load_multitarget_data(args.data_name, 'train',
                                  event_size,
                                  data_filter=args.data_filter,
                                  base_path=base_path,
                                  x_hr=args.window_hr_x,
                                  y_hr=args.window_hr_y,
                                  test=args.testmode,
                                  elapsed_time=args.elapsed_time,
                                  midnight=args.midnight,
                                  labrange=args.labrange,
                                  excl_ablab=args.excl_ablab,
                                  excl_abchart=args.excl_abchart
                                  )

        test_dataset, _, _ = load_multitarget_data(args.data_name, 'test',
                                                event_size,
                                                data_filter=args.data_filter,
                                                base_path=base_path,
                                                x_hr=args.window_hr_x,
                                                y_hr=args.window_hr_y,
                                                test=args.testmode,
                                                elapsed_time=args.elapsed_time,
                                                midnight=args.midnight,
                                                labrange=args.labrange,
                                                excl_ablab=args.excl_ablab,
                                                excl_abchart=args.excl_abchart
                                                )

        valid_dataset, _, _ = load_multitarget_data(args.data_name, 'valid',
                                                 event_size,
                                                 data_filter=args.data_filter,
                                                 base_path=base_path,
                                                 x_hr=args.window_hr_x,
                                                 y_hr=args.window_hr_y,
                                                 test=args.testmode,
                                                 elapsed_time=args.elapsed_time,
                                                 midnight=args.midnight,
                                                 labrange=args.labrange,
                                                 excl_ablab=args.excl_ablab,
                                                 excl_abchart=args.excl_abchart
                                                 )

        seqlen_idx = 2
        loss_fn = torch.nn.BCELoss() #masked_bce_loss

    elif args.data_name == 'jsb':
        import models.dmm.polyphonic_data_loader as poly
        jsb_data = poly.load_data(poly.JSB_CHORALES)

        train_dataset = to_multihot_sorted_vectors_jsb(
            jsb_data["train"]["sequences"])
        test_dataset = to_multihot_sorted_vectors_jsb(
            jsb_data["test"]["sequences"])
        valid_dataset = to_multihot_sorted_vectors_jsb(
            jsb_data["valid"]["sequences"])

        event_size = jsb_data["train"]["sequences"].size()[-1]
        train_data_path = "{}/tmp_jsb_data/".format(args.base_path)
        os.system("mkdir -p {}".format(train_data_path))
        seqlen_idx = 2
        vecidx2label = None
        loss_fn = masked_bce_loss
        target_type = 'multi'

    else:
        """ 
        Dataset for non-mimic3 (single target) 
        """
        target_type = 'single'

        base_path = '{}/{}.data/{}'.format(args.base_path,
                                           args.data_name,
                                           args.data_filter)

        # TODO: add test datset mode
        data_tensors, event_size = get_dataset(base_path,
                                               args.data_name,
                                               cv_fold=None)
        train_dataset = data_tensors['train']
        test_dataset = data_tensors['test']
        valid_dataset = data_tensors['valid']
        seqlen_idx = 0
        vecidx2label = None
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        train_data_path = base_path

    logger.info('event_size: {}'.format(event_size))

    tr_len = len(train_dataset[seqlen_idx])
    va_len = len(valid_dataset[seqlen_idx])
    te_len = len(test_dataset[seqlen_idx])

    logger.info("# train seq: {}".format(tr_len))
    logger.info("# valid seq: {}".format(va_len))
    logger.info("# test seq: {}".format(te_len))

    return train_dataset, test_dataset, valid_dataset, event_size, \
           train_data_path, seqlen_idx, vecidx2label, loss_fn, target_type


def load_model(args, trainer, device, login, renew_token):
    if login:
        renew_token()
    with Timing(f'Loading model from {args.load_model_from}...', logger=logger):
        
        try:
            pretrained_dict = torch.load(
                args.load_model_from, map_location=torch.device('cpu'))

        except AttributeError:
            logger.info(
                'Model is not just state_dict, loading whole model '
                '(source code change might affect the performance).')
            if not args.use_cuda:
                pretrained_dict = torch.load(args.load_model_from,
                                           map_location='cpu')
            else:
                pretrained_dict = torch.load(args.load_model_from)
    
    if args.self_correct and args.correct_mode == 'train_corrector':
        # pretrained_dict = pretrained_model.state_dict()
        model_dict = trainer.model.state_dict()

        # Fiter out unneccessary keys
        things_to_load = ('predictor',)
        filtered_dict = {k: v for k,
                         v in pretrained_dict.items() if k.startswith(things_to_load)}
        model_dict.update(filtered_dict)
        trainer.model.load_state_dict(model_dict)
    else:
        trainer.model.load_state_dict(pretrained_dict)
    
    if args.use_cuda and not args.baseline_type.startswith("hmm"):
        trainer.model = trainer.model.to(device)

    
    return trainer


def evaluation(model, trainer, args, web_logger, device, test_dataset,
               save_final_model, eval_test_only=True, train_dataset=None,
               valid_dataset=None) :
    if args.load_model_from is None:
        web_logger.log_parameter('best_epoch', trainer.best_epoch)

        checkpoint_name = '{}epoch_{}.model'.format(trainer.model_prefix,
                                                    trainer.best_epoch)
        logger.info('Load best validation-f1 model (at epoch {}): {}...'.format(
            trainer.best_epoch, checkpoint_name))
        trainer.load(checkpoint_name)
        if trainer.use_cuda and not args.baseline_type.startswith('hmm'):
            model.to(device)

    with Timing('Doing final evaluation...', logger=logger):
        if not eval_test_only:
            logger.info('Results on training data')
            sys.stdout.flush()
            trainer.infer_model(train_dataset, test_name='train', final=True)
            logger.info('Results on valid data')
            sys.stdout.flush()
            trainer.infer_model(valid_dataset, test_name='valid', final=True)

        logger.info('Results on test data')
        sys.stdout.flush()
        if args.eval_on_cpu and not args.baseline_type.startswith('hmm'):
            logger.info('Run evaluation on CPU')
            cpu = torch.device('cpu')
            trainer = trainer.to(cpu)
            trainer.device = cpu
            trainer.model.device = cpu
            trainer.use_cuda = False
            trainer.model.use_cuda = False
        _, _, _, test_final_f1 = trainer.infer_model(test_dataset,
                                                     test_name='test',
                                                     final=True)

    if save_final_model:
        final_model_name = '{}_final.model'.format(args.model_prefix)
        with Timing('Saving final model to {final_model_name} ...', logger=logger):
            if model is not None:
                if hasattr(model, 'parameters'):
                    torch.save(model.state_dict(), final_model_name)
                else:
                    pickle.dump(model, open(final_model_name, 'wb'),
                                protocol=pickle.HIGHEST_PROTOCOL)

        os.system('rm -rf {}epoch*.model'.format(args.model_prefix))



def get_weblog(api_key, args):
    if args.force_comet_off:
        class Empty():
            def log_figure(self, *args, **kwargs):
                pass
            def log_table(self, *args, **kwargs):
                pass
            def log_parameter(self, *args, **kwargs):
                pass
            def log_metric(self, *args, **kwargs):
                pass
            def log_other(self, *args, **kwargs):
                pass
            def log_parameters(self, *args, **kwargs):
                pass
            def end(self, *args, **kwargs):
                pass
        web_logger = Empty()

    elif not args.testmode or args.force_comet:
        web_logger = Experiment(api_key=api_key, auto_param_logging=True,
                                auto_output_logging="simple",
                                auto_metric_logging=True,
                                workspace="jeongmin",
                                project_name=args.data_name+'-v4')
    else:
        class Empty():
            def log_parameter(self, *args, **kwargs):
                pass
            def log_metric(self, *args, **kwargs):
                pass
            def log_other(self, *args, **kwargs):
                pass
            def log_parameters(self, *args, **kwargs):
                pass
            def end(self, *args, **kwargs):
                pass
        web_logger = Empty()
    return web_logger

def add_main_setup(args):
    device = torch.device("cuda:{}".format(args.gpu_id) \
                          if args.use_cuda else 'cpu')

    if socket.gethostname().split('.')[0] in ['vector', 'euler']\
            or socket.gethostname()[-12:] == 'crc.pitt.edu' \
            or socket.gethostname().split('.')[-1] == 'local' \
            or socket.gethostname()[-17:] == 'wireless.pitt.edu':
        login = False
    else:
        login = True

    if not login:
        renew_token = lambda *args: None  # empty function
    else:
        from AFSLogin import renew_token, init_token
        init_token()

    if 'timing' in args.baseline_type:
        args.pred_time = True

    if args.data_filter is None:
        args.data_filter = ''

    if login:
        renew_token()

    os.system("mkdir -p eval_obj")
    os.system("mkdir -p trained")
    os.system("mkdir -p plots")
    os.system("mkdir -p plots/train_progress")

    args.device = device
    args.login = login
    args.renew_token = renew_token

    return args
