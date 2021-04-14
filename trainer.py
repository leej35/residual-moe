# -*- coding: utf-8 -*-

# Import statements
import torch.optim as optim
from torch.autograd import Function
import sys
import logging
import os
import copy
import time
import warnings
import socket
from packaging import version

import multiprocessing
multiprocessing.set_start_method('spawn', True)


import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from models.memnet import MemNetE2E, PredErrorModel

# import pyro
# from pyro.optim import ClippedAdam
from utils.project_utils import cp_save_obj, cp_load_obj
from models.transformer.Optim import ScheduledOptim
from models.base_seq_model import masked_bce_loss

tabulate.PRESERVE_WHITESPACE = True

from utils.project_utils import Timing, repackage_hidden, set_event_mask, \
    load_multitarget_data, pack_n_sort, load_simulated_data, load_multitarget_dic, \
    get_data_file_path, masked_unroll_loss, masked_loss, topk_pred,  flat_tensor
from utils.evaluation_utils import MultiLabelEval
from utils.evaluation_utils_multiclass import MultiClassEval
from utils.tensor_utils import fit_input_to_output
from utils.trainer_utils import (
    remove_zeroed_batch_elems, get_optimizer, get_scheduler, get_dataloader, 
    _get_memorize_time, process_batch, 
    get_event_weights, run_evals, run_non_RNN_models, get_hash
)

from models.adaptive_models import train_adaptive_model, neural_caching_model
from models.adaptive_persistent_cache import (
    train_error_pred_model, LTAM)

# from pytorch_revgrad import RevGrad


_use_native_amp = False
_use_apex = False

# # Force to use APEX
# from utils.project_utils import is_apex_available

# if is_apex_available():
#     from apex import amp
# _use_apex = True

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):

    from utils.project_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)

warnings.filterwarnings('ignore')

eps = 0

debug = False
debug_test = False
debug_grad = False
debug_vis = True
softmax = nn.Softmax(dim=2)
logsoftmax = nn.LogSoftmax()
sigmoid = nn.Sigmoid()
relu = nn.ReLU()


class Trainer(nn.Module):
    def __init__(self,
                 model,
                 event_size=None,
                 loss_fn=None,
                 loss_time_fn=None,
                 epoch=101, learning_rate=0.001,
                 print_every=10, valid_every=20, save_every=20,
                 model_prefix='', use_cuda=False, batch_size=64,
                 batch_first=True,
                 curriculum_learning=False, curriculum_rate=1.35,
                 max_seq_len_init=2,
                 weight_decay=0,
                 lr_scheduler=False, lr_scheduler_numiter=15,
                 lr_scheduler_multistep=False, lr_scheduler_epochs=[],
                 lr_scheduler_mult=0.5,
                 lr_scheduler_ror=False,
                 num_workers=4,
                 web_logger=None,
                 optim='adam',
                 pred_time=False,
                 bptt=0,
                 patient=None,
                 scale=0.001,
                 device=None,
                 baseline_type='None',
                 window_hr_x=None,
                 window_hr_y=None,
                 event_dic=None,
                 memorize_time=False,
                 target_type='multi',
                 use_bce_logit=False,
                 use_bce_stable=False,
                 pp_type='default',
                 d_path=None,
                 renew_token_fn=lambda *args: None,
                 aime_2019_eval=False,
                 aime_2019_eval_macro=False,
                 args=None,
                 target_size=None,
                 ):
        """

        :rtype: None
        """
        super(Trainer, self).__init__()

        self.model = model
        self.d_path = d_path

        self.web_logger = web_logger

        self.event_size = args.event_size
        self.target_type = args.target_type
        self.event_dic = args.event_dic
        self.loss_fn = args.loss_fn
        self.renew_token_fn = args.renew_token

        self.device = args.device

        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
        self.use_cuda = args.use_cuda
        self.batch_size = args.batch_size
        self.print_every = args.print_every
        self.valid_every = args.valid_every
        self.save_every = args.save_every
        self.model_prefix = args.model_prefix
        self.batch_first = args.batch_first
        self.num_workers = args.num_workers

        self.curriculum_learning = args.curriculum_learning
        self.curriculum_rate = args.curriculum_rate
        self.max_seq_len_init = args.curriculum_init

        self.lr_scheduler = args.lr_scheduler
        self.lr_scheduler_multistep = args.lr_scheduler_multistep
        self.lr_scheduler_epochs = args.lr_scheduler_epochs
        self.lr_scheduler_numiter = args.lr_scheduler_numiter
        self.lr_scheduler_mult = args.lr_scheduler_mult
        self.weight_decay = args.weight_decay
        self.optim = args.optimizer
        self.use_bce_logit = args.use_bce_logit
        self.use_bce_stable = args.use_bce_stable

        self.pred_time = args.pred_time
        self.bptt = args.bptt
        self.patient = args.patient_stop

        self.baseline_type = args.baseline_type
        self.window_hr_x = args.window_hr_x
        self.window_hr_y = args.window_hr_y

        self.memorize_time = args.memorize_time
        self.pp_type = args.pred_period_type
        self.aime_2019_eval = args.aime_eval
        self.aime_2019_eval_macro = args.aime_eval_macro

        self.lr_scheduler_ror = args.lr_scheduler_ror

        # time-pred related args (needs to be part of Trainer args)
        self.scale = scale
        self.loss_time_fn = loss_time_fn
        self.force_epoch = args.force_epoch

        self.target_event = args.target_event
        self.force_checkpointing = args.force_checkpointing
        self.force_auroc = args.force_auroc
        self.force_plot_auroc = args.force_plot_auroc
        self.track_weight_change = args.weight_change
        self.loss_tol = args.loss_tol
        self.args = args
        self.elapsed_time = args.elapsed_time

        self.skip_save_prior = (args.simulated_data or args.data_name == 'tipas')
        self.pred_future_steps = args.pred_future_steps

        self.adapt_lstm = args.adapt_lstm
        self.adapt_residual = args.adapt_residual
        self.adapt_lstm_only = args.adapt_lstm_only
        self.adapt_fc_only = args.adapt_fc_only
        self.event_weight_loss = args.event_weight_loss

        self.adapt_mem = args.adapt_mem
        self.neural_caching = args.neural_caching
        self.ncache_window = args.ncache_window
        self.ncache_theta = args.ncache_theta
        self.ncache_lambdah = args.ncache_lambdah
        self.early_terminate_inference = args.early_terminate_inference
        self.mem_size = args.mem_size
        self.read_mode = args.read_mode
        self.log_hidden_target_error_file = args.log_hidden_target_error_file
        self.train_error_pred = args.train_error_pred
        self.train_learn_to_use_mem = args.train_learn_to_use_mem
            
        self.event_types = None
        self.topk_threshold = 0
        self.pos_weight = None
        self.best_epoch = 0

        self.da = args.da  # domain adaptation strategy
        self.da_change_every = args.da_change_every
        self.da_input = args.da_input
        self.da_percentile = args.da_percentile  # to split source and target sets
        self.da_lambda = args.da_lambda
        self.da_pooling = args.da_pooling
        # self.da_opposite_loss = args.da_opposite_loss

        # domain adaptation data structures
        self.da_inst_loss = {}
        self.da_domain_dic = {'source': [], 'target': []}
        self.da_domain_dic_inv = {}
        
        self.da_f_loss = nn.BCELoss(reduction='none')
        self.da_f_domain_loss = nn.BCELoss()
        if self.da_input == 'hidden':
            da_input_dim = args.hidden_dim
        elif self.da_input == 'proj':
            da_input_dim = args.target_size
        if self.da:            
            da_domain_pred_layer = nn.Sequential(
                GradientReversal(),
                nn.Linear(da_input_dim, args.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(args.hidden_dim, 1),
                PoolingLayer(self.da_pooling, dim=1),
                nn.Sigmoid(),
            ).to(self.device)
            self.model.da_domain_pred_layer = da_domain_pred_layer

        self.timing_vec = torch.zeros(1, self.event_size)
        self.event_counter = torch.zeros(self.event_size)

        if self.event_dic and 'category' in list(self.event_dic.values())[0]:
            self.event_types = list(set([list(self.event_dic.values())[i]['category'] \
                                         for i in
                                         range(len(list(self.event_dic.values())))]))
            self.event_types += ['lab_normal', 'lab_abnormal', 'chart_normal', 
                                 'chart_abnormal']
             
        if self.use_cuda:
            self.event_counter = self.event_counter.to(self.device)
            self.timing_vec = self.timing_vec.to(self.device)

        if self.baseline_type in ['sk_timing_linear_reg', 'copylast',
                                  'majority_ts', 'majority_all',
                                  'timing_random', 'timing_mean', 'random']:
            self.epoch = 1

        
        # counters for adaptive learning
        self.cnt_update = 0
        self.cnt_time_step = 0

        self.patient_count_steps = {}
        self.adapt_switch_stat = {'pop': 0, 'inst': 0}
        self.adapt_switch_stat_steps = {}
        
        if self.adapt_mem:
            self.ltam_model = LTAM(args)

        if self.args.fp16 and _use_native_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.args.fp16:
            logger.info(
                f"fp16: use_native_amp: {_use_native_amp} use_apex: {_use_apex}"
            )

        self.self_correct = args.self_correct

        self.subgroup_adaptation = args.subgroup_adaptation
        self.adapt_on_error = args.sg_adapt_on_error

        self.moe = args.moe

    def run_train_error_pred(self):
        """
        train a error-prediction model 
        """
        assert self.hidden_target_error is not None, \
            "run infer_model with train data with --log_hidden_target_error_file first"
        hidden = self.hidden_target_error["hidden"]
        error = self.hidden_target_error["error"]
        self.model.pred_error_model = train_error_pred_model(
            self, self.model.pred_error_model, hidden, error, 
            self.args.pred_error_train_stop_gap, 
            self.args.pred_error_train_min_epoch,
            self.args.pred_error_learning_rate,
        )


    def train_model(
        self, train_data, valid_data=None, do_checkpoint=True, 
    ):
        best_metric = 0
        best_loss = 100
        prev_loss = 0

        if self.event_weight_loss is not None:
            self.event_weights = get_event_weights(self, self.event_weight_loss)
        else:
            self.event_weights = None
            
        if self.args.ptn_event_cnt:
            import collections
            c = collections.Counter()

        if self.model != None and hasattr(self.model, 'parameters'):
            optimizer = get_optimizer(self)

            if self.lr_scheduler and not self.baseline_type.startswith('hmm'):
                scheduler = get_scheduler(self, optimizer)

            # override 
            if hasattr(self.model, 'rnn_type') and self.model.rnn_type in [
                    'transformer', 'transformer-lstm', 'transformer-sep-lstm']:
                optimizer = ScheduledOptim(
                    optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09, 
                    weight_decay=self.weight_decay),
                    init_lr=self.args.learning_rate, d_model=self.args.hidden_dim,
                    n_warmup_steps=self.args.n_warmup_steps)

        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex \
                        to use fp16 training.")
            self.model, optimizer = amp.initialize(
                self.model, optimizer, opt_level=self.args.fp16_opt_level)


        start_time = time.time()
        max_seq_len = 0
        patient_cnt = 0

        logger.info('Start training')

        if self.pp_type == 'adaptive':
            dataloader = get_dataloader(self, train_data, shuffle=False)
            logger.info(self.d_path)
            self.model.pp.load_prior(dataloader, d_path=self.d_path,
                                     prep_loader=False, force_reset=self.skip_save_prior,
                                     not_save=self.skip_save_prior, 
                                     prior_from_mimic=self.args.prior_from_mimic,
                                     vecidx2mimic=self.args.vecidx2mimic
                                     )
            del dataloader

        if hasattr(self.model, "rb_init") and hasattr(self.model, "pred_period") \
                and "prior" in self.model.rb_init and self.model.pred_period:
            
            prior_next_bin = self.model.pp.prior[:, 0].to(self.device)
            
            if "asbias" in self.model.rb_init:
                self.model.r_bias_linear.bias.data.fill_(0)
                self.model.r_bias_linear.bias.data += prior_next_bin
            
            if "asweight" in self.model.rb_init:
                self.model.r_bias_linear.weight.data[list(range(self.model.event_size)), list(
                    range(self.model.event_size))] += prior_next_bin
                self.model.r_bias_linear.bias.data.fill_(0)

            
        if self.memorize_time and self.target_type == 'multi':
            dataloader = get_dataloader(self, train_data, shuffle=False)
            _get_memorize_time(self, dataloader)
            del dataloader

        epoch = 0

        while True:
            def run_train():
                if self.baseline_type in ['sk_timing_linear_reg']:
                    train_x, train_y = [], []
                else:
                    train_x, train_y = None, None

                item_avg_cnt = []
                dataloader = get_dataloader(self, train_data, shuffle=False)
                t_loss = 0
                t_da_loss = 0
                epoch_start_time = time.time()

                for i, data in enumerate(dataloader):
                    update_cnt = 0

                    inp, trg, len_inp, len_trg, inp_time, trg_time, hadm_ids \
                        = process_batch(self, data, epoch)

                    if self.moe:
                        pred, gating_score = self.model(inp, len_inp, trg)

                        loss = self.loss_fn(pred, trg.float(),
                                            len_trg,
                                            self.use_bce_logit,
                                            self.use_bce_stable,
                                            pos_weight=self.pos_weight,
                                            event_weight=self.event_weights
                        )
                        t_loss += loss.item()

                        loss.backward()
                        optimizer.step()
                        self.model.zero_grad()


                    elif self.baseline_type == 'None':
                        
                        # all RNNs based models
                        
                        if type(inp) == torch.Tensor:
                            batch_size, seq_len, _ = inp.size()
                            
                        else:
                            batch_size = len(inp)
                            seq_len = len(inp[0])

                        if self.args.fp16 and _use_native_amp:
                            with autocast():
                                hidden = self.model.init_hidden(batch_size=batch_size)
                        else:
                            hidden = self.model.init_hidden(batch_size=batch_size)

                        bptt_size = self.bptt if self.bptt else seq_len
                        for j in range(0, seq_len, bptt_size):

                            if self.bptt:
                                seqlen = min(self.bptt, seq_len - j)
                            else:  # no bptt
                                seqlen = seq_len

                            if type(inp) == torch.Tensor:
                                inp_seq = inp[:, j:j + seqlen]
                            else:
                                inp_seq = [ibatch[j:j + seqlen] for ibatch in inp]

                            if type(trg) == torch.Tensor:
                                trg_seq = trg[:, j:j + seqlen]
                            else:
                                trg_seq = [ibatch[j:j + seqlen] for ibatch in trg]

                            if trg_time is not None:
                                if type(trg_time) == torch.Tensor:
                                    trg_time_seq = trg_time[:, j:j + seqlen]
                                else:
                                    trg_time_seq = [ibatch[j:j + seqlen] for ibatch in trg_time]
                            else:
                                trg_time_seq = None

                            if inp_time is not None:
                                if type(inp_time) == torch.Tensor:
                                    inp_time_seq = inp_time[:, j:j + seqlen]
                                else:
                                    inp_time_seq = [ibatch[j:j + seqlen] for ibatch in inp_time]
                            else:
                                if type(inp) == torch.Tensor:
                                    inp_time_seq = torch.zeros(
                                        inp_seq.size()).to(self.device)
                                else:
                                    inp_time_seq = [[[]]]

                            seqlen_v = torch.LongTensor([seqlen] * batch_size)
                            
                            if type(len_inp) == list:
                                len_inp = torch.tensor(len_inp)
                            if type(len_trg) == list:
                                len_trg = torch.tensor(len_trg)

                            len_inp_step = torch.min(len_inp, seqlen_v)
                            len_inp -= len_inp_step
                            len_trg_step = torch.min(len_trg, seqlen_v)
                            len_trg -= len_trg_step

                            # inp_seq: n_batch * max_seq_len * n_events
                            # x : max_seq_len * n_events
                            # len_seq : n_batch
                            
                            if sum(len_inp_step) < 1:
                                continue

                            hidden = repackage_hidden(hidden)

                            # removing zero-lengths batch elements
                            if type(hidden) != list:
                                hidden = hidden.squeeze(0)

                            hidden, inp_seq, trg_seq, len_inp_step, len_trg_step, inp_time_seq, trg_time_seq = \
                                remove_zeroed_batch_elems(self, 
                                    hidden, inp_seq, trg_seq, len_inp_step,
                                    len_trg_step,
                                    trg_time_seq=trg_time_seq,
                                    inp_time_seq=inp_time_seq,
                                )

                            def is_hier_lstm():
                                if hasattr(self.model, 'hier_lstms'): 
                                    if self.model.rnn_type in self.model.hier_lstms:
                                        return True
                                return False

                            if is_hier_lstm() and type(hidden) != list:
                                hidden = hidden.unsqueeze(0)

                            forward_f_input = [
                                inp_seq, trg_seq, len_inp_step, hidden,
                                trg_time_seq, inp_time_seq
                            ]

                            if self.da and self.da_domain_dic_inv != {}:  
                                # domain adaptation: compute domain target
                                da_domain_trg = torch.FloatTensor([
                                    self.da_domain_dic_inv[get_hash(inst_seq)] == 'target' \
                                        for inst_seq in inp_seq.cpu().numpy()
                                ]).to(inp_seq.device)
                                forward_f_input.append(da_domain_trg)

                            if self.args.fp16 and _use_native_amp:
                                with autocast():
                                    hidden, loss, pred_seq, da_loss = self.forward_and_loss(
                                        *forward_f_input
                                    )

                            else:
                                hidden, loss, pred_seq, da_loss = self.forward_and_loss(
                                    *forward_f_input
                                )    

                            # len_bptt = sum(len_inp_step).float()
                            # if self.use_cuda:
                            #     len_bptt = len_bptt.to(self.device)
                            #
                            # loss = loss / len_bptt

                            t_loss += loss.item()
                            
                            if da_loss:
                                t_da_loss += da_loss

                            if self.args.grad_accum_steps > 1:
                                loss = loss / self.args.grad_accum_steps

                            if self.args.fp16 and _use_native_amp:
                                self.scaler.scale(loss).backward()
                            elif self.args.fp16 and _use_apex:
                                with amp.scale_loss(loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                loss.backward()

                            if (i + 1) % self.args.grad_accum_steps == 0:
                                if self.args.fp16 and _use_native_amp:
                                    self.scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(
                                        self.model.parameters(), 
                                        self.args.max_grad_norm
                                    )
                                elif self.args.fp16 and _use_apex:
                                    torch.nn.utils.clip_grad_norm_(
                                        amp.master_params(optimizer), 
                                        self.args.max_grad_norm
                                    )
                                else:
                                    torch.nn.utils.clip_grad_norm_(
                                        self.model.parameters(), 
                                        self.args.max_grad_norm
                                    )

                                if self.args.fp16 and _use_native_amp:
                                    self.scaler.step(optimizer)
                                    self.scaler.update()
                                else:
                                    optimizer.step()

                                self.model.zero_grad()

                            if self.da:
                                # compute bce prediction loss and save it into dict 
                                da_loss = self.da_f_loss(pred_seq, trg_seq)
                                da_loss, inp_seq = da_loss.cpu(), inp_seq.cpu()
                                for inst_loss, inst_seq in zip(da_loss, inp_seq):
                                    k = get_hash(inst_seq.numpy())
                                    
                                    # average across time and events altogether
                                    avg_inst_loss = inst_loss.mean() 
                                    if k not in self.da_inst_loss:
                                        self.da_inst_loss[k] = []
                                    self.da_inst_loss[k].append(avg_inst_loss.item())
                                    

                            if self.args.ptn_event_cnt:
                                for bseq in trg_seq:
                                    ptn_items = [
                                        x.item() for x in \
                                            torch.nonzero(bseq.sum(0)).cpu()
                                    ]
                                    for e in ptn_items:
                                        c[e] += 1

                    else:
                        output, item_avg_cnt, t_loss, update_cnt, train_x, train_y \
                            = run_non_RNN_models(
                                self, item_avg_cnt, inp, trg, trg_time, len_inp, 
                                train_x, train_y, optimizer, t_loss, update_cnt,)

                    # if self.model is not None and hasattr(self.model,
                    #                                       'parameters'):
                    #     logger.info('update_cnt: {}'.format(update_cnt))
                    #     logger.info('t_loss: {}'.format(t_loss))

                # At the end of each epoch

                if self.baseline_type in ['sk_timing_linear_reg']:
                    train_x = flat_tensor(train_x).numpy()
                    train_y = flat_tensor(train_y).numpy()

                    self.model.fit(train_x, train_y)

                if self.baseline_type == 'timing_mean':
                    self.timing_vec = torch.mean(self.timing_vec, dim=0)

                if self.baseline_type.startswith('majority'):
                    self.topk_threshold = int(np.ceil(np.mean(item_avg_cnt)))

                if self.model is None:
                    logger.info('topk_threshold: {}'.format(self.topk_threshold))

                # # `clip_grad_norm` helps prevent the exploding gradient problem
                # in RNNs / LSTMs.
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                # for p in self.model.parameters():
                #     p.data.add_(-self.learning_rate, p.grad.data)

                if (epoch % self.print_every) == 0:
                    if self.curriculum_learning:
                        max_seq_len_str = 'max_seq_len={}'.format(max_seq_len)
                    else:
                        max_seq_len_str = ''
                    train_loss = (t_loss + eps) / (len(dataloader) + eps)
                    logger.info(' epoch {} train_loss = {:.6f} '
                                'epoch time={:.2f}s lr: {:.6f} {}'.format(
                        epoch, train_loss,
                        time.time() - epoch_start_time,
                        optimizer.param_groups[0]['lr'] if not (
                            self.baseline_type.startswith('hmm') \
                                or 'transformer' in self.args.rnn_type
                        ) else 0,
                        max_seq_len_str)
                    )
                    if self.da:
                        da_loss = (t_da_loss + eps) / (len(dataloader) + eps)
                        logger.info(" da_loss {:.6f}".format(da_loss))
                        self.web_logger.log_metric(
                            "train_domain_loss", da_loss, step=epoch
                        )

                    self.web_logger.log_metric(
                        "train_loss", train_loss, step=epoch
                    )



            run_train()

            # Make sure deallocation has taken place
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # domain adaptation: split target and source domains
            if self.da_inst_loss != {} and epoch % self.da_change_every == 0:
                
                # report performance of each domain
                def compute_avg_and_report(domain_set):
                    list_of_list = [
                        self.da_inst_loss[k] for k in self.da_domain_dic[domain_set]
                    ]
                    domain_auprc = np.array([
                        item for sublist in list_of_list for item in sublist
                    ]).mean()
                    self.web_logger.log_metric(
                        f"auprc_{domain_set}", domain_auprc, step=epoch
                    )

                compute_avg_and_report('source')
                compute_avg_and_report('target')

                def split_source_target():
                    
                    inst_loss = [(hash_id, np.array(loss_value).mean()) \
                        for hash_id, loss_value in self.da_inst_loss.items()]

                    # smaller numbers (better performing ones) comes first
                    inst_loss = sorted(inst_loss, key=lambda x: x[1])

                    inst_loss_nums = [x[1] for x in inst_loss]

                    percidx = inst_loss_nums.index(np.percentile(
                        inst_loss_nums, 100*self.da_percentile, interpolation='nearest'))

                    self.da_domain_dic['source'] = [x[0] for x in inst_loss[:percidx]]
                    self.da_domain_dic['target'] = [x[0] for x in inst_loss[percidx:]]
                    
                    self.da_domain_dic_inv.clear()
                    for hash_id in self.da_domain_dic['source']:
                        self.da_domain_dic_inv[hash_id] = 'source'
                    for hash_id in self.da_domain_dic['target']:
                        self.da_domain_dic_inv[hash_id] = 'target'

                split_source_target()

            if epoch % self.valid_every == 0:

                logger.info('\nResults on training data')
                tr_loss, tr_time_loss, tr_events_loss, tr_auroc, _ = \
                    self.infer_model(train_data, cur_epoch=epoch,
                                        test_name='train', return_metrics=True)

                if valid_data is not None:
                    # Do not test when it is the final one,
                    # since it will be called outside

                    logger.info('Results on valid data')
                    valid_loss, val_time_loss, val_events_loss, val_auroc, _ = \
                        self.infer_model(valid_data, cur_epoch=epoch,
                                        test_name='valid', return_metrics=True)
                    
                    self.web_logger.log_metric(
                        "valid_loss", valid_loss, step=epoch
                    )

                    target_metric = val_auroc
                    target_loss = valid_loss
                else:
                    target_metric = tr_auroc
                    target_loss = tr_loss

                logger.info('total time={:.2f}m '.format(
                    (time.time() - start_time)/60))
    
                msg = ''   

                if ((target_metric > best_metric or self.force_checkpointing) 
                        and self.model is not None) \
                        or self.baseline_type in ['sk_timing_linear_reg']:
                    best_metric = target_metric
                    self.best_epoch = epoch
                    self.renew_token_fn()
                    os.system('rm -rf {}epoch*.model'.format(self.model_prefix))
                    checkpoint_name = '{}epoch_{}.model'.format(
                        self.model_prefix, self.best_epoch)
                    self.save(checkpoint_name)

                    patient_cnt = 0
                    stegnant_auroc = False

                else:
                    if self.patient and (patient_cnt > self.patient):
                        logger.info(
                            'stop training after {} '
                            'patient epochs (at {})'.format(
                                patient_cnt, epoch)
                        )
                        break

                    stegnant_auroc = True


                if (best_loss > target_loss) and (valid_data is None):
                    # NOTE: for hyper-parameter tuning run, do not check steganat_loss
                    # It is only activate for the final test run
                    best_loss = target_loss
                    patient_cnt = 0
                    stegnant_loss = False
                else:
                    stegnant_loss = True
                
                min_epoch_bar = True
                if valid_data is None and self.epoch > epoch:
                    min_epoch_bar = False

                if valid_data is None:
                    # NOTE: when no valid data, stegnant check based on 
                    # train set will be voided
                    loss_diff = abs(prev_loss - target_loss)

                    if (self.loss_tol != -1) and (loss_diff < self.loss_tol) \
                            and min_epoch_bar:
                        logger.info(
                            'loss diff ({:.6f}) is less than tol ({:.6f}). '
                            'Finish train.'.format(loss_diff, self.loss_tol))

                        break

                    prev_loss = target_loss

                    stegnant_loss = True  # default value for valid_data is None

                if self.args.n_warmup_steps <= epoch:
                    check_warmup_over = True
                else:
                    check_warmup_over = False

                if stegnant_loss and stegnant_auroc and min_epoch_bar and check_warmup_over:
                    patient_cnt += 1
                    msg += '| ptn: {}/{} '.format(patient_cnt, self.patient)

                if self.lr_scheduler_ror:
                    scheduler.step(target_loss)

                logger.info(
                    'target loss={:.6f} | best loss={:.6f} '
                    'auroc={:.2f} (epoch {}) {}'.format(
                        target_loss, best_loss, best_metric, self.best_epoch, msg))

            if (self.lr_scheduler and self.model != None
                    and not self.lr_scheduler_ror
                    and not self.baseline_type.startswith('hmm')):
                scheduler.step()
                if self.lr_scheduler_multistep:
                    if epoch in self.lr_scheduler_epochs:
                        logger.info('Current learning rate: {:g}'.format(
                            self.learning_rate * self.lr_scheduler_mult ** (
                                    self.lr_scheduler_epochs.index(
                                        epoch) + 1)))
                else:
                    if epoch % self.lr_scheduler_numiter == 0:
                        logger.info('Current learning rate: {:g}'.format(
                            self.learning_rate * self.lr_scheduler_mult ** (
                                    epoch // self.lr_scheduler_numiter)))

            if self.save_every > 0 and epoch % self.save_every == 0 \
                    and self.model is not None:
                checkpoint_name = '{}epoch_{}.model'.format(
                    self.model_prefix, epoch)
                with Timing(f'Saving checkpoint to {checkpoint_name}...', logger=logger):
                    self.save(checkpoint_name)

            if self.track_weight_change and (self.model is not None):
                raise NotImplementedError

            if epoch >= self.epoch and self.force_epoch:
                break

            if self.args.ptn_event_cnt:
                break

            epoch += 1

        logger.info(
            'train done. best AUROC:{:.2f} epoch:{}'.format(best_metric, self.best_epoch)
        )

        if self.args.ptn_event_cnt:
            import csv
            with open('patient_event_counter.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                for k,v in c.items():
                    k = k+1
                    label = self.event_dic[k]["category"] + '--' + self.event_dic[k]["label"]
                    logger.info("{}: {}".format(label, v))
                    spamwriter.writerow([label, v])


        if (self.model is not None and hasattr(self.model, 'parameters')
                and do_checkpoint):
            checkpoint_name = '{}epoch_{}.model'.format(
                self.model_prefix, self.best_epoch)
            logger.info(
                'Load best validation-f1 model (at epoch {}): {}...'.format(
                    self.best_epoch, checkpoint_name))
            self.load(checkpoint_name)
            if self.use_cuda:
                self.to(self.device)
        return best_metric

    def forward_and_loss(
        self, inp_seq, trg_seq, len_inp_step, hidden, trg_time_seq, inp_time_seq,
        da_domain_trg=None,
    ):
        
        if self.self_correct:
            # NOTE: forward_and_loss() is only called for training and 
            #       prediction will not be used here. Only loss matter.
            loss, pred = self.model(
                inp_seq, len_inp_step, trg_seq, hidden, 
                run_mode=self.args.correct_mode
            )

        elif self.subgroup_adaptation:

            loss, pred_seq = \
                self.model(inp_seq, len_inp_step, trg_seq, mode='train')

            if self.adapt_on_error:
                # we need to compute loss here
                loss = self.loss_fn(pred_seq, trg_seq.float(),
                                    len_inp_step,
                                    self.use_bce_logit,
                                    self.use_bce_stable,
                                    pos_weight=self.pos_weight,
                                    event_weight=self.event_weights
                                    )
            
        else:

            plain_out, time_output, hidden = \
                self.model(inp_seq, len_inp_step, hidden,
                        trg_times=trg_time_seq,
                        inp_times=inp_time_seq,
                        return_hidden_seq=True,
                )

            if self.target_type == 'multi' and not self.use_bce_logit:
                pred_seq = sigmoid(plain_out)
                # assert pred_seq > 0 and pred_seq < 1
                # pred_seq = torch.clamp(pred_seq, min=0, max=1)

            else:
                pred_seq = plain_out


            if pred_seq.dim() == 4 and pred_seq.size(2) == 1:
                pred_seq = pred_seq.squeeze(2)

            if self.model.rnn_type in ['MyLSTM', 'LSTM']:
                if pred_seq.numel() == trg_seq.numel():
                    pred_seq = pred_seq.view(trg_seq.size())

            if self.target_type == 'multi':
                # TODO: weighted loss, make another function to comptued weigthed-event loss
                loss = self.loss_fn(pred_seq, trg_seq.float(),
                                    len_inp_step,
                                    self.use_bce_logit,
                                    self.use_bce_stable,
                                    pos_weight=self.pos_weight,
                                    event_weight=self.event_weights
                                    )
            elif self.target_type == 'single':
                pred_seq = pred_seq.flatten(0, 1)
                trg_seq = trg_seq.flatten(0, 1)
                loss = self.loss_fn(pred_seq, trg_seq)
            else:
                raise NotImplementedError

            if self.pred_time:
                mae, rmse = masked_unroll_loss(
                    self.loss_time_fn, time_output, trg_time_seq,
                    len_inp_step, mask_neg=True)
                loss_time = rmse * self.scale
                # loss_time = torch.log(torch.sqrt(loss_time))
                # loss_time = loss_time / len_bptt
                loss = loss + loss_time

        da_loss = None

        if self.da and da_domain_trg != None:
            if self.da_input == 'hidden':
                da_domain_p_input = hidden
            elif self.da_input == 'proj':
                da_domain_p_input = pred_seq
            else:
                raise NotImplementedError

            da_domain_pred = self.model.da_domain_pred_layer(da_domain_p_input)

            try:
                da_loss = self.da_f_domain_loss(da_domain_pred, da_domain_trg)
            except ValueError as e:
                print(f"inp_seq :{inp_seq.shape}")
                print(f"trg_seq :{trg_seq.shape}")
                print(f"da_domain_p_input :{da_domain_p_input.shape}")
                print(f"da_domain_pred :{da_domain_pred.shape}")
                print(f"da_domain_trg :{da_domain_trg.shape}")
                raise e

            # da_loss = self.revgrad(da_loss)
            # if self.da_opposite_loss:
            #     da_loss = da_loss * -1

            loss = loss + self.da_lambda * da_loss

        return hidden, loss, pred_seq, da_loss


    def infer_model(
        self, test_data, cur_epoch=None, test_name=None,
        final=False,
        export_csv=False, csv_file=None,
        eval_multithread=False, return_metrics=False,
        no_other_metrics_but_flat=False,
        sg_loss_func=F.binary_cross_entropy,
        seq_err_pooling='mean',
    ):
        instance_errors = {}

        start_test = time.time()
        if self.model != None and hasattr(self.model, 'parameters'):
            self.model.eval()

        # load prior for periodicity prior
        if self.pp_type == 'adaptive' and self.model.pp.prior is None:
            self.model.pp.load_prior(d_path=self.d_path, prep_loader=False,
                                     force_reset=self.skip_save_prior,
                                     not_save=self.skip_save_prior, 
                                     prior_from_mimic=self.args.prior_from_mimic,
                                     vecidx2mimic=self.args.vecidx2mimic)

        dataloader = get_dataloader(self, test_data, shuffle=False)

        t_loss, loss_times, loss_events = 0, 0, 0

        if self.target_type == 'multi':
            eval = MultiLabelEval(self.event_size, 
                                  use_cuda=self.use_cuda,
                                  macro_aucs=True,
                                  micro_aucs=False,
                                  device=self.device, 
                                  event_dic=self.event_dic,
                                  event_types=self.event_types,
                                  pred_labs=self.args.pred_labs,
                                  pred_normal_labchart=self.args.pred_normal_labchart,)

        elif self.target_type == 'single':
            eval = MultiClassEval(self.event_size, use_cuda=self.use_cuda,
                                  device=self.device,
                                  ks=[1, 2, 5, 10, 20, 100])
        else:
            raise NotImplementedError

        if self.baseline_type in ['sk_timing_linear_reg']:
            test_x, test_y = [], []

        self.write_cnt_steps = 0

        # if self.adapt_mem and (test_name=='ltam_write' or final):
        #     # freeze weights
        #     self.model.embed_input.weight.requires_grad = False 

        #     # step_model.fc_out.weight.requires_grad = False 
        #     # step_model.fc_out.bias.requires_grad = False 
        #     for param in self.model.fc_out.parameters():
        #         param.requires_grad = False

        #     # !! Wrong Way: step_model.rnn.requires_grad = False
        #     for param in self.model.rnn.parameters():
        #         param.requires_grad = False

        if self.log_hidden_target_error_file:
            self.hidden_target_error = {
                'hidden':[], 'target':[], 'error':[], 
                'hidden_seq': {}, 'target_seq': {}, 'error_seq': {}}
            self.bce_loss = torch.nn.BCELoss(reduction='none')
        else:
            self.hidden_target_error = None

        for i, data in enumerate(dataloader):

            if self.early_terminate_inference and (i > 0):
                break

            inp, trg, len_inp, len_trg, inp_time, trg_time, hadm_ids \
                = process_batch(self, data, 0)

            # with torch.no_grad():
            self.model.eval()
            loss_cnt = 0

            if self.baseline_type == 'copylast':

                if self.target_type == 'multi':
                    pred = (inp > 0).float()
                    eval.update(pred=pred, trg=trg, len_trg=len_trg,
                                final=final)
                elif self.target_type == 'single':
                    pred = inp
                    raise NotImplementedError


            elif self.baseline_type == 'majority_ts':

                eff_inp = []
                for inp_step in range(inp.size(1)):
                    eff_inp.append(inp[:, :(inp_step + 1), :].sum(1))
                ts_counter = torch.stack(eff_inp, dim=1)
                if self.use_cuda:
                    ts_counter = ts_counter.to(self.device)

                pred = topk_pred(ts_counter, inp.size(),
                                    threshold=self.topk_threshold)
                if self.use_cuda:
                    pred = pred.to(self.device)

                eval.update(pred=pred, trg=trg, len_trg=len_trg,
                            final=final)

            elif self.baseline_type == 'majority_all':
                n_b, n_s = inp.size()[:2]
                event_counter = torch.stack(
                    [self.event_counter] * n_b * n_s,
                    dim=0).view(n_b, n_s, -1)

                pred = topk_pred(event_counter, inp.size(),
                                    threshold=self.topk_threshold)
                if self.use_cuda:
                    pred = pred.to(self.device)

                eval.update(pred=pred, trg=trg, len_trg=len_trg,
                            final=final)

            elif self.baseline_type == 'random':
                pred = (torch.rand(trg.size()) > 0.5).float()
                eval.update(pred=pred, trg=trg, len_trg=len_trg,
                            final=final)

            elif self.baseline_type in ['sk_timing_linear_reg']:
                eff_inp = []

                for inp_step in range(inp.size(1)):
                    eff_inp.append(inp[:, :(inp_step + 1), :].sum(1))
                eff_inp = torch.stack(eff_inp, dim=1)
                test_x.append(eff_inp)
                test_y.append(trg_time.cpu())

            elif self.baseline_type in ['logistic_binary',
                                        'logistic_count',
                                        'logistic_last',
                                        'logistic_last_mlp',
                                        'logistic_binary_mlp',
                                        'logistic_count_mlp',
                                        'timing_linear_reg']:
                eff_inp = []

                if self.baseline_type in ['logistic_binary',
                                            'logistic_count',
                                            'logistic_binary_mlp',
                                            'logistic_count_mlp',
                                            'timing_linear_reg']:
                    for inp_step in range(inp.size(1)):
                        eff_inp.append(inp[:, :(inp_step + 1), :].sum(1))

                    eff_inp = torch.stack(eff_inp, dim=1)

                    if self.use_cuda:
                        eff_inp = eff_inp.to(self.device)

                    if self.baseline_type in ['logistic_binary',
                                                'logistic_binary_mlp']:
                        eff_inp = (eff_inp > 0).float()

                elif self.baseline_type in ['logistic_last',
                                            'logistic_last_mlp']:
                    eff_inp = inp.float()

                self.model.zero_grad()
                output = self.model(eff_inp.view(-1, eff_inp.size(2)))

                if self.baseline_type in ['logistic_binary',
                                            'logistic_count',
                                            'logistic_last',
                                            'logistic_last_mlp',
                                            'logistic_binary_mlp',
                                            'logistic_count_mlp',
                                            ]:

                    output = output.view((eff_inp.size(0), eff_inp.size(1), -1))

                    pred_seq = sigmoid(output)

                    # single output
                    if self.target_event > -1:
                        trg = trg[:, :, self.target_event].unsqueeze(-1)

                        if pred_seq.dim() == 2:
                            pred_seq = pred_seq.unsqueeze(-1)

                    pred = (pred_seq > 0.5).float()

                    inp_first_step = eff_inp[:, 0, :].unsqueeze(1)
                    if self.args.pred_labs or self.args.pred_normal_labchart:
                        inp_first_step = fit_input_to_output(inp_first_step, 
                            self.args.inv_id_mapping.keys())

                    eval.update(pred=pred, trg=trg, len_trg=len_trg,
                                final=final,
                                prob=pred_seq,
                                skip_detail_eval=(self.target_event > -1),
                                force_auroc=self.force_auroc,
                                inp_first_step=inp_first_step)

                    loss = self.loss_fn(pred_seq, trg, len_inp)

                elif self.baseline_type in ['timing_linear_reg']:
                    pred_seq = relu(output.view(eff_inp.size()))
                    mae, rmse = masked_unroll_loss(self.loss_time_fn,
                                                    pred_seq, trg_time,
                                                    len_inp, mask_neg=True)
                    loss = rmse
                    loss_times += loss

                t_loss += loss.item()
                loss_cnt += 1

            elif self.baseline_type == 'timing_mean':
                pred = self.timing_vec.expand(trg_time.size())
                mae, rmse = masked_unroll_loss(self.loss_time_fn, pred,
                                                trg_time, len_inp,
                                                mask_neg=True)
                loss_times += rmse

            elif self.baseline_type == 'timing_random':
                pred = relu(torch.rand(trg.size()))
                mae, rmse = masked_unroll_loss(self.loss_time_fn, pred,
                                                trg_time, len_inp,
                                                mask_neg=True)
                loss_times += rmse

            elif self.baseline_type.startswith('hmm'):
                # combine x and y

                seq = torch.cat((inp, trg[:, -1, :].unsqueeze(1)), dim=1)
                seq_len = len_inp + 1

                # logger.info('seq size:{}'.format(seq.size()))

                get_preds = self.model['get_pred']
                guide = self.model['guide']
                # pred_model = self.model['pred']
                args = self.model['args']
                elbo = self.model['elbo']
                model = self.model['model']
                preds, probs = get_preds(guide, model, seq, seq_len, args)
                loss = elbo.loss(model, guide, seq, seq_len, args,
                                    include_prior=False)
                num_observations = float(seq_len.sum())
                t_loss += loss / num_observations

                preds = preds[:, 1:, :]
                probs = probs[:, 1:, :]

                # nan detect
                assert (preds != preds).sum() == 0, "NaN!"
                assert (probs != probs).sum() == 0, "NaN!"

                inp_first_step = inp[:, 0, :].unsqueeze(1)
                if self.args.pred_labs or self.args.pred_normal_labchart:
                    inp_first_step = fit_input_to_output(inp_first_step, 
                        self.args.inv_id_mapping.keys())

                eval.update(pred=preds, trg=trg, len_trg=len_trg,
                            final=final,
                            prob=probs,
                            skip_detail_eval=(self.target_event > -1),
                            force_auroc=self.force_auroc,
                            inp_first_step=inp_first_step)

            elif self.baseline_type == 'None':
                """
                RNN Models
                """
                if type(inp) == torch.Tensor:
                    batch_size, seq_len, _ = inp.size()
                else:
                    batch_size = len(inp)
                    seq_len = len(inp[0])

                if ((
                        self.adapt_lstm or self.adapt_lstm_only \
                        or self.adapt_fc_only or self.adapt_residual \
                            or self.neural_caching
                    ) and final):
                    read_batch_size = 1
                else:
                    read_batch_size = batch_size
                
                bptt_time_loss = 0

                bptt_size = self.bptt if self.bptt else seq_len
                        
                if type(len_inp) == list:
                    len_inp = torch.tensor(len_inp)
                if type(len_trg) == list:
                    len_trg = torch.tensor(len_trg)

                for b in range(0, batch_size, read_batch_size):

                    if self.early_terminate_inference and (b > 1):
                        break

                    if self.args.fp16 and _use_native_amp:
                        with autocast():
                            loss_events, loss_times, inp_first_step, loss_cnt, t_loss, \
                                loss_times, bptt_time_loss, instance_errors = self.process_inference(
                                    b, inp, trg, inp_time, trg_time, 
                                    len_inp, len_trg, read_batch_size,
                                    seq_len, bptt_size, final, test_name, eval, 
                                    loss_events, loss_cnt, t_loss, loss_times,
                                    bptt_time_loss, batch_size, i, instance_errors,
                                    hadm_ids,
                                    sg_loss_func, seq_err_pooling, 
                            )

                    else:
                        loss_events, loss_times, inp_first_step, loss_cnt, t_loss, \
                            loss_times, bptt_time_loss, instance_errors = self.process_inference(
                                b, inp, trg, inp_time, trg_time, 
                                len_inp, len_trg, read_batch_size,
                                seq_len, bptt_size, final, test_name, eval, 
                                loss_events, loss_cnt, t_loss, loss_times,
                                bptt_time_loss, batch_size, i, instance_errors, 
                                hadm_ids,
                                sg_loss_func, seq_err_pooling, 
                        )

            else:
                raise NotImplementedError

            if self.model is not None and hasattr(self.model, 'parameters') \
                    and loss_cnt > 0:
                t_loss /= loss_cnt

        def log_hidden_target_error():
            path = f"{self.args.model_prefix}_{test_name}_hidden_target_error_file_export"
            os.system(f"mkdir -p {path}")
            logger.debug(f"log mem contents into files in {path}")

            if self.hidden_target_error['hidden'] != []:
                hidden_stack = torch.stack(self.hidden_target_error['hidden'])

                with open(f"{path}/hidden.npy", 'wb') as f:
                    np.save(f, hidden_stack.cpu().detach().numpy())

                torch.save(
                    self.hidden_target_error['hidden_seq'], 
                    f"{path}/hidden_seq.pt"
                )

            if self.hidden_target_error['target'] != []:
                target_stack = torch.stack(self.hidden_target_error['target'])

                with open(f"{path}/target.npy", 'wb') as f:
                    np.save(f, target_stack.cpu().detach().numpy())

                torch.save(
                    self.hidden_target_error['target_seq'], 
                    f"{path}/target_seq.pt"
                )

            if self.hidden_target_error['error'] != []:
                error_stack = torch.stack(self.hidden_target_error['error'])

                with open(f"{path}/error.npy", 'wb') as f:
                    np.save(f, error_stack.cpu().detach().numpy())

                torch.save(
                    self.hidden_target_error['error_seq'], 
                    f"{path}/error_seq.pt"
                )

        if self.train_learn_to_use_mem \
                and test_name in ['final_test', 'final_train','ltam_read', 'ltam_write']:
            logger.info(f"write step count: {self.ltam_model.write_cnt_steps}")
            self.web_logger.log_parameter(
                'write_step_count', self.ltam_model.write_cnt_steps)
            logger.info(f"permem mem average error: {self.ltam_model.memnet.errors.mean()}")
            self.web_logger.log_parameter(
                'permem_slots_error_avg', self.ltam_model.memnet.errors.mean().item())

            if sum(self.ltam_model.memnet.errors.size()) > 1:
                logger.info(f"permem mem median error: {self.ltam_model.memnet.errors.median()}")
                self.web_logger.log_parameter(
                    'permem_slots_error_median', self.ltam_model.memnet.errors.median().item())

            if self.args.log_mem_file:
                self.ltam_model.memnet.log_mem_to_file(f"{self.args.model_prefix}_mem_file_export")
                
            if self.log_hidden_target_error_file:
                log_hidden_target_error()
        
        if self.log_hidden_target_error_file \
                and test_name in ['final_test', 'final_train']:
            log_hidden_target_error()

        if self.baseline_type in ['sk_timing_linear_reg']:
            test_x = flat_tensor(test_x).numpy()
            test_y = flat_tensor(test_y).numpy()

            pred_y = self.model.predict(test_x)

            mae, rmse = masked_unroll_loss(self.loss_time_fn,
                                           torch.Tensor(pred_y).unsqueeze(0),
                                           torch.Tensor(test_y).unsqueeze(0),
                                           [pred_y.shape[0]], mask_neg=True)
            loss_times = rmse * len(dataloader)

        mse_loss_avg = (loss_times) / (len(dataloader))
        event_loss_avg = (loss_events) / (len(dataloader))
        self.renew_token_fn()

        if self.baseline_type.startswith('timing_'):
            logger.info('Time Pred Loss: {:5.4f} Event Pred Loss: {:5.4f}'.format(
                mse_loss_avg, event_loss_avg))

            self.web_logger.log_metric("{}: {}".format(
                test_name, 'time_loss'), mse_loss_avg, step=cur_epoch)
            self.web_logger.log_metric("{}: {}".format(
                test_name, 'event_loss'), event_loss_avg, step=cur_epoch)

        if not self.baseline_type.startswith('timing_'):
            
            # event prediction metrics

            if final and self.target_event > -1:
                eval.eval['flat']['y_probs'][self.target_event] = \
                    eval.eval['flat']['y_probs'][0]
                eval.eval['flat']['y_trues'][self.target_event] = \
                    eval.eval['flat']['y_trues'][0]

            if self.target_type == 'single':
                f1, acc = eval.compute(epoch=cur_epoch,
                                       test_name=test_name,
                                       web_logger=self.web_logger,
                                       final=final, verbose=True,
                                       )

            elif self.target_type == 'multi':
                output = eval.compute(eval.eval['flat'], epoch=cur_epoch,
                                      test_name=test_name,
                                      web_logger=self.web_logger,
                                      final=final, export_csv=(export_csv and final),
                                      event_dic=self.event_dic,
                                      csv_file=csv_file,
                                      use_multithread=eval_multithread,
                                      return_metrics=return_metrics,
                                      force_auroc=(self.force_auroc and (
                                          test_name != 'train')),
                                      force_plot_auroc=self.force_plot_auroc,
                                      # self.args.target_auprc and
                                      target_auprc=((test_name != 'train'))
                                      )
                if csv_file is not None: 
                    np.save(csv_file.replace('.csv', '_event_dic.npy'), self.event_dic)

                if return_metrics:
                    (f1, acc, metrics_container) = output

                    if self.args.target_auprc:
                        output_metric = metrics_container["mac_auprc"]
                    else:
                        output_metric = metrics_container["mac_auroc"]

                else:
                    (f1, acc) = output
                    output_metric = f1

                if final and (not no_other_metrics_but_flat):
                    run_evals(self, csv_file, eval, test_name, final,
                              export_csv, eval_multithread, cur_epoch)
        else:
            acc = f1 = 1.0 / mse_loss_avg

        end_test = time.time()
        logger.info(
            'Evaluation done in {:.3f}s\n'.format(end_test - start_test))

        if self.model is not None and hasattr(self.model, 'parameters'):
            self.model.train()  # turn off eval mode

        # adaptive switch counter save
        if csv_file is not None: 
            # switch count
            if sum(self.adapt_switch_stat.values()) != 0:
                self.adapt_switch_stat["ratio_instance_model"] \
                    = self.adapt_switch_stat["inst"] / sum(self.adapt_switch_stat.values())
            else:
                self.adapt_switch_stat["ratio_instance_model"] = 0

            with open(csv_file.replace(".csv", "_adapt_switch_stat.txt"), 'w') as f:
                print(self.adapt_switch_stat, file=f)

            # switch count by steps
            df_switch = pd.DataFrame.from_dict(self.adapt_switch_stat_steps, orient='index')
            if self.adapt_switch_stat_steps != {}:
                df_switch["ratio_instance_model"] \
                    = df_switch.iloc[:, 1] / (df_switch.iloc[:, 0] + df_switch.iloc[:, 1])
            df_switch.to_csv(csv_file.replace(".csv", "_adapt_switch_stat_steps.csv"))

            # patient count by steps
            df_ptn = pd.DataFrame.from_dict(self.patient_count_steps, orient='index')
            df_ptn.to_csv(csv_file.replace(".csv", "_patient_count_steps.csv"))

        # Adaptive learning update counter 
        if self.cnt_update > 0:
            logger.info('adaptive learning: count update: {}'.format(self.cnt_update))
            logger.info('adaptive learning: total time-steps: {}'.format(self.cnt_time_step))
            logger.info('average # updates/time-step: {:.4f}'.format(self.cnt_update/self.cnt_time_step))
            self.web_logger.log_parameter('cnt_update', self.cnt_update)
            self.web_logger.log_parameter('cnt_time_step', self.cnt_time_step)
            self.web_logger.log_parameter('cnt_avg_update_per_timestep', self.cnt_update/self.cnt_time_step)

        return (t_loss + eps) / (len(dataloader) + eps), \
               (loss_times + eps) / (len(dataloader) + eps), \
               (loss_events + eps) / (len(dataloader) + eps), \
            output_metric, instance_errors

    def save(self, path):
        self.renew_token_fn()
        if self.model is not None:
            if hasattr(self.model, 'parameters'):
                torch.save(self.model.state_dict(), path)
            elif self.baseline_type.startswith('hmm'):
                # pickle.dump(self.model, open(path, 'wb'))

                # pyro.get_param_store().save(path)
                pass
            #     output = {
            #         'get_pred': self.model['get_pred'],
            #         'guide': self.model['guide'],
            #         'args':self.model['args'],
            #         'elbo':self.model['elbo'],
            #         'state_dict': self.model['model'].state_dict(),
            #     }
            #     torch.save(output, path)

            else:
                pickle.dump(self.model, open(path, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        self.renew_token_fn()
        if self.model is not None:
            if hasattr(self.model, 'parameters'):
                if self.use_cuda:
                    self.model.cpu().load_state_dict(torch.load(path))
                else:
                    self.model.cpu().load_state_dict(
                        torch.load(path, map_location='cpu'))
            elif self.baseline_type.startswith('hmm'):
                pass
                # pyro.get_param_store().load(path)    
            #     payload = torch.load(path)

            #     self.model['get_pred'] = payload['get_pred']
            #     self.model['guide'] = payload['guide']
            #     self.model['args'] = payload['args']
            #     self.model['elbo'] = payload['elbo']
            #     self.model['model'].load_state_dict(payload['model'])

            else:
                self.model = pickle.load(open(path, 'rb'))

    def load_best_epoch_model(self):
        self.web_logger.log_parameter('best_epoch', self.best_epoch)

        checkpoint_name = '{}epoch_{}.model'.format(self.model_prefix,
                                                    self.best_epoch)
        logger.info(
            'Load best validation-auroc model (at epoch {}): {}...'.format(
                self.best_epoch, checkpoint_name))
        self.load(checkpoint_name)
        if self.use_cuda and not self.baseline_type.startswith('hmm'):
            self.model = self.model.to(self.device)

    def save_final_model(self):
        final_model_name = '{}_final.model'.format(self.model_prefix)
        with Timing('Saving final model to {final_model_name} ...', logger=logger):
            if self.model is not None:
                if hasattr(self.model, 'parameters'):
                    torch.save(self.model.state_dict(), final_model_name)
                else:
                    pickle.dump(self.model, open(final_model_name, 'wb'),
                                protocol=pickle.HIGHEST_PROTOCOL)
        self.renew_token_fn()
        os.system('rm -rf {}epoch*.model'.format(self.model_prefix))


    def process_inference(self, b, inp, trg, inp_time, trg_time, len_inp, len_trg, 
                          read_batch_size, seq_len, bptt_size, final, test_name, 
                          eval, loss_events, loss_cnt, t_loss, loss_times, 
                          bptt_time_loss, batch_size, batch_idx, instance_errors,
                          hadm_ids, 
                          sg_loss_func, seq_err_pooling='mean'):
        # extract small read batch from mini batch
        inp_b = inp[b: b + read_batch_size]
        trg_b = trg[b: b + read_batch_size]
        if inp_time is not None:
            inp_time_b = inp_time[b: b + read_batch_size]
        if trg_time is not None:
            trg_time_b = trg_time[b: b + read_batch_size]

        len_inp_b = len_inp[b: b + read_batch_size]
        len_trg_b = len_trg[b: b + read_batch_size]
        
        for j in range(0, seq_len, bptt_size):

            if self.bptt:
                seqlen = min(self.bptt, seq_len - j)
            else:  # no bptt
                seqlen = seq_len

            if type(inp) == torch.Tensor:
                inp_seq = inp_b[:, j:j + seqlen]
            else:
                inp_seq = [ibatch[j:j + seqlen]
                            for ibatch in inp_b]

            if type(trg) == torch.Tensor:
                trg_seq = trg_b[:, j:j + seqlen]
            else:
                trg_seq = [ibatch[j:j + seqlen]
                            for ibatch in trg_b]

            if trg_time is not None:
                if type(trg_time) == torch.Tensor:
                    trg_time_seq = trg_time_b[:, j:j + seqlen]
                else:
                    trg_time_seq = [ibatch[j:j + seqlen]
                                    for ibatch in trg_time_b]
            else:
                trg_time_seq = None

            if inp_time is not None:
                if type(inp_time) == torch.Tensor:
                    inp_time_seq = inp_time_b[:, j:j + seqlen]
                else:
                    inp_time_seq = [ibatch[j:j + seqlen]
                                    for ibatch in inp_time_b]
            else:
                if type(inp) == torch.Tensor:
                    inp_time_seq = torch.zeros(
                        inp_seq.size()).to(self.device)
                else:
                    inp_time_seq = [[[]]]

            seqlen_v = torch.LongTensor([seqlen] * read_batch_size)

            len_inp_step = torch.min(len_inp_b, seqlen_v)

            len_inp_b -= len_inp_step

            len_trg_step = torch.min(len_trg_b, seqlen_v)
            len_trg_b -= len_trg_step

            # inp_seq: n_batch * max_seq_len * n_events
            # x : max_seq_len * n_events
            # len_seq : n_batch

            # hidden = repackage_hidden(hidden)

            if sum(len_inp_step) < 1:
                continue

            self.model.zero_grad()

            hidden = self.model.init_hidden(
                batch_size=read_batch_size)

            # removing zero-lengths batch elements
            hidden, inp_seq, trg_seq, len_inp_step, len_trg_step, inp_time_seq, trg_time_seq = \
                remove_zeroed_batch_elems(self, hidden, inp_seq,
                                                trg_seq,
                                                len_inp_step,
                                                len_trg_step,
                                                trg_time_seq=trg_time_seq,
                                                inp_time_seq=inp_time_seq)

            if type(hidden) != list:
                hidden = hidden.squeeze(0)

            # if hidden.dim() == 4 and hidden.size(0) == 1:
            #     hidden = hidden.squeeze(0)
            bypass_sigmoid = False

            if self.self_correct:
                loss, sig_out = self.model(
                    inp_seq, len_inp_step, trg_seq, hidden, 
                    run_mode=self.args.correct_mode
                )

            elif self.moe:
                sig_out, gating_score = self.model(inp_seq, len_inp_step, trg_seq)

                loss = masked_bce_loss(sig_out, trg_seq, len_trg_step)


            elif self.subgroup_adaptation:
                _, sig_out = \
                    self.model(inp_seq, len_inp_step, trg_seq, mode='test')
                
                sig_out = torch.clamp(sig_out, 0, 1)

                loss = F.binary_cross_entropy(sig_out, trg_seq)

            else:
                if ((self.adapt_lstm or self.adapt_lstm_only \
                        or self.adapt_fc_only or self.adapt_residual
                    ) and final
                ):

                    plain_out, time_pred = train_adaptive_model(self, 
                        inp_seq, trg_seq, inp_time_seq, trg_time_seq,
                        len_inp_step, self.model, b, batch_size, batch_idx=batch_idx)
                    
                elif (self.neural_caching and final):
                    plain_out, time_pred, hidden = neural_caching_model(self, 
                        self.ncache_window, self.ncache_theta, self.ncache_lambdah,
                        inp_seq, trg_seq, inp_time_seq, trg_time_seq,
                        len_inp_step, self.model, b, batch_size, batch_idx=batch_idx)

                elif (self.adapt_mem and (test_name in ['ltam_write', 'ltam_read'] or final)):

                    bypass_sigmoid = True

                    plain_out, time_pred, hidden = self.ltam_model(
                        test_name, inp_seq, trg_seq, inp_time_seq, trg_time_seq,
                        len_inp_step, self.model, batch_size, batch_idx=batch_idx)

                else:
                    plain_out, time_pred, hidden = self.model(inp_seq,
                                                        len_inp_step,
                                                        hidden,
                                                        trg_times=trg_time_seq,
                                                        inp_times=inp_time_seq,
                                                        return_hidden_seq=True
                                                        )

                if plain_out.size(1) != trg_seq.size(1):
                    pad_len = trg_seq.size(1) - plain_out.size(1)
                    z_pad = torch.zeros(plain_out.size(0), pad_len, plain_out.size(2))
                    plain_out = torch.cat((plain_out, z_pad), dim=1)

                if plain_out.size(0) == inp_seq.size(1) and plain_out.size(1) == inp_seq.size(0):
                    plain_out = plain_out.transpose(0, 1)

                if self.target_type == 'multi' and not bypass_sigmoid:
                    sig_out = sigmoid(plain_out)
                else:
                    sig_out = plain_out

                # if len(sig_out.size()) > 3:
                #     sig_out = sig_out.squeeze()
                if sig_out.dim() == 4 and sig_out.size(2) == 1:
                    sig_out = sig_out.squeeze(2)

                if self.model.rnn_type in ['MyLSTM', 'LSTM']:
                    if sig_out.numel() == trg_seq.numel():
                        sig_out = sig_out.view(trg_seq.size())

                if self.target_type == 'multi':

                    if self.use_bce_logit:
                        loss = self.loss_fn(plain_out, trg_seq.float(),
                                            len_inp_step,
                                            self.use_bce_logit,
                                            event_weight=self.event_weights
                                            )
                    else:
                        # sig_out = torch.clamp(sig_out, min=0, max=1)
                        loss = self.loss_fn(sig_out, trg_seq.float(),
                                            len_inp_step,
                                            use_stable=self.use_bce_stable)

                elif self.target_type == 'single':
                    # TODO: FIX contiguous
                    sig_out = sig_out.flatten(0, 1)
                    trg_seq = trg_seq.flatten(0, 1)
                    loss = self.loss_fn(sig_out, trg_seq)
                else:
                    raise NotImplementedError
                
                if test_name == "store_instance_errors":

                    batch_error = sg_loss_func(
                        sig_out, trg_seq.float(), reduction='none'
                    )

                     # pooling over time-steps
                    if seq_err_pooling == 'mean':
                        batch_error = batch_error.mean(1) 
                    elif seq_err_pooling == 'sum':
                        batch_error = batch_error.sum(1) 
                    elif seq_err_pooling == 'max':
                        batch_error = batch_error.max(1)
                    elif seq_err_pooling == 'none':
                        pass
                    else:
                        raise NotImplementedError

                    for inst_inp_seq, inst_error in zip(inp_seq, batch_error):
                        hash_id = get_hash(inst_inp_seq.cpu().numpy())
                        instance_errors[hash_id] \
                            = inst_error.cpu().detach().numpy()

            if self.log_hidden_target_error_file:
                loss_instance = self.bce_loss(sig_out.to(self.device), trg_seq.to(self.device))

                if self.moe:
                    hidden = gating_score

                for b_idx, step_len in enumerate(len_inp_step):
                    for step in range(step_len):
                        if hidden != None:
                            self.hidden_target_error['hidden'].append(
                                hidden[b_idx, step].cpu().detach()) 
                        self.hidden_target_error['target'].append(
                            trg_seq[b_idx, step].cpu().detach()) 
                        self.hidden_target_error['error'].append(
                            loss_instance[b_idx, step].cpu().detach())
                        
                    hadm_id = hadm_ids[b_idx]

                    if hidden != None:
                        self.hidden_target_error['hidden_seq'][hadm_id] = hidden[b_idx].cpu().detach()
                    self.hidden_target_error['target_seq'][hadm_id] = trg_seq[b_idx].cpu().detach()
                    self.hidden_target_error['error_seq'][hadm_id] = loss_instance[b_idx].cpu().detach()

            loss_events += loss.item()
            loss_cnt += 1

            if self.pred_time:
                if time_pred.size() == ():
                    continue

                # time_target = times[:, 1:]
                mae, rmse = masked_unroll_loss(
                    self.loss_time_fn, time_pred, trg_time_seq,
                    len_inp_step, mask_neg=True)
                bptt_time_loss += rmse.item()

                if len(loss.size()) == 0:
                    loss = loss.unsqueeze(0)
                loss += rmse * self.scale

            t_loss += loss.item()
            loss_times += bptt_time_loss

            # ==========
            # evaluation
            # ----------

            if self.target_type == 'multi':
                pred = (sig_out > 0.5).float()
            elif self.target_type == 'single':
                pred = sig_out.argmax(dim=1)
                if str(self.loss_fn) == 'CrossEntropyLoss()':
                    sig_out = F.softmax(sig_out, dim=-1)

            if trg_seq.dim() == 3 and pred.dim() == 2:
                pred = pred.unsqueeze(1)
                sig_out = sig_out.unsqueeze(1)
            elif trg_seq.dim() == 3 and pred.dim() == 1:
                pred = pred.unsqueeze(0).unsqueeze(0)
                sig_out = sig_out.unsqueeze(0).unsqueeze(0)

            if type(inp) == torch.Tensor:
                inp_first_step = inp_seq[:, 0, :].unsqueeze(1)
            else:
                inp_first_step = torch.zeros(
                    read_batch_size, self.event_size).to(self.device).float()
                for b_idx, iseq in enumerate(inp_seq): 
                    for event in iseq[0]:
                        inp_first_step[b_idx, event - 1] = 1
                inp_first_step = inp_first_step.unsqueeze(1)

            if self.args.pred_labs or self.args.pred_normal_labchart:
                inp_first_step = fit_input_to_output(inp_first_step,
                    self.args.inv_id_mapping.keys())

            if debug:
                logger.info('pred: {}'.format(pred.size()))
                logger.info('sig_out: {}'.format(sig_out.size()))
                logger.info('trg_seq: {}'.format(trg_seq.size()))
                logger.info('len_trg_step: {}'.format(len_trg_step.size()))
                logger.info('inp_first_step: {}'.format(
                    inp_first_step.size()))

            if self.args.eval_on_cpu or not torch.cuda.is_available():
                device = torch.device('cpu')
            else:
                device = torch.device('cuda')

            eval.update(pred=pred.to(device), trg=trg_seq.float().to(device),
                        len_trg=len_trg_step,
                        base_step=j, final=final, prob=sig_out.to(device),
                        force_auroc=(self.force_auroc and (
                            test_name != 'train')),
                        inp_first_step=inp_first_step.to(device))
        # loss_times /= loss_cnt
        # loss_events /= loss_cnt

        return (
            loss_events, loss_times, inp_first_step, loss_cnt, t_loss, 
            loss_times, bptt_time_loss, instance_errors
        )


class PoolingLayer(nn.Module):
    def __init__(self, f_pooling, dim):
        super(PoolingLayer, self).__init__()
        self.f_pooling = f_pooling
        self.dim = dim

    def forward(self, x):
        if self.f_pooling == 'mean':
            x = x.mean(self.dim)
        elif self.f_pooling == 'max':
            x, _ = x.max(self.dim)
        elif self.f_pooling == 'sum':
            x = x.sum(self.dim)
        return x


# https://github.com/jvanvugt/pytorch-domain-adaptation

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
