import sys
import copy
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.project_utils import (set_event_mask, grad_info)
from utils.tensor_utils import fit_input_to_output
from .memnet import MemNetE2E, PredErrorModel, LTAMControllerV2, ContextTargetMLP
from .attention import BahdanauAttention

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)

class LTAM(nn.Module):
    def __init__(self, args):
        super(LTAM, self).__init__()

        self.event_size = args.event_size
        self.hidden_dim = args.hidden_dim
        self.target_size = args.target_size
        self.mem_size = args.mem_size
        self.mem_policy = args.mem_policy
        self.mem_is_unbounded = args.mem_is_unbounded
        self.mem_key = args.mem_key
        self.mem_content = args.mem_content
        self.mem_merge = args.mem_merge
        self.use_mem_gpu = args.use_mem_gpu
        self.read_threshold = args.mem_read_error_threshold
        self.similarity_threshold = args.mem_read_similarity_threshold
        self.lambdah = args.ncache_lambdah
        self.read_mode = args.read_mode
        self.verbose = args.verbose
        self.write_cnt_steps = 0
        self.train_learn_to_use_mem = args.train_learn_to_use_mem
        self.use_context_target_stats = args.use_context_target_stats
        self.nn_mlp = args.nn_mlp

        self.log_idf_context = None
        self.prob_context_target = None
        self.gaussian_width = args.gaussian_width
        self.shrink_event_dim = args.shrink_event_dim

        self.pred_labs = args.pred_labs
        self.pred_normal_labchart = args.pred_normal_labchart
        self.use_mapped_key = args.use_mapped_key
        self.inv_id_mapping = args.inv_id_mapping

        # Switch to GPU mode if possible
        if torch.cuda.is_available() and self.use_mem_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if args.adapt_mem:
            
            if self.mem_key == 'hidden':
                key_size = args.hidden_dim 
            elif self.mem_key == 'input' and self.use_mapped_key:
                key_size = args.target_size
            else:
                key_size = args.event_size

            self.memnet = MemNetE2E(
                key_size=key_size,
                value_size=args.target_size,
                num_slots=args.mem_size,
                policy=args.mem_policy,
                is_unbounded=args.mem_is_unbounded,
            ).to(self.device)

        if args.train_error_pred:
            self.pred_error_model = PredErrorModel(
                input_size=args.hidden_dim,
                output_size=args.target_size,
                hidden_size=args.pred_error_hidden_dim,
                act_func=args.pred_error_act_func,
            ).to(self.device)

        if args.train_learn_to_use_mem:
            
            if args.nn_mlp:
                if self.mem_key == 'input':
                    nn_dim = self.event_size
                elif self.mem_key == 'hidden':
                    nn_dim = self.hidden_dim
                nn_dim = nn_dim * int(self.read_mode.lstrip('nn'))
            else:
                nn_dim = args.hidden_dim

            if self.mem_merge not in ['gating_hidden', 'mem_only_gating']:
                nn_dim += args.target_size

            hidden_sizes = list(np.linspace(
                nn_dim,
                args.target_size,
                num=args.ltam_num_layer + 2).astype(int))

            self.read_controller = LTAMControllerV2(
                hidden_sizes=hidden_sizes,
                p_dropout=args.ltam_dropout,
                act_func=args.pred_error_act_func,
            ).to(self.device)

        if self.mem_merge == 'concat':
            self.fc_out_merge = nn.Linear(
                self.target_size * 2, self.target_size).to(self.device)
        

        if self.use_context_target_stats:
            hidden_sizes = list(np.linspace(
                args.event_size + args.target_size,
                args.target_size,
                num=args.ltam_num_layer + 2).astype(int))

            self.context_target_mlp = ContextTargetMLP(
                hidden_sizes=hidden_sizes,
                p_dropout=args.ltam_dropout,
                act_func=args.pred_error_act_func,
            ).to(self.device)

        if self.mem_merge == 'attention':
            self.attn = BahdanauAttention(
                hidden_size=self.target_size,
                key_size=self.target_size,
                query_size=self.target_size,
                ).to(self.device)

            if self.mem_key == 'hidden':
                query_size = self.hidden_dim 
            elif self.mem_key == 'input' and self.use_mapped_key:
                query_size = self.target_size
            else:
                query_size = self.event_size

            self.linear_t = nn.Linear(
                query_size, self.target_size).to(self.device)

    def forward(self, mode, inp_seq, trg_seq, inp_time_seq, trg_time_seq,
                len_inp_step, popl_model, full_batch_size,
                tol=1e-03, d_loss_tol=1e-04, batch_idx=0):
        """
        previous neural_persist_caching_model
        """

        self.memnet = self.memnet.to(self.device)
        if self.train_learn_to_use_mem:
            self.read_controller = self.read_controller.to(self.device)

        if batch_idx > 0 and self.verbose:
            print('adaptation: batch_idx : {}'.format(batch_idx))

        # original_device = inp_seq.device

        criterion = torch.nn.BCELoss(reduction='none')

        inp_seq = inp_seq.to(self.device)
        trg_seq = trg_seq.to(self.device)

        batch_size = len(inp_seq)

        # rnn operation device setting
        if torch.cuda.is_available():
            rnn_device = torch.device("cuda")
        else:
            rnn_device = self.device
            
        popl_model = popl_model.to(rnn_device)

        hidden = popl_model.init_hidden(
            batch_size=batch_size, device=rnn_device)
        
        if self.use_context_target_stats:
            self.log_idf_context = self.log_idf_context.to(self.device)
            self.prob_context_target = self.prob_context_target.to(self.device)

            

        with torch.no_grad():
            plain_out, time_pred, hidden = popl_model(inp_seq.to(rnn_device), 
                                                    len_inp_step,
                                                    hidden,
                                                    trg_times=trg_time_seq,
                                                    inp_times=inp_time_seq,
                                                    return_hidden_seq=True)
        
        # back to current device from RNN device
        plain_out = plain_out.to(self.device)
        hidden = hidden.to(self.device)

        pred_out_seq = []
        effective_seq_len = len_inp_step.max().item()
        if mode == 'ltam_write':
            self.write_cnt_steps += len_inp_step.sum().item()
        # effective_seq_len = inp_seq.size(1)

        if self.verbose:
            print(f"NPM mode: {mode}")
        
        if (self.pred_labs or self.pred_normal_labchart) and self.use_mapped_key:
            inp_seq = fit_input_to_output(inp_seq,
                self.inv_id_mapping.keys())

        for sub_step in range(effective_seq_len):

            # autoregressive sequence extract
            current_input = inp_seq[:, sub_step]
            current_hidden = hidden[:, sub_step]
            next_trg = trg_seq[:, sub_step]
            next_plain_output = plain_out[:, sub_step] 

            next_pred = torch.sigmoid(next_plain_output)

            if self.mem_key == 'input':
                mem_key = current_input
            else:
                mem_key = current_hidden


            if self.mem_policy == 'error':
                popl_model_error = criterion(next_pred, next_trg)
            else:
                popl_model_error = None

            if mode == 'ltam_write':
                if self.mem_content == 'target':
                    mem_value = next_trg
                elif self.mem_content == 'prob_adjust':
                    mem_value = next_trg - next_pred
                else:
                    raise NotImplementedError

                self.memnet.write_mem(
                    key=mem_key.to(self.device),
                    value=mem_value.to(self.device), 
                    error=popl_model_error.to(self.device))

            else:
                if self.verbose:
                    print("read mem")

                if not (self.mem_merge.startswith('fixed_prob') or self.mem_merge == 'random'):
                    predicted_read, error_slots, similarity_slot = self.memnet.read_mem(
                        query=mem_key.to(self.device), read_mode=self.read_mode, 
                        content_avg=(self.mem_merge != 'attention' and not self.nn_mlp),
                        width=self.gaussian_width, shrink_event_dim=self.shrink_event_dim)


                if self.mem_merge == 'lambdah':
                    next_pred = predicted_read * \
                        self.lambdah + (1-self.lambdah) * next_pred

                elif self.mem_merge == 'concat':

                    next_pred = torch.cat(
                        (next_pred, predicted_read), dim=1)
                    event_mask = set_event_mask(
                        event_size=self.target_size, param_size=1)
                    event_mask = torch.cat((torch.ones(
                        self.target_size, self.target_size), event_mask), dim=1).to(self.device)
                    self.fc_out_merge.weight.data.mul_(event_mask)

                    next_pred = self.fc_out_merge(next_pred)

                elif self.mem_merge == 'threshold':

                    read_gate = error_slots > self.read_threshold

                    if self.similarity_threshold is not None:
                        read_gate = read_gate * \
                            (similarity_slot > self.similarity_threshold)

                    read_gate = read_gate.unsqueeze(1)

                    next_pred = predicted_read * read_gate + next_pred

                elif self.mem_merge == 'add':
                    next_pred = predicted_read + next_pred

                elif self.mem_merge == 'pred_error':
                    """
                    parameters of pred_error_model() are trained outside of this function 
                    (on separate step)
                    """
                    lstm_error_pred = self.pred_error_model(current_hidden)
                    next_pred = (
                        1-lstm_error_pred) * predicted_read + lstm_error_pred * next_pred

                elif self.mem_merge == 'gating_hidden_and_mem':
                    """
                    learn the amount of memory contents to be read in separate loop
                    """
                    read_input = torch.cat(
                        (current_hidden, predicted_read), dim=1)
                    read_gate = self.read_controller(read_input)
                    next_pred = predicted_read * read_gate + next_pred
                
                elif self.mem_merge == 'gating_hidden':
                    read_gate = self.read_controller(current_hidden)
                    next_pred = predicted_read * read_gate + next_pred * (1 - read_gate)


                elif self.mem_merge == 'nn-mlp':
                    """
                    learn the amount of memory contents to be read in separate loop
                    """
                    read_gate = self.read_controller(
                        current_hidden, predicted_read)
                    next_pred = predicted_read * read_gate + next_pred


                elif self.mem_merge == 'attention':

                    if self.verbose:
                        logger.debug(f"self.device: {self.device}")
                        logger.debug(f"mem_key.device: {mem_key.device}")
                        logger.debug(
                            f"self.linear_t.weight.device: {self.linear_t.weight.device}")

                    attn_query = self.linear_t(mem_key).unsqueeze(1)
                    if self.mem_key == 'input':
                        attn_query = torch.sigmoid(attn_query)
                    
                    n_batch = predicted_read.size(0)
                    zero_tensor = torch.zeros(
                        n_batch, 1, self.target_size, device=self.device)
                    
                    attn_context = torch.cat(
                        (predicted_read, zero_tensor), dim=1)

                    attn_read, attn_scores = self.attn(
                        query=attn_query, 
                        proj_key=attn_context,
                        value=attn_context)
                    attn_read = attn_read.squeeze(1)
                    
                    next_pred = attn_read + next_pred
                    # next_plain_output = attn_read

                elif self.mem_merge == 'mem_only':

                    next_pred = predicted_read

                elif self.mem_merge == 'mem_only_gating':
                    read_gate = self.read_controller(current_hidden)
                    next_pred = predicted_read * read_gate


                elif self.mem_merge == 'random':
                    next_pred = torch.rand(
                        next_pred.shape, device=next_pred.device)

                elif self.mem_merge.startswith('fixed_prob'):
                    prob = float(self.mem_merge.lstrip('fixed_prob'))

                    next_pred = torch.ones(
                        next_pred.shape, device=next_pred.device) * prob

                else:
                    next_pred = next_pred
            

                if self.use_context_target_stats:

                    sig_idf = current_input * \
                        self.log_idf_context
                    sig_ct = torch.mm(
                        current_input, self.prob_context_target.t())

                    sig_read = self.context_target_mlp(sig_idf, sig_ct)
                    next_pred += sig_read

            pred_out_seq.append(next_pred)

        # zero_paddings = [torch.zeros(
        #     plain_out.size(-1)).unsqueeze(0).to(self.device)] * (inp_seq.size(1) - effective_seq_len)
        # pred_out_seq.extend(zero_paddings)

        pred_out_seq = torch.stack(pred_out_seq).transpose(0, 1)

        time_pred_seq = None

        pred_out_seq = pred_out_seq.clamp(0, 1)

        return pred_out_seq, time_pred_seq, hidden


def train_error_pred_model(self, model, hidden, error, stop_gap, min_epoch,
                           learning_rate, verbose=False):
    batch_size = self.batch_size
    criteria = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)
    losses, epoch, prev_epoch_loss = [], 0, 0
    while True:
        epoch_loss = 0
        for i in range(0, len(hidden), batch_size):

            inp = torch.stack(hidden[i: i+batch_size])
            trg = torch.stack(error[i: i+batch_size])

            model.zero_grad()
            pred = model(inp)

            # average over batch elements first -> target specific errors.
            loss = criteria(pred, trg).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch > 0 and ((prev_epoch_loss - epoch_loss) < stop_gap and epoch > min_epoch):
            break
        prev_epoch_loss = epoch_loss
        losses.append(epoch_loss)
        if self.verbose:
            print(
                f"error_pred_model: train: epoch{epoch}, loss: {epoch_loss:.8f}")
        epoch += 1
    if self.verbose:
        print("error_pred_model: train done.")
    return model
