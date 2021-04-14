import sys
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.project_utils import set_event_mask
from models.residual_seq_net import ResidualSeqNet
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)



def neural_caching_model(trainer, window, theta, lambdah, inp_seq, trg_seq, inp_time_seq, trg_time_seq,
                            len_inp_step, popl_model, b, full_batch_size,
                            tol=1e-03, d_loss_tol=1e-04, batch_idx=0):

    if batch_idx > 0:
        logger.info('adaptation: batch_idx : {}'.format(batch_idx))

    original_device = inp_seq.device

    # if torch.cuda.is_available():
        # Switch to GPU mode if possible

    device = torch.device('cuda')
    trainer = trainer.to(device)
    trainer.device = device
    trainer.use_cuda = True
    inp_seq, trg_seq = inp_seq.to(device), trg_seq.to(device)

    batch_size = len(inp_seq)

    hidden = popl_model.init_hidden(batch_size=batch_size, device=device)

    plain_out, time_pred, hidden = popl_model(inp_seq, len_inp_step,
                                                hidden,
                                                trg_times=trg_time_seq,
                                                inp_times=inp_time_seq,
                                                return_hidden_seq=True)

    plain_out_seq = []
    effective_seq_len = len_inp_step.item()
    for sub_step in range(effective_seq_len):

        # autoregressive sequence extract
        sub_inp = inp_seq[:, :(sub_step + 1)]
        sub_trg = trg_seq[:, :(sub_step + 1)]
        sub_hidden = hidden[:, :(sub_step + 1)]

        next_plain_output = plain_out[:, sub_step]

        if sub_step > 0:
            valid_pointer_history = sub_hidden[:, - \
                (window + 1):-1].squeeze(0)
            valid_next_history = sub_trg[:, -(window + 1):-1]
            current_hidden_step = sub_hidden[:, -1].squeeze(0)
            logits = torch.mv(valid_pointer_history, current_hidden_step)
            ptr_attn = F.softmax(theta * logits).view(-1, 1)
            ptr_dist = (ptr_attn.expand_as(valid_next_history)
                        * valid_next_history).squeeze().sum(0)

            next_plain_output = lambdah * ptr_dist + \
                (1 - lambdah) * next_plain_output

        plain_out_seq.append(next_plain_output)

    zero_paddings = [torch.zeros(
        plain_out.size(-1)).unsqueeze(0).to(device)] * (inp_seq.size(1) - effective_seq_len)
    plain_out_seq.extend(zero_paddings)

    plain_out_seq = torch.stack(plain_out_seq).transpose(0, 1)

    time_pred_seq = None

    # Switch back to CPU mode if it is original device
    device = torch.device('cpu')
    trainer = trainer.to(device)
    trainer.device = device
    trainer.use_cuda = False
    plain_out_seq = plain_out_seq.to(device)
    hidden = hidden.to(device)
    return plain_out_seq, time_pred_seq, hidden

@staticmethod
def compute_decayed_loss(criterion, pred, trg, kernel_bandwidth, kernel_func):
    loss = criterion(pred, trg)

    if loss.dim() == 4 and loss.size(2) == 1:
        loss = loss.squeeze(2)

    # apply decay kernel
    kernel = kernel_func(loss, kernel_bandwidth)
    loss = loss * kernel
    loss = loss.mean()

    return loss

@staticmethod
def decay_kernel(seq_tensor, bandwidth=3):
    # bandwith: if it is large, it returns uniform dist
    #           if it is small (=1), put high weight on very recent.
    # seq_tensor: n_batch x n_step x n_events
    n_step = seq_tensor.size(1)
    steps = torch.range(1, n_step).to(seq_tensor.device) - 1
    kernel = torch.exp(-(steps / bandwidth))
    kernel = kernel.unsqueeze(0).unsqueeze(2).expand_as(seq_tensor).flip(1)
    return kernel


def get_residual_model(trainer, popl_model):
    res_model = ResidualSeqNet(popl_model.embedding_dim, popl_model.hidden_dim, popl_model.event_size,
                               popl_model.num_layers, popl_model.dropout, popl_model.device, rnn_type=trainer.model.rnn_type)

    res_model.embed_input = copy.deepcopy(popl_model.embed_input)
    res_model.rnn = copy.deepcopy(popl_model.rnn)
    res_model.fc_out = copy.deepcopy(popl_model.fc_out)

    res_model.embed_input.weight.requires_grad = False
    for param in res_model.rnn.parameters():
        param.requires_grad = False
    for param in res_model.fc_out.parameters():
        param.requires_grad = False

    return res_model


def train_adaptive_model(trainer, inp_seq, trg_seq, inp_time_seq, trg_time_seq,
                         len_inp_step, popl_model, b, full_batch_size,
                         tol=1e-03, d_loss_tol=1e-04, batch_idx=0):

    kernel_bandwidth = trainer.args.adapt_bandwidth
    loss_type = trainer.args.adapt_loss
    adapt_LR = trainer.args.adapt_lr

    original_device = inp_seq.device

    # if torch.cuda.is_available():
    # Switch to GPU mode if possible

    device = torch.device('cuda')
    trainer = trainer.to(device)
    trainer.device = device
    trainer.use_cuda = True
    inp_seq, trg_seq = inp_seq.to(device), trg_seq.to(device)
    # else:
    #     device = torch.device('cpu')

    batch_size = len(inp_seq)

    plain_out_seq, time_pred_seq = [], []

    if loss_type == 'bce':
        criterion = torch.nn.BCELoss(reduction='none')
    elif loss_type == 'mse':
        criterion = torch.nn.MSELoss(reduction='none')

    if batch_idx > 0:
        logger.info('adaptation: batch_idx : {}'.format(batch_idx))

    for sub_step in range(inp_seq.size(1)):

        trainer.cnt_time_step += 1
        if sub_step not in trainer.patient_count_steps:
            trainer.patient_count_steps[sub_step] = 0
        trainer.patient_count_steps[sub_step] += 1

        if trainer.args.adapt_pop_based or (sub_step == 0) or trainer.args.adapt_sw_pop:
            # everystep, we fork from population model, or init new at step 0

            # if trainer.args.adapt_sw_pop and sub_step > 0:
            # # compare population vs individual model
            # sub_inp_prevstep = inp_seq[:, :(sub_step)]
            # sub_trg_prevstep = trg_seq[:, :(sub_step)]
            # hidden = popl_model.init_hidden(batch_size=batch_size, device=gpu)

            if trainer.adapt_lstm or trainer.adapt_lstm_only or trainer.adapt_fc_only:
                step_model = copy.deepcopy(popl_model).to(device)

                # freeze weights
                step_model.embed_input.weight.requires_grad = False

                if trainer.adapt_lstm_only:
                    # step_model.fc_out.weight.requires_grad = False
                    # step_model.fc_out.bias.requires_grad = False
                    for param in step_model.fc_out.parameters():
                        param.requires_grad = False

                if trainer.adapt_fc_only:
                    # !! Wrong Way: step_model.rnn.requires_grad = False
                    for param in step_model.rnn.parameters():
                        param.requires_grad = False

                optimizer = optim.Adam(
                    step_model.parameters(), lr=adapt_LR)

            elif trainer.adapt_residual:
                step_model = trainer.get_residual_model(popl_model).to(device)

                optimizer = optim.Adam(
                    [
                        {
                            'params': step_model.fc_out_residual.parameters(),
                            'weight_decay': trainer.args.adapt_residual_wdecay
                        },
                        {
                            'params': step_model.embed_input.parameters(),
                            'weight_decay': trainer.weight_decay
                        },
                        {
                            'params': step_model.rnn.parameters(),
                            'weight_decay': trainer.weight_decay
                        },
                        {
                            'params': step_model.fc_out.parameters(),
                            'weight_decay': trainer.weight_decay
                        }
                    ], lr=adapt_LR)

        step_model.train()

        hidden = popl_model.init_hidden(
            batch_size=batch_size, device=device)

        # logger.info("hidden:{}".format(hidden))

        # autoregressive sequence extract
        sub_inp = inp_seq[:, :(sub_step + 1)]
        sub_trg = trg_seq[:, :(sub_step + 1)]

        # start sub training routine
        prev_loss = 0
        mini_epoch = 1
        patient_cnt = 0

        if trainer.args.verbose:
            logger.info('-'*16)

        if trainer.args.verbose and sub_step > 0:
            _sub_inp_step = sub_inp[:, :-1]
            _len_inp_step = torch.LongTensor(
                [_sub_inp_step.size(1)]).to(device)

            output, time_pred, hidden = step_model(
                _sub_inp_step,
                _len_inp_step,
                hidden,
                trg_times=trg_time_seq,
                inp_times=inp_time_seq,
            )
            output = torch.sigmoid(output)
            before_loss = criterion(output, sub_trg[:, :-1]).mean()
            logger.info(f'before adaptive train loss: {before_loss}')

        while sub_step > 0:
            step_model.zero_grad()
            # make sure that we use the last step's hidden for the prediction
            _sub_inp_step = sub_inp[:, :-1]
            _len_inp_step = torch.LongTensor(
                [_sub_inp_step.size(1)]).to(device)

            output, time_pred, hidden = step_model(
                _sub_inp_step,
                _len_inp_step,
                hidden,
                trg_times=trg_time_seq,
                inp_times=inp_time_seq,
            )
            output = torch.sigmoid(output)

            loss = trainer.compute_decayed_loss(
                criterion, output, sub_trg[:, :-1], kernel_bandwidth, trainer.decay_kernel)

            loss.backward(retain_graph=True)
            optimizer.step()

            d_loss = prev_loss - loss

            if loss > prev_loss:
                patient_cnt += 1
            else:
                patient_cnt = 0

            if trainer.args.verbose:
                logger.info('batch:{}/{} seq:{}/{} mini epoch: {}, loss: {:.8f}, d_loss:{:.8f} patient_cnt:{}'.format(
                    b, full_batch_size, sub_step, inp_seq.size(1), mini_epoch, loss, d_loss, patient_cnt))

            if (patient_cnt > trainer.patient) or (loss < tol) or (abs(d_loss) < d_loss_tol):
                if trainer.args.verbose and sub_step > 0:
                    _sub_inp_step = sub_inp[:, :-1]
                    _len_inp_step = torch.LongTensor(
                        [_sub_inp_step.size(1)]).to(device)

                    output, time_pred, hidden = step_model(
                        _sub_inp_step,
                        _len_inp_step,
                        hidden,
                        trg_times=trg_time_seq,
                        inp_times=inp_time_seq,
                    )
                    output = torch.sigmoid(output)

                    after_loss = criterion(output, sub_trg[:, :-1]).mean()
                    logger.info(
                        f'after (in-loop) adaptive train loss: {after_loss}')

                break

            prev_loss = loss
            mini_epoch += 1
            trainer.cnt_update += 1

        if trainer.args.verbose and sub_step > 0:
            _sub_inp_step = sub_inp[:, :-1]
            _len_inp_step = torch.LongTensor(
                [_sub_inp_step.size(1)]).to(device)

            output, time_pred, hidden = step_model(
                _sub_inp_step,
                _len_inp_step,
                hidden,
                trg_times=trg_time_seq,
                inp_times=inp_time_seq,
            )
            output = torch.sigmoid(output)
            after_loss = criterion(output, sub_trg[:, :-1]).mean()
            logger.info(f'after (out-loop) adaptive train loss: {after_loss}')

        # output after sub-training
        step_model.eval()
        _len_inp_step = torch.LongTensor([sub_inp.size(1)]).to(device)
        instance_out, time_pred, hidden = step_model(
            sub_inp,
            _len_inp_step,
            hidden,
            trg_times=trg_time_seq,
            inp_times=inp_time_seq,
        )

        # default output
        pred_step = instance_out[:, -1]

        if trainer.args.adapt_switch:
            # Online Switching model: choose best among population model and instance-specific
            # based on recent error rate.

            hidden = popl_model.init_hidden(
                batch_size=batch_size, device=device)
            popl_model.eval()

            try:
                _len_inp_step = torch.LongTensor(
                    [sub_inp.size(1)]).to(device)
                pop_output, time_pred, hidden = popl_model(
                    sub_inp,
                    _len_inp_step,
                    hidden,
                    trg_times=trg_time_seq,
                    inp_times=inp_time_seq,
                )
            except RuntimeError as e:
                logger.info('sub_inp:{}'.format(sub_inp.size()))
                logger.info('hidden:{}'.format(hidden.size()))
                logger.info('len_inp_step:{}'.format(len_inp_step))
                raise e

            pop_loss = trainer.compute_decayed_loss(criterion,
                                                    torch.sigmoid(
                                                        pop_output), sub_trg, kernel_bandwidth,
                                                    trainer.decay_kernel)

            # instance models' loss
            instance_loss = trainer.compute_decayed_loss(criterion,
                                                         torch.sigmoid(
                                                             instance_out), sub_trg, kernel_bandwidth,
                                                         trainer.decay_kernel)

            if sub_step not in trainer.adapt_switch_stat_steps:
                trainer.adapt_switch_stat_steps[sub_step] = {
                    'pop': 0, 'inst': 0}

            # compare losses of the two
            if instance_loss > pop_loss:
                pred_step = pop_output[:, -1]
                trainer.adapt_switch_stat['pop'] += 1
                trainer.adapt_switch_stat_steps[sub_step]['pop'] += 1
            else:
                trainer.adapt_switch_stat['inst'] += 1
                trainer.adapt_switch_stat_steps[sub_step]['inst'] += 1

        plain_out_seq.append(pred_step)  # use the last time-step only

        if time_pred is not None:
            time_pred_seq.append(time_pred[:, -1])

    plain_out_seq = torch.stack(plain_out_seq).transpose(0, 1)

    if time_pred is not None:
        time_pred_seq = torch.stack(time_pred_seq)
    else:
        time_pred_seq = None

    # Switch back to CPU mode if it is original device
    # if original_device.type == 'cpu':
    device = torch.device('cpu')
    trainer = trainer.to(device)
    trainer.device = device
    trainer.use_cuda = False
    plain_out_seq.to(device)

    return plain_out_seq, time_pred_seq
