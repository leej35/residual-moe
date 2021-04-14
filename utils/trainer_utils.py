from torch.autograd import Function
import sys
import logging
import hashlib

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau

from utils.evaluation_utils import MultiLabelEval, export_timestep_metrics
from utils.tensor_utils import \
    DatasetWithLength_multi, \
    DatasetWithLength_single, \
    padded_collate_multi, \
    padded_collate_single, \
    sort_minibatch_multi, \
    sort_minibatch_single

from utils.project_utils import masked_unroll_loss

sigmoid = nn.Sigmoid()
relu = nn.ReLU()

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)

def remove_zeroed_batch_elems(trainer, hidden, inp_seq, trg_seq, len_inp_step,
                                len_trg_step, inp_time_seq=None,
                                trg_time_seq=None):
    # if type(hidden) not in [list, tuple]:
    #     hidden = hidden.unsqueeze(0)

    _len_inp_step = len_inp_step
    for b_idx, b_len in reversed(list(enumerate(_len_inp_step))):
        if b_len == 0:
            if trainer.model.rnn_type in trainer.model.hier_lstms:
                for nl in range(trainer.model.num_layers):
                    if hidden[nl][0].size(0) != 1:
                        hidden[nl][0] = hidden[nl][0].squeeze()
                        hidden[nl][1] = hidden[nl][1].squeeze()

                    h0 = torch.cat([hidden[nl][0][0:b_idx],
                                    hidden[nl][0][b_idx + 1:]])
                    c0 = torch.cat([hidden[nl][1][0:b_idx],
                                    hidden[nl][1][b_idx + 1:]])
                    hidden[nl] = [h0, c0]

            elif trainer.model.rnn_type in ['LSTM', 'MyLSTM']:
                while len(hidden[0].size()) > 2:
                    hidden[0] = hidden[0].squeeze(0)
                    hidden[1] = hidden[1].squeeze(0)

                h0 = torch.cat([hidden[0][0:b_idx],
                                hidden[0][b_idx + 1:]])
                c0 = torch.cat([hidden[1][0:b_idx],
                                hidden[1][b_idx + 1:]])
                hidden = [h0, c0]

            elif trainer.model.rnn_type in ['GRU']:
                hidden = torch.cat([hidden[0:b_idx],
                                    hidden[b_idx + 1:]])

            if trainer.model.rnn_type in trainer.model.hier_lstms or \
                    trainer.model.rnn_type in ['LSTM', 'MyLSTM', 'GRU',
                                                'NoRNN']:
                inp_seq = torch.cat([inp_seq[0:b_idx],
                                        inp_seq[b_idx + 1:]])
                trg_seq = torch.cat([trg_seq[0:b_idx],
                                        trg_seq[b_idx + 1:]])

                len_inp_step = torch.cat([len_inp_step[0:b_idx],
                                            len_inp_step[b_idx + 1:]])
                len_trg_step = torch.cat([len_trg_step[0:b_idx],
                                            len_trg_step[b_idx + 1:]])

            if trg_time_seq is not None:
                trg_time_seq = torch.cat([trg_time_seq[0:b_idx],
                                            trg_time_seq[b_idx + 1:]])
            if inp_time_seq is not None:
                inp_time_seq = torch.cat([inp_time_seq[0:b_idx],
                                            inp_time_seq[b_idx + 1:]])

    if type(hidden) not in [list, tuple]:
        hidden = hidden.unsqueeze(0)

    return hidden, inp_seq, trg_seq, len_inp_step, len_trg_step, inp_time_seq, trg_time_seq

def get_optimizer(trainer):
    if trainer.optim == 'sparse_adam':
        optimizer = optim.SparseAdam(trainer.model.parameters(),
                                        lr=trainer.learning_rate)
    elif trainer.optim == 'sgd':
        optimizer = optim.SGD(trainer.model.parameters(),
                                lr=trainer.learning_rate,
                                weight_decay=trainer.weight_decay)
    elif trainer.optim == 'rmsprop':
        optimizer = optim.RMSprop(trainer.model.parameters(),
                                    lr=trainer.learning_rate,
                                    weight_decay=trainer.weight_decay)
    elif trainer.optim == 'clippedadam':
        optimizer = None
        # optimizer = ClippedAdam(trainer.model.parameters(),
        #                         lr=trainer.learning_rate)
    else:
        optimizer = optim.Adam(trainer.model.parameters(),
                                lr=trainer.learning_rate,
                                weight_decay=trainer.weight_decay)
    return optimizer


def get_scheduler(trainer, optimizer):
    if trainer.lr_scheduler_ror:
        scheduler = ReduceLROnPlateau(
            optimizer, factor=trainer.lr_scheduler_mult, verbose=True, patience=5)
    elif trainer.lr_scheduler_multistep:
        scheduler = MultiStepLR(optimizer,
                                milestones=trainer.lr_scheduler_epochs,
                                gamma=trainer.lr_scheduler_mult)
    else:
        scheduler = StepLR(optimizer,
                            step_size=trainer.lr_scheduler_numiter,
                            gamma=trainer.lr_scheduler_mult)
    return scheduler

def get_dataloader(trainer, input_data, shuffle=True):
    if isinstance(input_data, tuple):
        if trainer.target_type == 'multi':
            data = DatasetWithLength_multi(input_data)
        elif trainer.target_type == 'single':
            data = DatasetWithLength_single(input_data)
        else:
            raise NotImplementedError
    else:
        data = input_data

    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=trainer.batch_size,
        shuffle=shuffle,
        num_workers=trainer.num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=padded_collate_multi
        if trainer.target_type == 'multi' \
        else padded_collate_single)
    return dataloader



def _get_memorize_time(trainer, dataloader):
    trg_times = []
    for i, data in enumerate(dataloader):
        if trainer.pred_time:
            _, _, trg_time, _, _ = data
            trg_time = trg_time / (trainer.window_hr_y * 3600)
            # trg_time: n_batch x maxseqlen x n_events
            trg_times.append(trg_time.view(-1, trg_time.size(-1)))
    trg_times = torch.cat(trg_times, dim=0)
    cnt_timing_nz = torch.sum(trg_times > 0, dim=0)
    mem_time_avg = torch.mean(trg_times, dim=0)
    if trainer.use_cuda:
        mem_time_avg = mem_time_avg.to(trainer.device)
    trainer.model.mem_time_avg = torch.nn.Parameter(mem_time_avg)

def process_batch(trainer, data, epoch):

    if trainer.target_type == 'multi':
        if trainer.pred_time or trainer.elapsed_time:
            inp, trg, inp_time, trg_time, len_inp, len_trg, hadmids = data

            if type(inp_time) == torch.Tensor:
                inp_time = inp_time / 3600
            else:
                inp_time = [[[tk/3600 for tk in tj]
                    for tj in ti] for ti in inp_time]

            if type(trg_time) == torch.Tensor:
                trg_time = trg_time / 3600
            else:
                trg_time = [[[tk/3600 for tk in tj]
                    for tj in ti] for ti in trg_time]
        else:
            inp, trg, len_inp, len_trg, hadmids = data
            trg_time = inp_time = None

        """
        inp: batch_size x max_seq_len x n_events
        len_inp: batch_size
        """

        if type(inp) == torch.Tensor and type(trg) == torch.Tensor:
            inp, trg, len_inp, len_trg, trg_time, inp_time, hadmids = \
                sort_minibatch_multi(inp, trg, len_inp, len_trg,
                                     trg_time, inp_time, hadmids)
            inp, trg = inp.float(), trg.float()
            len_inp = torch.LongTensor(len_inp)
            len_trg = torch.LongTensor(len_trg)

        if trainer.curriculum_learning:
            max_seq_len = int(trainer.max_seq_len_init * (
                trainer.curriculum_rate ** epoch))
            max_seq_len = min(max_seq_len, inp.size(1))
            index = torch.LongTensor(list(range(max_seq_len)))
            if trainer.use_cuda:
                index = index.to(trainer.device)

            inp = torch.index_select(inp, 1, index)
            len_inp = [min(max_seq_len, x) for x in len_inp]

    elif trainer.target_type == 'single':
        events, times, lengths = data
        events, times, lengths = sort_minibatch_single(events,
                                                        times,
                                                        lengths)
        inp = events[:, :-1]
        trg = events[:, 1:]
        inp_time = times[:, :-1]
        trg_time = times[:, 1:]
        len_inp = torch.LongTensor(lengths) - 1
        len_trg = torch.LongTensor(lengths) - 1
    else:
        raise NotImplementedError

    if trainer.use_cuda:
        if type(inp) == torch.Tensor:
            inp = inp.to(trainer.device)
        if type(trg) == torch.Tensor:
            trg = trg.to(trainer.device)
        if inp_time is not None and type(inp_time) == torch.Tensor:
            inp_time = inp_time.to(trainer.device)
        if trg_time is not None and type(trg_time) == torch.Tensor:
            trg_time = trg_time.to(trainer.device)

    return inp, trg, len_inp, len_trg, inp_time, trg_time, hadmids


def _remove_chartevent_discrete_value(trainer, item_label):
    if item_label.startswith('chart'):
        item_label = item_label[:item_label.rfind('-')]
    return item_label


def get_event_weights(trainer, weight_file, import_val=1.0, nonimport_val=0.5):

    imp_events = list(np.loadtxt(weight_file, delimiter=',', dtype=str))
    imp_events = [trainer._remove_chartevent_discrete_value(
        x) for x in imp_events]

    weight_vector = torch.ones(len(trainer.event_dic)).to(
        trainer.device) * nonimport_val

    for idx, info in trainer.event_dic.items():
        if info['category'] + '--' + info['label'] in imp_events:
            weight_vector[idx - 1] = import_val

    return weight_vector


def run_evals(trainer, csv_file, eval, test_name, final, export_csv, eval_multithread, cur_epoch):

    # adaptive switch counter save
    if csv_file is not None:
        # switch count
        if sum(trainer.adapt_switch_stat.values()) != 0:
            trainer.adapt_switch_stat["ratio_instance_model"] = trainer.adapt_switch_stat["inst"] / sum(
                trainer.adapt_switch_stat.values())
        else:
            trainer.adapt_switch_stat["ratio_instance_model"] = 0

        with open(csv_file.replace(".csv", "_adapt_switch_stat.txt"), 'w') as f:
            print(trainer.adapt_switch_stat, file=f)

        # switch count by steps
        df_switch = pd.DataFrame.from_dict(
            trainer.adapt_switch_stat_steps, orient='index')
        if trainer.adapt_switch_stat_steps != {}:
            df_switch["ratio_instance_model"] = df_switch.iloc[:,
                                                                1] / (df_switch.iloc[:, 0] + df_switch.iloc[:, 1])
        df_switch.to_csv(csv_file.replace(
            ".csv", "_adapt_switch_stat_steps.csv"))

        # patient count by steps
        df_ptn = pd.DataFrame.from_dict(
            trainer.patient_count_steps, orient='index')
        df_switch.to_csv(csv_file.replace(
            ".csv", "_patient_count_steps.csv"))

    logger.info('first occur')
    eval.compute(eval.eval['first_occur'], epoch=cur_epoch,
                    test_name=test_name,
                    web_logger=trainer.web_logger,
                    option_str=' first_occur',
                    final=final,
                    event_dic=trainer.event_dic,
                    export_csv=(export_csv and final),
                    csv_file=csv_file.replace(
                        '.csv', '_occur_first.csv'),
                    use_multithread=eval_multithread)
    logger.info('second occur')
    eval.compute(eval.eval['second_occur'], epoch=cur_epoch,
                    test_name=test_name,
                    web_logger=trainer.web_logger,
                    option_str=' second_occur',
                    final=final,
                    event_dic=trainer.event_dic,
                    export_csv=(export_csv and final),
                    csv_file=csv_file.replace(
        '.csv', '_occur_second.csv'),
        use_multithread=eval_multithread)
    logger.info('second and later occur')
    eval.compute(eval.eval['second_later_occur'], epoch=cur_epoch,
                    test_name=test_name,
                    web_logger=trainer.web_logger,
                    option_str=' second_later_occur',
                    final=final,
                    event_dic=trainer.event_dic,
                    export_csv=(export_csv and final),
                    csv_file=csv_file.replace(
        '.csv', '_occur_second_later.csv'),
        use_multithread=eval_multithread)
    logger.info('later occur')
    eval.compute(eval.eval['later_occur'], epoch=cur_epoch,
                    test_name=test_name,
                    web_logger=trainer.web_logger,
                    option_str=' later_occur',
                    final=final,
                    event_dic=trainer.event_dic,
                    export_csv=(export_csv and final),
                    csv_file=csv_file.replace(
        '.csv', '_occur_later.csv'),
        use_multithread=eval_multithread)

    overall = eval.eval['overall_num'].cpu()
    first = eval.eval['first_occur_num'].cpu()
    second = eval.eval['second_occur_num'].cpu()
    later = eval.eval['later_occur_num'].cpu()
    denom = first + second + later + 1e-10

    first_ratio = first / denom * 100
    second_ratio = second / denom * 100
    later_ratio = later / denom * 100

    logger.info(
        'overall: {}. first+second+later={}'.format(overall,
                                                    denom - 1e-10))

    logger.info('first: {:.3f}% {}'.format(first_ratio, first))
    logger.info('second: {:.3f}% {}'.format(
        second_ratio, second))
    logger.info('later: {:.3f}% {}'.format(later_ratio, later))

    trainer.web_logger.log_metric("num_first", first)
    trainer.web_logger.log_metric("num_second", second)
    trainer.web_logger.log_metric("num_later", later)
    trainer.web_logger.log_metric("ratio_first", first_ratio)
    trainer.web_logger.log_metric(
        "ratio_second", second_ratio)
    trainer.web_logger.log_metric("ratio_later", later_ratio)

    logger.info('\n')

    logger.info('start zero')
    eval.compute(eval.eval['start_zero'], epoch=cur_epoch,
                    test_name=test_name,
                    web_logger=trainer.web_logger,
                    option_str=' start_zero',
                    final=final,
                    event_dic=trainer.event_dic,
                    export_csv=(export_csv and final),
                    csv_file=csv_file.replace(
                        '.csv', '_sz.csv'),
                    use_multithread=eval_multithread)
    logger.info('start one')
    eval.compute(eval.eval['start_one'], epoch=cur_epoch,
                    test_name=test_name, web_logger=trainer.web_logger,
                    option_str=' start_one',
                    final=final,
                    event_dic=trainer.event_dic,
                    export_csv=(export_csv and final),
                    csv_file=csv_file.replace(
                        '.csv', '_so.csv'),
                    use_multithread=eval_multithread)

    logger.info('\n')

    logger.info('occur steps')
    container_occur_steps = []
    for step in eval.eval['occur_steps'].keys():
        logger.info('occur step: {}'.format(step))
        _, _, container = eval.compute(
            eval.eval['occur_steps'][step], epoch=step,
            test_name=test_name,
            web_logger=trainer.web_logger,
            tstep=True,
            final=final,
            event_dic=trainer.event_dic,
            return_metrics=True,
            export_csv=(export_csv and final),
            csv_file=csv_file.replace(
                '.csv', '_occur_step_{}.csv'.format(step)),
            use_multithread=eval_multithread)
        container_occur_steps.append(container)

    if export_csv:
        export_timestep_metrics(
            csv_file.replace(
                '.csv', '_occur_step_all.csv'),
            trainer.model_prefix,
            container_occur_steps
        )

    logger.info('\n')

    container_steps = []
    for step in range(len(eval.eval['tstep'])):
        logger.info('step: {}'.format(step))
        _, _, container = eval.compute(
            eval.eval['tstep'][step], epoch=step,
            test_name=test_name,
            web_logger=False,  # NOTE: web log off for time-step
            tstep=True, verbose=False,
            final=final,
            return_metrics=True,
            use_multithread=eval_multithread)
        container_steps.append(container)

    if export_csv:
        export_timestep_metrics(
            csv_file.replace('.csv', '_timestep.csv'),
            trainer.model_prefix,
            container_steps,
            event_dic=trainer.event_dic,
        )

    # Repeat AND Time-step

    container_steps_first_occur = []
    container_steps_second_occur = []
    container_steps_second_later_occur = []
    container_steps_later_occur = []

    for step in range(len(eval.eval['tstep-repeat'])):

        logger.info(f"repeat-step: {step} first_occur")

        _, _, container = eval.compute(
            eval.eval['tstep-repeat'][step]['first_occur'], epoch=step,
            test_name=test_name,
            web_logger=False,  # NOTE: web log off for time-step
            tstep=True,
            verbose=True,
            final=final,
            return_metrics=True,
            use_multithread=eval_multithread)
        container_steps_first_occur.append(container)

        logger.info(f"repeat-step: {step} second_occur")

        _, _, container = eval.compute(
            eval.eval['tstep-repeat'][step]['second_occur'], epoch=step,
            test_name=test_name,
            web_logger=False,  # NOTE: web log off for time-step
            tstep=True,
            verbose=True,
            final=final,
            return_metrics=True,
            use_multithread=eval_multithread)
        container_steps_second_occur.append(container)

        logger.info(f"repeat-step: {step} second_later_occur")

        _, _, container = eval.compute(
            eval.eval['tstep-repeat'][step]['second_later_occur'], epoch=step,
            test_name=test_name,
            web_logger=False,  # NOTE: web log off for time-step
            tstep=True,
            verbose=True,
            final=final,
            return_metrics=True,
            use_multithread=eval_multithread)
        container_steps_second_later_occur.append(
            container)

        logger.info(f"repeat-step: {step} later_occur")

        _, _, container = eval.compute(
            eval.eval['tstep-repeat'][step]['later_occur'], epoch=step,
            test_name=test_name,
            web_logger=False,  # NOTE: web log off for time-step
            tstep=True,
            verbose=True,
            final=final,
            return_metrics=True,
            use_multithread=eval_multithread)
        container_steps_later_occur.append(container)

    if export_csv:
        export_timestep_metrics(
            csv_file.replace(
                '.csv', '_timestep_{}.csv'.format('first_occur')),
            trainer.model_prefix,
            container_steps_first_occur
        )
        export_timestep_metrics(
            csv_file.replace(
                '.csv', '_timestep_{}.csv'.format('second_occur')),
            trainer.model_prefix,
            container_steps_second_occur
        )
        export_timestep_metrics(
            csv_file.replace(
                '.csv', '_timestep_{}.csv'.format('second_later_occur')),
            trainer.model_prefix,
            container_steps_second_later_occur
        )
        export_timestep_metrics(
            csv_file.replace(
                '.csv', '_timestep_{}.csv'.format('later_occur')),
            trainer.model_prefix,
            container_steps_later_occur
        )

    if trainer.event_dic and 'category' in list(trainer.event_dic.values())[0]:
        for etype in trainer.event_types:
            logger.info('event-type: {}'.format(etype))
            eval.compute(eval.eval['etypes'][etype],
                            epoch=cur_epoch,
                            test_name=test_name,
                         web_logger=trainer.web_logger,
                            option_str=' ' + etype,
                            event_dic=trainer.event_dic,
                            export_csv=(export_csv and final),
                            csv_file=csv_file.replace(
                '.csv', '_category_{}.csv'.format(etype)),
                final=final)

            # Category AND Time-step
            container_steps = []
            for step in range(len(eval.eval['tstep-etype'])):
                logger.info('step: {}'.format(step))
                _, _, container = eval.compute(
                    eval.eval['tstep-etype'][step][etype], epoch=step,
                    test_name=test_name,
                    web_logger=False,  # NOTE: web log off for time-step
                    tstep=True,
                    verbose=False,
                    final=final,
                    return_metrics=True,
                    use_multithread=eval_multithread)
                container_steps.append(container)

            if export_csv:
                export_timestep_metrics(
                    csv_file.replace(
                        '.csv', '_timestep_{}.csv'.format(etype)),
                    trainer.model_prefix,
                    container_steps
                )
            logger.info(
                'event-type: {} [start-one]'.format(etype))
            eval.compute(eval.eval['etypes_start_one'][etype],
                            epoch=cur_epoch,
                            test_name=test_name,
                         web_logger=trainer.web_logger,
                            option_str=' ' + etype + ' start_one',
                            event_dic=trainer.event_dic,
                            export_csv=(export_csv and final),
                            csv_file=csv_file.replace(
                                '.csv', '_cat_so_{}.csv'.format(etype)),
                            final=final)

            logger.info(
                'event-type: {} [start-zero]'.format(etype))
            eval.compute(eval.eval['etypes_start_zero'][etype],
                            epoch=cur_epoch,
                            test_name=test_name,
                         web_logger=trainer.web_logger,
                            option_str=' ' + etype + ' start_zero',
                            export_csv=(export_csv and final),
                            event_dic=trainer.event_dic,
                            csv_file=csv_file.replace(
                                '.csv', '_cat_sz_{}.csv'.format(etype)),
                            final=final)

            logger.info('event-type: {} [occur]'.format(etype))
            eval.compute(eval.eval['etypes_start_zero'][etype],
                            epoch=cur_epoch,
                            test_name=test_name,
                         web_logger=trainer.web_logger,
                            option_str=' ' + etype + ' start_zero',
                            export_csv=(export_csv and final),
                            event_dic=trainer.event_dic,
                            csv_file=csv_file.replace(
                                '.csv', '_cat_sz_{}.csv'.format(etype)),
                            final=final)

            logger.info(
                'event-type: {} [first occur]'.format(etype))
            eval.compute(eval.eval['etypes_first_occur'][etype], epoch=cur_epoch,
                            test_name=test_name,
                         web_logger=trainer.web_logger,
                            option_str=' ' + etype + ' first_occur',
                            final=final,
                            event_dic=trainer.event_dic,
                            export_csv=(export_csv and final),
                            csv_file=csv_file.replace(
                                '.csv', '_occur_first_{}.csv'.format(etype)),
                            use_multithread=eval_multithread)

            logger.info(
                'event-type: {} [second occur]'.format(etype))
            eval.compute(eval.eval['etypes_second_occur'][etype], epoch=cur_epoch,
                            test_name=test_name,
                         web_logger=trainer.web_logger,
                            option_str=' ' + etype + ' second_occur',
                            final=final,
                            event_dic=trainer.event_dic,
                            export_csv=(export_csv and final),
                            csv_file=csv_file.replace(
                '.csv', '_occur_second_{}.csv'.format(etype)),
                use_multithread=eval_multithread)

            logger.info(
                'event-type: {} [later occur]'.format(etype))
            eval.compute(eval.eval['etypes_later_occur'][etype], epoch=cur_epoch,
                            test_name=test_name,
                         web_logger=trainer.web_logger,
                            option_str=' ' + etype + ' later_occur',
                            final=final,
                            event_dic=trainer.event_dic,
                            export_csv=(export_csv and final),
                            csv_file=csv_file.replace(
                '.csv', '_occur_later_{}.csv'.format(etype)),
                use_multithread=eval_multithread)


def run_non_RNN_models(trainer, item_avg_cnt, inp, trg, trg_time, len_inp, train_x, train_y, optimizer, t_loss, update_cnt,):

    if trainer.baseline_type.startswith('majority'):
        item_avg_cnt.append(
            (inp.sum() / (inp.size()[0] * inp.size()[1])).item())

    # if trainer.baseline_type in ['copylast', 'majority_ts',
    #                             'timing_random', 'random']:
    #     continue

    elif trainer.baseline_type == 'majority_all':
        trainer.event_counter += inp.view(-1, inp.size(2)).sum(0)

    elif trainer.baseline_type in ['sk_timing_linear_reg']:
        eff_inp = []

        for inp_step in range(inp.size(1)):
            eff_inp.append(inp[:, :(inp_step + 1), :].sum(1))
        eff_inp = torch.stack(eff_inp, dim=1)
        train_x.append(eff_inp)
        train_y.append(trg_time.cpu())

    elif (trainer.baseline_type.startswith("logistic")
            or trainer.baseline_type == 'timing_linear_reg'):
        eff_inp = []

        if trainer.baseline_type in ['logistic_binary',
                                    'logistic_count',
                                    'logistic_binary_mlp',
                                    'logistic_count_mlp',
                                    'timing_linear_reg']:
            for inp_step in range(inp.size(1)):
                eff_inp.append(
                    inp[:, :(inp_step + 1), :].sum(1))

            eff_inp = torch.stack(eff_inp, dim=1)

            if trainer.use_cuda:
                eff_inp = eff_inp.to(trainer.device)

            if trainer.baseline_type.startswith('logistic_binary'):
                eff_inp = (eff_inp > 0).float()

        elif trainer.baseline_type.startswith('logistic_last'):
            eff_inp = inp.float()

        trainer.model.zero_grad()

        output = trainer.model(eff_inp.view(-1, eff_inp.size(2)))

        if trainer.baseline_type.startswith("logistic"):

            output = output.view(
                (eff_inp.size(0), eff_inp.size(1), -1))
            pred_seq = sigmoid(output)

            # single output
            if trainer.target_event > -1:
                trg = trg[:, :,
                            trainer.target_event].unsqueeze(-1)

                if pred_seq.dim() == 2:
                    pred_seq = pred_seq.unsqueeze(-1)

            loss = trainer.loss_fn(pred_seq, trg, len_inp)

        elif trainer.baseline_type in ['timing_linear_reg']:
            pred_seq = relu(output.view(eff_inp.size()))
            _, loss = masked_unroll_loss(
                trainer.loss_time_fn, pred_seq, trg_time,
                len_inp, mask_neg=True)

        t_loss += loss.item()
        loss.backward()
        optimizer.step()
        update_cnt += 1

    elif trainer.baseline_type == 'timing_mean':
        trainer.timing_vec = torch.cat([trainer.timing_vec,
                                        trg_time.view(-1,
                                                        trainer.event_size)])

    elif trainer.baseline_type.startswith('hmm'):
        # combine x and y
        seq = torch.cat(
            (inp, trg[:, -1, :].unsqueeze(1)), dim=1)
        seq_len = len_inp + 1
        seq = seq.to(trainer.device)
        seq_len = seq_len.to(trainer.device)
        args = trainer.model['args']
        svi = trainer.model['svi']

        num_observations = float(seq_len.sum())

        with autograd.detect_anomaly():
            try:
                loss = svi.step(seq, seq_len, args=args)
            except ValueError as e:
                raise ValueError(e)
            except RuntimeError as e:
                raise RuntimeError(e)

        t_loss += loss / num_observations
        update_cnt += 1

        #NOTE: ReduceLROnPlateau not work under Pyro Yet.
        # svi.optim.set_epoch(epoch=epoch)

        # scheduler = trainer.model['scheduler']

    else:
        raise NotImplementedError

    return output, item_avg_cnt, t_loss, update_cnt, train_x, train_y


def get_hash(x):
    return hashlib.md5(x).digest()


# from https://github.com/fungtion/DANN/blob/master/models/functions.py
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def get_hash(x):
    return hashlib.md5(x).digest()
