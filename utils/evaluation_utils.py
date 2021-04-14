
import sys
import logging
import copy
import torch
import os
import csv
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from tabulate import tabulate

import matplotlib
from functools import reduce
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tabulate.PRESERVE_WHITESPACE = True
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, \
    average_precision_score, precision_score, recall_score, confusion_matrix,\
    roc_curve

from numba import jit

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)



class MultiLabelEval(object):
    def __init__(self, event_size, use_cuda=False, device=None, event_dic=None,
                 event_types=None, micro_aucs=True, macro_aucs=True,
                 pred_labs=False, pred_normal_labchart=False):
        self.device = device
        self.eval = {}
        et = torch.FloatTensor([])
        self._metric = {'total_cnt': 0,
                        'total_exact_match': 0,
                        'total_hamming_loss': torch.zeros(1).to(device),
                        'total_correct': torch.zeros(1).to(device),
                        'total_match': torch.zeros(1).to(device),
                        'total_pred': torch.zeros(1).to(device),
                        'total_gold': torch.zeros(1).to(device),
                        'total_num': torch.zeros(1).to(device),
                        'pred_count': torch.zeros(event_size).to(device),
                        'total_cnt': 0,
                        'total_exact_match': 0,
                        }

        if macro_aucs:
            self._metric['y_probs'] = {i: [] for i in range(event_size)}
            self._metric['y_trues'] = {i: [] for i in range(event_size)}
        if micro_aucs:
            self._metric['y_ex_probs'] = {i: [] for i in range(event_size)}
            self._metric['y_ex_trues'] = {i: [] for i in range(event_size)}

        self.eval['flat'] = copy.deepcopy(self._metric)
        self.eval['start_zero'] = copy.deepcopy(self._metric)
        self.eval['start_one'] = copy.deepcopy(self._metric)
        self.eval['zero_zero'] = copy.deepcopy(self._metric)
        self.eval['zero_one'] = copy.deepcopy(self._metric)
        self.eval['one_zero'] = copy.deepcopy(self._metric)
        self.eval['one_one'] = copy.deepcopy(self._metric)

        # self.eval['interval_0'] = copy.deepcopy(self._metric)
        # self.eval['interval_1'] = copy.deepcopy(self._metric)
        # self.eval['interval_2'] = copy.deepcopy(self._metric)
        # self.eval['interval_3'] = copy.deepcopy(self._metric)
        # self.eval['interval_4'] = copy.deepcopy(self._metric)
        # self.eval['interval_5'] = copy.deepcopy(self._metric)

        self.eval['first_occur'] = copy.deepcopy(self._metric)
        self.eval['second_occur'] = copy.deepcopy(self._metric)
        self.eval['second_later_occur'] = copy.deepcopy(self._metric)
        self.eval['later_occur'] = copy.deepcopy(self._metric)
        self.eval['overall_num'] = 0
        self.eval['first_occur_num'] = 0
        self.eval['second_occur_num'] = 0
        self.eval['later_occur_num'] = 0

        self.eval['tstep'] = {}
        self.eval['tstep-etype'] = {}
        self.eval['occur_steps'] = {}
        self.eval['tstep-repeat'] = {}

        self.use_cuda = use_cuda
        self.micro_aucs = micro_aucs
        self.macro_aucs = macro_aucs

        self.pred_labs = pred_labs
        self.pred_normal_labchart = pred_normal_labchart
        self.event_dic = event_dic
        self.event_idxs = {}

        if event_dic is not None and event_types is not None:
            self.eval['etypes'] = {}
            self.event_types = event_types
            for e_type in self.event_types:
                self.eval['etypes'][e_type] = copy.deepcopy(
                    self._metric)

            for e_type in self.event_types:
                self.event_idxs[e_type] = []

            for eid, info in event_dic.items():

                # NOTE: as itemid in event_dic starts from 1 
                # but the itemid in the predicted vector starts from 0,
                # let's remap the id -1 (Sep 11 2019)

                eid = eid - 1

                self.event_idxs[info['category']].append(eid)

                if not pred_normal_labchart: 
                    #NOTE: pred_normal_labchart does not separate normal/abnormal on target side

                    if '-NORMAL' in info['label']:
                        if info['category'] == 'chart':
                            self.event_idxs['chart_normal'].append(eid)
                        elif info['category'] == 'lab':
                            self.event_idxs['lab_normal'].append(eid)

                    elif '-ABNORMAL' in info['label']:
                        if info['category'] == 'chart':
                            self.event_idxs['chart_abnormal'].append(eid)
                        elif info['category'] == 'lab':
                            self.event_idxs['lab_abnormal'].append(eid)

            self.eval['etypes_start_one'] = copy.deepcopy(self.eval['etypes'])
            self.eval['etypes_start_zero'] = copy.deepcopy(self.eval['etypes'])
            self.eval['etypes_first_occur'] = copy.deepcopy(self.eval['etypes'])
            self.eval['etypes_second_occur'] = copy.deepcopy(self.eval['etypes'])
            self.eval['etypes_later_occur'] = copy.deepcopy(self.eval['etypes'])

            self.eval['etypes_zz'] = copy.deepcopy(self.eval['etypes'])
            self.eval['etypes_zo'] = copy.deepcopy(self.eval['etypes'])
            self.eval['etypes_oz'] = copy.deepcopy(self.eval['etypes'])
            self.eval['etypes_oo'] = copy.deepcopy(self.eval['etypes'])

    def _to_cuda(self, dic):
        if type(dic) is not dict:
            raise NotImplementedError('input should be a type of dic')
        rtn = {}
        for k, v in dic.items():
            if isinstance(v, torch.Tensor):
                rtn[k] = v.to(self.device)
            else:
                rtn[k] = v

        return rtn

    def update_container(self, dic, prob, trg, mask=None):

        """
        Only save current prediction & true target in container and
        compute metrics later at the end of epoch
        """
        if len(prob.size()) == 2:
            prob = prob.unsqueeze(0)
            trg = trg.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        for event_idx in range(prob.size(-1)):
            """
            select patient sequences only with y_true=1 exist
            (skip those don't have y_true=1)
            """
            p = prob[:, :, event_idx].contiguous().view(-1)
            t = trg[:, :, event_idx].contiguous().view(-1)
            if mask is not None:
                m = mask[:, :, event_idx].byte().contiguous().view(-1)
                p = torch.masked_select(p, m)
                t = torch.masked_select(t, m)
                    
            # move to cpu (they will be consumed by cpu eventually)
            p = p.to('cpu')
            t = t.to('cpu')

            # append event-specific sequences to container

            if self.macro_aucs:
                dic['y_probs'][event_idx] += p.tolist()
                dic['y_trues'][event_idx] += t.tolist()

            if self.micro_aucs:
                dic['y_ex_probs'][event_idx].append(p)
                dic['y_ex_trues'][event_idx].append(t)

        return dic

    def trad_update(self, dic, pred, trg, mask=None):
        """

        :param pred: seqlen x n_events
        :param trg: seqlen x n_events
        :param mask: seqlen x n_events
        :return:
        """
        dic['pred_count'] += pred.data.sum()
        correct = torch.sum(torch.mul(pred, trg))
        dic['total_exact_match'] += torch.equal(pred, trg)
        dic['total_match'] += (pred == trg).float().sum().data
        dic['total_gold'] += trg.sum().data  # recall
        dic['total_num'] += reduce(lambda x, y: x * y, trg.size())
        dic['total_correct'] += correct.data
        dic['total_pred'] += pred.sum().data  # prec
        dic['total_hamming_loss'] += (pred != trg).float().sum().data
        dic['total_cnt'] += 1

        if mask is not None:
            # when mask is provided, remove number of non-active entries in
            # the mask on the statsitics (^1 : invert op)
            num_zero_items = (mask.byte() ^ 1).sum().data.float()

            if pred.is_cuda:
                device = pred.get_device()
                num_zero_items = num_zero_items.to(device)

            dic['total_match'] -= num_zero_items
            dic['total_num'] -= num_zero_items

        return dic

    def _get_zeroone_mask(self, _trg, inp_first_step):

        if len(_trg.size()) == 2:
            raise NotImplementedError

        assert len(_trg.size()) == 3

        n_b, n_s, n_e = _trg.size()

        next_steps = _trg[:, :, :]
        prev_steps = torch.cat((inp_first_step, _trg[:, :-1, :]), dim=1)
        first_steps = _trg[:, 0, :].unsqueeze(1)
        zten = torch.zeros(n_b, 1, n_e).byte()

        zz_mask = (prev_steps == 0) & (next_steps == 0)
        zo_mask = (prev_steps == 0) & (next_steps == 1)
        oz_mask = (prev_steps == 1) & (next_steps == 0)
        oo_mask = (prev_steps == 1) & (next_steps == 1)

        # r_mask = (next_steps == prev_steps) & (next_steps == 1)
        # nr_mask = (next_steps != prev_steps) & (next_steps == 1)

        # if self.use_cuda:
        #     zten = zten.mask(self.device)

        # r_mask = torch.cat([zten, r_mask], dim=1)
        # nr_mask = torch.cat([first_steps, nr_mask], dim=1)

        return zz_mask.float(), zo_mask.float(), oz_mask.float(), oo_mask.float()

    def _get_interval_mask(self, _trg, inp_first_step):

        if len(_trg.size()) == 2:
            raise NotImplementedError

        assert len(_trg.size()) == 3

        n_b, n_s, n_e = _trg.size()

        ac = torch.zeros(n_b, n_e).to(_trg.device)
        ec = torch.zeros(n_b, n_e).to(_trg.device)

        ac += inp_first_step.squeeze(1)

        mask_steps = []
        for step in range(n_s):
            input_step = _trg[:, step, :]
            ac = ac + input_step
            mask_ec = input_step * (ec + 1) # +1 to ec to avoid being zero

            # should update ec after update mask_ec
            ec = (ac > 0).float() * (ec + 1) * \
                (input_step == 0).float()

            # should subtract 1 later when interpret the mask's value
            mask_steps.append(mask_ec)
        
        mask_steps = torch.stack(mask_steps, dim=1).to(_trg.device)

        # remove the first occurrence on each event
        ac = torch.zeros(n_b, n_e).to(_trg.device)

        for step in range(n_s):
            ac = ac + mask_steps[:, step, :]
            mask_steps[:, step, :] *= (ac != 1)

        mask_int_0 = (mask_steps == 1).float()
        mask_int_1 = (mask_steps == 2).float()
        mask_int_2 = (mask_steps == 3).float()
        mask_int_3 = (mask_steps == 4).float()
        mask_int_4 = (mask_steps == 5).float()
        mask_int_5 = (mask_steps == 6).float()

        return mask_int_0, mask_int_1, mask_int_2, mask_int_3, mask_int_4, mask_int_5

    def _get_when_occur_mask(self, _trg, _inp_first_step, _trg_len_mask):
        """
        This function returns masks which fills index values n until n-th step.

        e.g.
        _trg:  [0,1,0,0,0,1,0,1,0]
        accum: [0,0,1,1,1,1,2,2,3]

        """
        # _trg = torch.FloatTensor([[[0,1,0,0,1,0,1,0,1],[0,1,0,0,1,0,0,0,0]],
        #      [[0,1,0,0,1,0,1,0,0],[0,0,0,0,0,0,0,0,0]]]).permute(0,2,1).cuda()

        if len(_trg.size()) == 2:
            raise NotImplementedError

        n_b, n_s, n_e = _trg.size()

        accum = copy.deepcopy(_trg).to(_trg.device)
        # cnt = torch.zeros(n_b, n_e).to(_trg.device)
        # NOTE: initialize cnt with the first step on input.
        cnt = _inp_first_step.squeeze(1)

        for step in range(n_s):
            accum[:, step, :] = cnt
            cnt += _trg[:, step, :]

        # accum.cpu().numpy()

        # detect trailing zero
        def mask_trailing_zeros(_trg):
            n_b, n_s, n_e = _trg.size()
            _cnt = torch.zeros(n_b, n_e).to(_trg.device)
            _accum = torch.zeros(_trg.size()).to(_trg.device)
            trg_flipped = copy.deepcopy(_trg).flip(1)
            for step in range(n_s):
                _cnt += trg_flipped[:, step, :]
                _accum[:, step, :] = _cnt
            _accum = _accum.flip(1)
            return _accum == 0

        trailing_zeros = mask_trailing_zeros(_trg)

        trailing_zeros *= ~(trailing_zeros * _trg_len_mask).bool()

        first_mask = (accum == 0) * ~trailing_zeros
        second_mask = (accum == 1) * ~trailing_zeros
        second_later_mask = (accum >= 1) * ~trailing_zeros
        later_mask = (accum > 1) * ~trailing_zeros

        return first_mask.float(), second_mask.float(), second_later_mask.float(), later_mask.float()

    def _get_when_occur_mask_nth_step(self, _trg):
        """
        This function returns masks which has only n-th (1,2,3+) event
        occurrence is 1.

        """

        if len(_trg.size()) == 2:
            raise NotImplementedError

        n_b, n_s, n_e = _trg.size()

        accum = copy.deepcopy(_trg).to(_trg.device)
        cnt = torch.zeros(n_b, n_e).to(_trg.device)

        for step in range(n_s):
            accum[:, step, :] = cnt
            cnt += _trg[:, step, :]

        accum.cpu().numpy()

        # detect trailing zero
        def mask_trailing_zeros(_trg):
            n_b, n_s, n_e = _trg.size()
            _cnt = torch.zeros(n_b, n_e).to(_trg.device)
            _accum = torch.zeros(_trg.size()).to(_trg.device)
            trg_flipped = copy.deepcopy(_trg).flip(1)
            for step in range(n_s):
                _cnt += trg_flipped[:, step, :]
                _accum[:, step, :] = _cnt

            _accum = _accum.flip(1)
            return _accum == 0

        trailing_zeros = mask_trailing_zeros(_trg)

        first_mask = (accum == 0) * ~trailing_zeros
        second_mask = (accum == 1) * ~trailing_zeros
        later_mask = (accum > 1) * ~trailing_zeros

        steps = sorted(torch.unique(accum).tolist())
        step_masks = {step: ((accum == step) * ~trailing_zeros).float()
                      for step in steps}

        return step_masks

    def _get_start_zero_one_mask(self, zz_mask, zo_mask, oz_mask, oo_mask):

        start_zero_mask = zz_mask.long() | zo_mask.long()
        start_one_mask = oz_mask.long() | oo_mask.long()

        return start_zero_mask.float(), start_one_mask.float()

    def _get_event_type_mask(self, _trg, event_types, event_idxs):
        # event_idx: element indices of the event type
        masks = {}
        zten = torch.zeros(_trg.size())
        zten = zten.to(_trg.device)

        for etype in event_types:
            masks[etype] = copy.deepcopy(zten)
            for idx in event_idxs[etype]:

                if len(_trg.size()) == 2:
                    masks[etype][:, idx] = 1
                elif len(_trg.size()) == 3:
                    masks[etype][:, :, idx] = 1
        return masks

    def _update_step_mask(self, dic, pred, trg, mask=None, prob=None,
                          final=False, force_auroc=False):

        if (prob is not None) and (final or force_auroc):

            # NOTE: prediction & target update to container
            dic = self.update_container(dic, prob, trg, mask)

        if mask is not None:
            pred = mask * pred
            trg = mask * trg

        dic = self.trad_update(dic, pred, trg, mask)

        return dic

    def update(self, pred, trg, len_trg, base_step=None, final=False,
               prob=None, load_to_cpu=False, 
               skip_detail_eval=False, force_auroc=False,
               inp_first_step=None):

        if trg.dim() > pred.dim():
            trg = trg.squeeze(0)
        elif trg.dim() < pred.dim():
            pred = pred.squeeze(0)
            if prob is not None:
                prob = prob.squeeze(0)

        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        try:
            n_b, n_s, n_e = pred.size()
        except ValueError as e:
            raise e

        trg_len_mask = torch.FloatTensor(
            [[1] * l + [0] * (n_s - l) for l in len_trg.tolist()])

        trg_len_mask = trg_len_mask.to(pred.device)

        if trg_len_mask.size() == (n_s, n_b):
            trg_len_mask = trg_len_mask.permute(1, 0)

        trg_len_mask = trg_len_mask.expand(n_e, n_b, n_s)
        trg_len_mask = trg_len_mask.permute(1, 2, 0)

        # flat stats
        self.eval['flat'] = self._update_step_mask(
            self.eval['flat'], pred, trg, mask=trg_len_mask,
            prob=prob, final=final, force_auroc=force_auroc)
        self.eval['overall_num'] += trg_len_mask.sum()

        if final and not skip_detail_eval:

            # when occur? (MLHC2019)
            first_occur_mask, second_occur_mask, second_later_mask, later_occur_mask = \
                self._get_when_occur_mask(trg, inp_first_step, trg_len_mask)

            self.eval['first_occur_num'] += first_occur_mask.sum()
            self.eval['second_occur_num'] += second_occur_mask.sum()
            self.eval['later_occur_num'] += later_occur_mask.sum()

            first_occur_mask *= trg_len_mask
            second_occur_mask *= trg_len_mask
            second_later_mask *= trg_len_mask
            later_occur_mask *= trg_len_mask

            self.eval['first_occur'] = self._update_step_mask(
                self.eval['first_occur'], pred, trg, 
                mask=first_occur_mask,
                prob=prob,
                final=final)
            self.eval['second_occur'] = self._update_step_mask(
                self.eval['second_occur'], pred, trg,
                mask=second_occur_mask, 
                prob=prob,
                final=final)
            self.eval['second_later_occur'] = self._update_step_mask(
                self.eval['second_later_occur'], pred, trg,
                mask=second_later_mask, 
                prob=prob,
                final=final)
            self.eval['later_occur'] = self._update_step_mask(
                self.eval['later_occur'], pred, trg, 
                mask=later_occur_mask,
                prob=prob,
                final=final)
            
            # when occur n-step mask
            step_masks = self._get_when_occur_mask_nth_step(trg)

            for step, step_mask in step_masks.items():
                
                step_mask *= trg_len_mask

                if step not in self.eval['occur_steps']:
                    self.eval['occur_steps'][step] = copy.deepcopy(
                        self._metric)

                self.eval['occur_steps'][step] = self._update_step_mask(
                    self.eval['occur_steps'][step], pred, trg, 
                    mask=step_mask,
                    prob=prob,
                    final=final)

            # # interval  masks
            # interval_masks = self._get_interval_mask(trg, inp_first_step)
            # interval_masks = [x * trg_len_mask for x in interval_masks]
            # mask_int_0, mask_int_1, mask_int_2, mask_int_3, mask_int_4, mask_int_5 = interval_masks

            # self.eval['interval_0'] = self._update_step_mask(
            #     self.eval['interval_0'], pred, trg, mask=mask_int_0, prob=prob,
            #     final=final)
            # self.eval['interval_1'] = self._update_step_mask(
            #     self.eval['interval_1'], pred, trg, mask=mask_int_1, prob=prob,
            #     final=final)
            # self.eval['interval_2'] = self._update_step_mask(
            #     self.eval['interval_2'], pred, trg, mask=mask_int_2, prob=prob,
            #     final=final)
            # self.eval['interval_3'] = self._update_step_mask(
            #     self.eval['interval_3'], pred, trg, mask=mask_int_3, prob=prob,
            #     final=final)
            # self.eval['interval_4'] = self._update_step_mask(
            #     self.eval['interval_4'], pred, trg, mask=mask_int_4, prob=prob,
            #     final=final)
            # self.eval['interval_5'] = self._update_step_mask(
            #     self.eval['interval_5'], pred, trg, mask=mask_int_5, prob=prob,
            #     final=final)

            # zero-one masks
            zero_one_masks = self._get_zeroone_mask(trg, inp_first_step)
            masks = [x * trg_len_mask for x in zero_one_masks]
            zz_mask, zo_mask, oz_mask, oo_mask = zero_one_masks
            
            self.eval['zero_zero'] = self._update_step_mask(
                self.eval['zero_zero'], pred, trg, mask=zz_mask, prob=prob,
                final=final)
            self.eval['zero_one'] = self._update_step_mask(
                self.eval['zero_one'], pred, trg, mask=zo_mask, prob=prob,
                final=final)
            self.eval['one_zero'] = self._update_step_mask(
                self.eval['one_zero'], pred, trg, mask=oz_mask, prob=prob,
                final=final)
            self.eval['one_one'] = self._update_step_mask(
                self.eval['one_one'], pred, trg, mask=oo_mask, prob=prob,
                final=final)

            # start one / zero
            z_mask, o_mask = self._get_start_zero_one_mask(zz_mask, zo_mask, oz_mask, oo_mask)
            z_mask *= trg_len_mask
            o_mask *= trg_len_mask
            self.eval['start_zero'] = self._update_step_mask(
                self.eval['start_zero'], pred, trg, mask=z_mask, prob=prob,
                final=final)
            self.eval['start_one'] = self._update_step_mask(
                self.eval['start_one'], pred, trg, mask=o_mask, prob=prob,
                final=final)

            # event category stats
            if self.event_dic \
                    and 'category' in list(self.event_dic.values())[0]:
                e_masks = self._get_event_type_mask(trg, self.event_types,
                                                    self.event_idxs)
                for category in self.event_types:

                    # category overall
                    self.eval['etypes'][category] = self._update_step_mask(
                        self.eval['etypes'][category], pred, trg,
                        mask=e_masks[category] * trg_len_mask, prob=prob,
                        final=final)

                    # category AND zero-zero
                    self.eval['etypes_zz'][category] = self._update_step_mask(
                        self.eval['etypes_zz'][category], pred, trg,
                        mask=e_masks[category] * zz_mask, prob=prob, final=final)

                    # category AND zero-one
                    self.eval['etypes_zo'][category] = self._update_step_mask(
                        self.eval['etypes_zo'][category], pred, trg,
                        mask=e_masks[category] * zo_mask, prob=prob, final=final)

                    # category AND one-zero
                    self.eval['etypes_oz'][category] = self._update_step_mask(
                        self.eval['etypes_oz'][category], pred, trg,
                        mask=e_masks[category] * oz_mask, prob=prob, final=final)

                    # category AND one-one
                    self.eval['etypes_oo'][category] = self._update_step_mask(
                        self.eval['etypes_oo'][category], pred, trg,
                        mask=e_masks[category] * oo_mask, prob=prob, final=final)

                    # category AND start-one
                    self.eval['etypes_start_one'][category] = self._update_step_mask(
                        self.eval['etypes_start_one'][category], pred, trg,
                        mask=e_masks[category] * o_mask, prob=prob, final=final)

                    # category AND start-zero
                    self.eval['etypes_start_zero'][category] = self._update_step_mask(
                        self.eval['etypes_start_zero'][category], pred, trg,
                        mask=e_masks[category] * z_mask, prob=prob, final=final)

                    # category AND occur first
                    self.eval['etypes_first_occur'][category] = self._update_step_mask(
                        self.eval['etypes_first_occur'][category], pred, trg,
                        mask=e_masks[category] * first_occur_mask, prob=prob, final=final)
                    
                    # category AND occur second
                    self.eval['etypes_second_occur'][category] = self._update_step_mask(
                        self.eval['etypes_second_occur'][category], pred, trg,
                        mask=e_masks[category] * second_occur_mask, prob=prob, final=final)
                    
                    # category AND occur later
                    self.eval['etypes_later_occur'][category] = self._update_step_mask(
                        self.eval['etypes_later_occur'][category], pred, trg,
                        mask=e_masks[category] * later_occur_mask, prob=prob, final=final)



            # step specific
            for step in range(n_s):
                if base_step is not None:
                    step += base_step
                if step not in self.eval['tstep']:
                    self.eval['tstep'][step] = copy.deepcopy(self._metric)
                if step >= prob.size(1):  # prevent step += base_step overflow
                    continue
                try:
                    _prob = prob[:, step, :] if prob is not None else None
                except IndexError as e:
                    raise e
                _trg = trg[:, step, :]
                _pred = pred[:, step, :]
                self.eval['tstep'][step] = self._update_step_mask(
                    self.eval['tstep'][step], _pred,
                    _trg, prob=_prob, final=final)

                # event category stats AND timestep
                if self.event_dic \
                        and 'category' in list(self.event_dic.values())[0]:
                    e_masks = self._get_event_type_mask(_trg, self.event_types,
                                                        self.event_idxs)
                                                        
                    if step not in self.eval['tstep-etype']:
                        self.eval['tstep-etype'][step] = {category: copy.deepcopy(
                            self._metric) for category in self.event_types}

                    for category in self.event_types:
                        self.eval['tstep-etype'][step][category] = self._update_step_mask(
                            self.eval['tstep-etype'][step][category], 
                            _pred, _trg,
                            mask=e_masks[category], #* trg_len_mask, 
                            prob=_prob,
                            final=final)

                # repeat / nonrepeat AND timestep
                if step not in self.eval['tstep-repeat']:
                    self.eval['tstep-repeat'][step] = {
                        'first_occur': copy.deepcopy(self._metric),
                        'second_occur': copy.deepcopy(self._metric),
                        'second_later_occur': copy.deepcopy(self._metric),
                        'later_occur': copy.deepcopy(self._metric),
                    }
                self.eval['tstep-repeat'][step]['first_occur'] = self._update_step_mask(
                    self.eval['tstep-repeat'][step]['first_occur'], _pred, _trg,
                    mask=first_occur_mask[:, step, :],
                    prob=_prob,
                    final=final)
                self.eval['tstep-repeat'][step]['second_occur'] = self._update_step_mask(
                    self.eval['tstep-repeat'][step]['second_occur'], _pred, _trg,
                    mask=second_occur_mask[:, step, :],
                    prob=_prob,
                    final=final)
                self.eval['tstep-repeat'][step]['second_later_occur'] = self._update_step_mask(
                    self.eval['tstep-repeat'][step]['second_later_occur'], _pred, _trg,
                    mask=second_later_mask[:, step, :],
                    prob=_prob,
                    final=final)
                self.eval['tstep-repeat'][step]['later_occur'] = self._update_step_mask(
                    self.eval['tstep-repeat'][step]['later_occur'], _pred, _trg,
                    mask=later_occur_mask[:, step, :],
                    prob=_prob,
                    final=final)


    @staticmethod
    def compute(dic, epoch, web_logger=None, test_name='', tstep=False,
                avg='micro', cv_str='', option_str='', verbose=True,
                final=False, export_csv=False, export_probs=True, event_dic=None, 
                csv_file=None, return_metrics=False, use_multithread=True, 
                force_auroc=False, force_plot_auroc=False, do_plot=False, 
                target_auprc=False):
        total_exact_match = (dic['total_exact_match'] * 100.0 + 1e-09) / \
                            (dic['total_cnt'] + 1e-09)
        total_hamming_loss = (
                dic['total_hamming_loss'] / dic['total_num']).item()
        total_correct = dic['total_correct'][0]
        total_match = dic['total_match'][0]
        total_pred = dic['total_pred'][0]
        total_gold = dic['total_gold'][0]
        total_num = dic['total_num'][0]
        if total_pred > 0:
            prec = 100.0 * total_correct / total_pred
        else:
            prec = 0
        if total_gold > 0:
            recall = 100.0 * total_correct / total_gold
        else:
            recall = 0
        if prec + recall > 0:
            f1 = 2 * prec * recall / (prec + recall)
        else:
            f1 = 0
        if total_num > 0:
            acc = 100.0 * total_match / total_num
        else:
            acc = 0

        micro_prior = total_gold / total_num

        container = {}

        mac_auprc, mac_auroc, mac_ap, mi_auprc, mi_auroc, mi_ap, \
        mac_auprc_std, mac_auroc_std, mac_ap_std = [0.0] * 9
        less_point_five_auroc = 0  # count events with AUROC less than 0.5

        if 'y_probs' in dic:

            # macro averaged auroc, auprc, and ap

            """
            Multithread
            """

            def _auprc_auroc_ap_wrapper(i, probs, trues, auroc_only):
                spr, sro, sap, sprec, srecall, sspec, stn, sfp, sfn, stp, sacc \
                    = get_auprc_auroc_ap(trues, probs, auroc_only=auroc_only)
                return {i: spr}, {i: sro}, {i: sap}, {i: sprec}, {i: srecall}, \
                    {i: sspec}, {i: stn}, {i: sfp}, {i: sfn}, {i: stp}, {i: sacc}

            num_cores = 40
            output = list(zip(*Parallel(n_jobs=num_cores, prefer="threads")(
                delayed(_auprc_auroc_ap_wrapper)(
                    i,
                    dic['y_probs'][i],
                    dic['y_trues'][i],
                    auroc_only=force_auroc and (not final) and (not target_auprc),
                ) \
                for i in dic['y_probs'].keys() \
                if (len(dic['y_trues'][i]) > 0
                    and dic['y_probs'][i] != []
                    and dic['y_trues'][i] != []
                    )
            )))

            if output == []:
                mac_auprcs, mac_aurocs, mac_aps, mac_precs, mac_recs, \
                    mac_specs, mac_tn, mac_fp, mac_fn, mac_tp, mac_acc = [{0: 0}] * 11
            else:
                (prcs, rocs, aps, precisions, recalls, specificities, tns, fps, fns, tps, accs) = output
                # mac_auprcs = {list(x.keys())[0]: list(x.values())[0] for x in prcs \
                #                 if not np.isnan(list(x.values())[0])}
                mac_auprcs = {list(x.keys())[0]: list(x.values())[0] for x in prcs
                                if list(x.values())[0] is not None}
                mac_aurocs = {list(x.keys())[0]: list(x.values())[0] for x in rocs \
                                if list(x.values())[0] is not None}
                mac_aps = {list(x.keys())[0]: list(x.values())[0] for x in aps \
                            if not np.isnan(list(x.values())[0])}
                mac_precs = {list(x.keys())[0]: list(x.values())[0] for x in precisions \
                            if not np.isnan(list(x.values())[0])}
                mac_recs = {list(x.keys())[0]: list(x.values())[0] for x in recalls \
                            if not np.isnan(list(x.values())[0])}
                mac_specs = {list(x.keys())[0]: list(x.values())[0] for x in specificities \
                            if not np.isnan(list(x.values())[0])}
                mac_tn = {list(x.keys())[0]: list(x.values())[0] for x in tns
                            if not np.isnan(list(x.values())[0])}
                mac_fp = {list(x.keys())[0]: list(x.values())[0] for x in fps
                            if not np.isnan(list(x.values())[0])}
                mac_fn = {list(x.keys())[0]: list(x.values())[0] for x in fns
                            if not np.isnan(list(x.values())[0])}
                mac_tp = {list(x.keys())[0]: list(x.values())[0] for x in tps
                            if not np.isnan(list(x.values())[0])}
                mac_acc = {list(x.keys())[0]: list(x.values())[0] for x in accs
                            if not np.isnan(list(x.values())[0])}

            if export_csv:
                # merge two dicts by sum values of same key
                pos_points = dict(Counter(mac_tp) + Counter(mac_fn))
                neg_points = dict(Counter(mac_fp) + Counter(mac_tn))
                export_event_metrics(
                    csv_file, event_dic,
                    {
                        'mac_auprc': mac_auprcs,
                        'mac_auroc': mac_aurocs,
                        'mac_aps': mac_aps,
                        'mac_prec' : mac_precs,
                        'mac_recs' : mac_recs,
                        'mac_specs' : mac_specs,
                        'tn': mac_tn,
                        'fp': mac_fp,
                        'fn': mac_fn,
                        'tp': mac_tp,
                        'pos_n_points': pos_points,
                        'neg_n_points': neg_points,
                        'acc': mac_acc,
                    }
                )

            # if logger:
            #     logger.log_histogram_3d(mac_auprcs.values(), name='auprc', step=epoch)
            #     logger.log_histogram_3d(mac_aurocs.values(), name='auroc', step=epoch)
            #     logger.log_histogram_3d(mac_precs.values(), name='precision', step=epoch)
            #     logger.log_histogram_3d(mac_specs.values(), name='specificity', step=epoch)
            #     logger.log_histogram_3d(mac_recs.values(), name='recall', step=epoch)

                ## This outputs actual prediction and target into csv
                ## So it takes lots of space in disk. used only when debug is need.
                # export_event_pred_target(
                #     csv_file, 
                #     event_dic, 
                #     {
                #         'mac_auprc': mac_auprcs,
                #         'mac_auroc': mac_aurocs,
                #         'mac_prec' : mac_precs,
                #         'mac_recs' : mac_recs,
                #         'mac_specs' : mac_specs,
                #         'tn': mac_tn,
                #         'fp': mac_fp,
                #         'fn': mac_fn,
                #         'tp': mac_tp,
                #         'pos_n_points': pos_points,
                #         'neg_n_points': neg_points,
                #         'acc': mac_acc,
                #     }, 
                #     dic['y_probs'], 
                #     dic['y_trues']
                # )

            if final or force_auroc:
                for event_id, event_auroc in mac_aurocs.items():
                    if not np.isnan(event_auroc) and event_auroc < 0.5:
                        less_point_five_auroc += 1

                    # Plot AUROC / AUPRC figures
                    if (do_plot and final and event_dic is not None 
                            and event_id in event_dic 
                            and csv_file is not None
                        ):

                        y_true = dic['y_trues'][event_id]
                        y_prob = dic['y_probs'][event_id]

                        # NOTE: event_id in event_dic starts from 1. 
                        event_name = event_dic[event_id + 1]["label"]
                        event_category = event_dic[event_id + 1]["category"]

                        _precision, _recall, thresholds = \
                            precision_recall_curve(y_true, y_prob, pos_label=1)
                        auprc = auc(_recall, _precision)

                        fpr, tpr, _ = roc_curve(y_true, y_prob,
                            sample_weight=None,
                            pos_label=1)
                        tr_test = 'train' if '_train.csv' in csv_file else 'test'
                        f_path = '/'.join(csv_file.split('/')[:-1]) + '/plots'
                        os.system("mkdir -p {}".format(f_path))

                        event_name = event_name.replace('/','_')
                        event_name = event_name.replace('[', '_')
                        event_name = event_name.replace(']', '_')
                        event_name = event_name.replace(' ', '_')

                        f_path += '/{}plot_auroc_{:.2f}_auprc_{:.2f}_e_{}_{}_{}.png'.format(
                            tr_test, event_auroc, auprc, event_id, event_name, event_category
                        )
                        draw_roc_curve(
                            event_category+' '+event_name,
                            fpr, tpr, event_auroc, 
                            _precision, _recall, auprc,
                            f_path
                        )   

            # Macro averagings

            logger.info('cnt nan (auprc) : {}'.format(sum(np.isnan(list(mac_auprcs.values())))))
            logger.info('cnt nan (auroc) : {}'.format(sum(np.isnan(list(mac_aurocs.values())))))

            mac_auprc = np.nanmean(list(mac_auprcs.values())) * 100
            mac_auroc = np.nanmean(list(mac_aurocs.values())) * 100
            mac_ap = np.nanmean(list(mac_aps.values())) * 100

            # precision, recall, specificity
            mac_prec = np.nanmean(list(mac_precs.values())) * 100
            mac_rec = np.nanmean(list(mac_recs.values())) * 100
            mac_spec = np.nanmean(list(mac_specs.values())) * 100

            mac_auprc_std = np.nanstd(list(mac_auprcs.values()))
            mac_auroc_std = np.nanstd(list(mac_aurocs.values()))
            mac_ap_std = np.nanstd(list(mac_aps.values()))

            mac_prec_std = np.nanstd(list(mac_precs.values())) * 100
            mac_rec_std = np.nanstd(list(mac_recs.values())) * 100
            mac_spec_std = np.nanstd(list(mac_specs.values())) * 100

            mac_acc = np.nanmean(list(mac_acc.values())) * 100

            # Weighted Averagings

            occur_events_num = {k : sum(list(v)) for k, v in dic['y_trues'].items()}

            all_occurs = sum(list(occur_events_num.values())) + 1e-10
            weight_events = {k: v/all_occurs for k, v in occur_events_num.items()}

            wgh_auprc = sum([val * weight_events[idx] for idx,
                             val in mac_auprcs.items() if ~np.isnan(val)]) * 100
            wgh_auroc = sum([val * weight_events[idx] for idx, 
                             val in mac_aurocs.items() if ~np.isnan(val)]) * 100

            if return_metrics:
                container = {
                    'mac_auprc': mac_auprc,
                    'mac_auroc': mac_auroc,
                    'mac_ap': mac_ap,
                    'mac_auprc_std': mac_auprc_std,
                    'mac_auroc_std': mac_auroc_std,
                    'mac_ap_std': mac_ap_std,
                    'mac_prec': mac_prec,
                    'mac_rec': mac_rec,
                    'mac_spec': mac_spec,
                    'mac_prec_std': mac_prec_std,
                    'mac_rec_std': mac_rec_std,
                    'mac_spec_std': mac_spec_std,
                    'wgh_auprc': wgh_auprc,
                    'wgh_auroc': wgh_auroc,
                    'events_mac_auprcs': mac_auprcs,
                    'events_mac_aurocs': mac_aurocs,
                }

        if 'y_ex_probs' in dic:
            # micro average
            mi_auprc, mi_auroc, mi_ap = [], [], []
            for i in range(len(dic['y_ex_probs'])):
                prob_exs, trg_exs = dic['y_ex_probs'][i], dic['y_ex_trues'][i]
                for prb_ex, trg_ex in zip(prob_exs, trg_exs):
                    prb_ex, trg_ex = prb_ex.cpu().numpy(), trg_ex.cpu().numpy()
                    if len(trg_ex) > 0:
                        try:
                            pr, ro, ap, sprec, srecall, sspec, tns, fps, fns, \
                                tps, accs = get_auprc_auroc_ap(trg_ex, prb_ex)
                            if not np.isnan(pr):
                                mi_auprc.append(pr)
                            if ro is not None:
                                mi_auroc.append(ro)
                            if not np.isnan(ap):
                                mi_ap.append(ap)
                        except IndexError:
                            # for nan-like error on sklearn
                            continue
            mi_auprc = np.nanmean(mi_auprc) * 100
            mi_auroc = np.nanmean(mi_auroc) * 100
            mi_ap = np.nanmean(mi_ap) * 100

        if 'auprc' in dic and final:
            # fast micro average
            mi_auprc = np.nanmean(dic['auprc'])
            mi_auroc = np.nanmean(dic['auroc'])
            mi_ap = np.nanmean(dic['ap'])

        if 'mapmeter' in dic and final:
            try:
                mi_auprc = dic['mapmeter'].value() * 100
            except AttributeError:
                mi_auprc = 0
        if 'aucmeter' in dic and final:
            try:
                mi_auroc = np.nanmean([dic['aucmeter'][i].value()[0]
                                       for i in list(dic['aucmeter'].keys())]) * 100
            except AttributeError:
                mi_auprc = 0
        # torch.cat(dic['y_trues'].values()).cpu().numpy()

        # 'EMA', '{:5.4f}%'.format(total_exact_match)
        # 'HamL', '{:5.4f}'.format(total_hamming_loss)
        if verbose:
            logger.info("\n"+tabulate(
                [['#TP', int(total_correct), 'Pr', '{:5.2f}%'.format(prec)],
                 ['#Pred', int(total_pred), 'Rc', '{:5.2f}%'.format(recall)],
                 ['#Gold', int(total_gold), 'F1', '{:5.2f}%'.format(f1)],
                 ['#Elements', int(total_num), 'Acc', '{:5.2f}%'.format(acc)],
                 ['AUPRC(ma)', '{:5.2f}% ({:5.4f})'.format(
                     mac_auprc, mac_auprc_std),
                  'AUPRC(mi)', '{:5.2f}%'.format(mi_auprc)],
                 ['AUROC(ma)', '{:5.2f}% ({:5.4f})'.format(mac_auroc,
                                                           mac_auroc_std),
                  'AUROC(mi)', '{:5.2f}%'.format(mi_auroc)],
                 ['AUPRC(wg)', '{:5.2f}%'.format(wgh_auprc),
                  'AUROC(wg)', '{:5.2f}%'.format(wgh_auroc)],
                 ['lpfAUROC', less_point_five_auroc, 
                    'mac_acc', '{:5.2f}%'.format(mac_acc)]
                 ],
                tablefmt='psql', ))
                #  ['AP(ma)', '{:5.2f}% ({:5.4f})'.format(
                #      mac_ap, mac_ap_std),
                #   'AP(mi)', '{:5.2f}%'.format(mi_ap)],

        if web_logger:
            tstep_str = ' time' if tstep else ''

            # Removed list:
            # - 'hml', 'auprc', 'auroc', 'ap', 'ema',
            # - total_exact_match, total_hamming_loss, mi_auprc, mi_auroc, mi_ap

            for met_name, score in zip(
                    ['precision', 'recall', 'f1', 'acc', 'micro_prior'],
                    [prec, recall, f1, acc, micro_prior]
            ):
                if type(score) == torch.Tensor:
                    score = score.cpu()
                web_logger.log_metric("{}: {}({}) {}{}{}".format(
                    test_name, met_name, avg, cv_str, tstep_str, option_str),
                    score, step=epoch)

            # Removed list:
            # - 'ap', 'ap_std', 'auprc_std', 'auroc_std',
            # - mac_ap, mac_ap_std, mac_auprc_std, mac_auroc_std,

            for met_name, score in zip(
                    [
                        'auprc', 'auroc', 'wgh_auprc', 'wgh_auroc',
                        'less_point_five_auroc'
                    ],
                    [
                        mac_auprc, mac_auroc, wgh_auprc, wgh_auroc,
                        less_point_five_auroc
                    ]
                ):
                if type(score) == torch.Tensor:
                    score = score.cpu()

                web_logger.log_metric("{}: {}({}) {}{}{}".format(
                    test_name, met_name, 'macro', cv_str, tstep_str,
                    option_str),
                    score, step=epoch)

        if return_metrics:
            return f1, acc, container

        return f1, acc


def get_auprc_auroc_ap(y_true, y_prob, skip_ap=True, auroc_only=False):
    if auroc_only:
        auroc = fast_auroc(y_true, y_prob)
        auprc, ap, prec, recall, specificity, tn, fp, fn, tp, acc = [0] * 10

    else:
        precision, recall, thresholds \
            = precision_recall_curve(y_true, y_prob, pos_label=1)

        auprc = auc(recall, precision)

        # calculate average precision score
        if skip_ap:
            ap = 0
        else:
            ap = average_precision_score(y_true, y_prob)
            
        try:
            # auroc = roc_auc_score(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob,
                                    sample_weight=None,
                                    pos_label=1)
            auroc = auc(fpr, tpr)          

        except ValueError:
            auroc = None
        
        # logger.info('y_prob: {} {}'.format(len(y_prob), y_prob))
        # logger.info('y_true: {} {}'.format(len(y_true), y_true))

        y_pred = (np.array([y_prob]) > 0.5).astype('float').flatten()

        recall = recall_score(y_true, y_pred, pos_label=1, average='binary')
        prec = precision_score(y_true, y_pred, pos_label=1, average='binary')

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except ValueError:
            tn, fp, fn, tp = -1, -1, -1, -1

        specificity = tn / (tn+fp)
        acc = (tp + tn) / sum([tn, fp, fn, tp])

    return auprc, auroc, ap, prec, recall, specificity, tn, fp, fn, tp, acc


def export_event_metrics(filename, event_dic, metrics_dic):
    np.save(filename.replace('.csv', '_metric_dic.npy'), metrics_dic)
    with open(filename, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # write header row
        csv_writer.writerow(
            ["category", "label"] + list(metrics_dic.keys())
        )

        # NOTE: in event_dic, event id starts from 1.  (so k is)
        #       in the multi-hot vector, event id starts from 0. (so keys in metrics_dic)
        #       So transpose (-1) needed.
        for k, v in event_dic.items():
            csv_writer.writerow(
                [v["category"], v["label"], ]
                + ['{:.8f}'.format(metric[k-1]) for metric in
                   list(metrics_dic.values()) if k-1 in metric]
            )


def export_event_pred_target(filename, event_dic, metrics_dic, pred, target):
    
    filename = filename.replace('.csv', '_prob_target.csv')

    with open(filename, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"')

        # write header row
        csv_writer.writerow(
            ["category", "label", "type"] + list(metrics_dic.keys())
        )

        for k, v in event_dic.items():
            if k in pred and k in target:

                pred_event = np.asarray(list(pred[k-1]))
                target_event = np.asarray(list(target[k-1]))
                
                sorted_probs_idxs = np.argsort(-pred_event)

                pred_event = pred_event[sorted_probs_idxs]
                target_event = target_event[sorted_probs_idxs]

                if k-1 in metrics_dic["mac_auroc"] and metrics_dic["mac_auroc"][k-1] < 0.3:

                    csv_writer.writerow(
                        [v["category"], v["label"], "prob", ]
                        + ['{:.8f}'.format(metric[k-1]) for metric in
                        list(metrics_dic.values()) if k-1 in metric]
                        + ['{:.8f}'.format(x) for x in pred_event]
                    )
                    csv_writer.writerow(
                        [v["category"], v["label"], "target",]
                        + ['{:.8f}'.format(metric[k-1]) for metric in
                        list(metrics_dic.values()) if k-1 in metric]
                        + ['{:.8f}'.format(x) for x in target_event]
                    )


def draw_roc_curve(event_name, fpr, tpr, auroc, precision, recall, auprc, fname):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot(recall, precision, color='navy',
             lw=lw, label='PR curve (area = %0.2f)' % auprc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate // Precision')
    plt.ylabel('True Positive Rate // Recall')
    plt.title('ROC: {}'.format(event_name))
    plt.legend(loc="lower right")
    plt.savefig(fname, format='png')


def export_timestep_metrics(filename, model_name, metrics, event_dic=None):
    """

    :param filename: string csv file name
    :param model_name: string model name (1nd column name)
    :param metrics: list of dictionary of metrics over time series.
        order of elements in list represents order in time series
        e.g. [ {'metrics_one': 0.1, 'metrics_two': 0.1}, ... ]
    """
    with open(filename, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # NOTE: manually inserting metric names
        metric_names = ["mac_auprc", "mac_auprc_std", "mac_auroc",
                        "mac_auroc_std"]
        # metric_names = metrics[0].keys()

        # write header row
        csv_writer.writerow([model_name] + metric_names)

        for step, metric_step in enumerate(metrics):

            csv_writer.writerow(
                [str(step), ]
                + ['{:.4f}'.format(metric_step[name]) for name in metric_names]
            )
    # Event-Type-Specific Results
    if event_dic is not None:
        for metric_name in ['events_mac_auprcs', 'events_mac_aurocs']:
            _filename = filename.replace('.csv', '_spec_{}.csv'.format(metric_name))

            with open(_filename, mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',',
                                        quotechar='"', quoting=csv.QUOTE_MINIMAL)

                csv_writer.writerow(
                    ['steps', ] + [event_dic[itemidx+1]["category"] + "--" + event_dic[itemidx+1]["label"] 
                    for itemidx in metric_step[metric_name].keys()]
                )

                for step, metric_step in enumerate(metrics):

                    csv_writer.writerow(
                        [str(step), ] + ['{:.8f}'.format(metric) for itemidx, metric in
                            metric_step[metric_name].items()]
                    )




def compute_auroc_multievent(y_true, y_pred):
    """
    y_pred, y_true : n_instances x n_events
    return event-specific auroc score
    """
    _, idx = y_pred.sort(dim=0)
    y_true = torch.gather(y_true, 0, idx).float()
    nfalse = torch.zeros(y_true.size(-1)).to(y_true.device)
    auc = torch.zeros(y_true.size(-1)).to(y_true.device)
    n = y_true.size(0)
    for i in range(n):
        y_i = y_true[i, :]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def _test_compute_auroc():
    y_pred = torch.rand(10, 4)
    y_true = (torch.rand(10, 4) > 0.5).float()
    auc_0 = compute_auroc_multievent(y_true, y_pred)
    logger.info(('y_pred: {}'.format(y_pred)))
    logger.info(('y_true: {}'.format(y_true)))
    for i in range(4):
        y = y_true[:, i].numpy()
        p = y_pred[:, i].numpy()
        t1 = auc_0[i]
        t2 = roc_auc_score(y, p)
        t3 = fast_auroc(y, p)
        logger.info(('{:.8f}, {:.8f}, {:.8f}'.format(t1, t2, t3)))


@jit
def fast_auroc(y_true, y_prob):
    # https://www.ibm.com/developerworks/community/blogs/
    # jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
    y_true = np.asarray(y_true)

    if len(np.unique(y_true)) != 2:
        # raise ValueError("Only one class present in y_true. ROC AUC score "
                        #  "is not defined in that case.")
        return np.nan

    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def test():
    a = (torch.rand(3, 4, 5) > 0).float().cuda()
    b = (torch.rand(3, 4, 5) > 0).float().cuda()
    device = a.get_device()

    eval = MultiLabelEval(5, use_cuda=True, device=device)
    eval.update(a, b, [4, 3, 2])
    eval.compute(eval.eval['flat'], 0)
    for step in range(len(eval.eval['tstep'])):
        logger.info(('step: {}'.format(step)))
        eval.compute(eval.eval['tstep'][step], 0)


