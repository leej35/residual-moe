# -*- coding: utf-8 -*-
"""
07 Oct 2017
A collection of utility packages
"""

# Import statements

import sys
import numpy as np
import time
import logging
from datetime import datetime
import os
import dill

import multiprocessing
# multiprocessing.set_start_method('spawn', True)
from utils.tensor_utils import to_multihot_sorted_vectors

import pickle as pickle
import torch
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)


try:
    from apex import amp  # noqa: F401

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex
    

def grad_info(t):
    logger.info(f"is_leaf:{t.is_leaf} grad_fn:{t.grad_fn} requires_grad:{t.requires_grad}")

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def count_parameters(model):
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad)


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return list(repackage_hidden(v) for v in h)

def batch_apply(fn, *inputs):
    """
    by lucasb-eyer: https://discuss.pytorch.org/t/operations-on-multi-dimensional-tensors/2548/3?u=jel158
    :param fn:
    :param inputs:
    :return:
    usage:

    """
    return torch.stack([fn(*(a[0] for a in args)) for args in zip(*(inp.split(1) for inp in inputs))])


def slurp_file(filename):
    with open(filename, 'r') as infile:
        return infile.read()


def check_file_exist(filename):
    if not os.path.exists(filename):
        raise ValueError('The file {} does not exist'.format(filename))


def read_npy(filename):
    with open(filename, 'rb') as infile:
        result = np.load(infile)
        try:
            return result.item()
        except:
            return result


def tprint(message):
    """A quick method to print messages prepended with time information"""
    logger.info('[{}]{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], message))


def setup_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt='[%(asctime)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class Timing(object):
    """A context manager that prints the execution time of the block it manages"""

    def __init__(self, message, file=sys.stdout, logger=None, one_line=True):
        self.message = message
        if logger is not None:
            self.default_logger = False
            self.one_line = False
            self.logger = logger
        else:
            self.default_logger = True
            self.one_line = one_line
            self.logger = None
        self.file = file

    def _log(self, message, newline=True):
        if self.default_logger:
            print(message, end='\n' if newline else '', file=self.file)
            try:
                self.file.flush()
            except:
                pass
        else:
            self.logger.info(message)

    def __enter__(self):
        self.start = time.time()
        self._log(self.message, not self.one_line)

    def __exit__(self, exc_type, exc_value, traceback):
        self._log('{}Done in {:.3f}s'.format('' if self.one_line else self.message, time.time()-self.start))


def save_obj(obj, filename):
    with open(filename, 'wb') as output:
        dill.dump(obj, output)


def load_obj(filename):
    with open(filename, 'rb') as inputf:
        return dill.load(inputf)

def cp_save_obj(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def cp_load_obj(filename):
    with open(filename, 'rb') as inputf:
        if sys.version_info.major == 3:
            return pickle.load(inputf, encoding='latin1')
        else:
            return pickle.load(inputf)


def flat_tensor(tensor):
    return torch.cat([x.view(-1, x.size(-1)) for x in tensor], dim=0)


def topk_pred(event_counter, inp_size, threshold):
    _, topk_idx = torch.topk(event_counter, k=threshold, dim=2)

    pred = torch.zeros(inp_size)
    for ti in range(topk_idx.size(0)):
        for tj in range(topk_idx.size(1)):
            for tk in range(topk_idx.size(2)):
                idx = topk_idx[ti][tj][tk]
                pred[ti][tj][idx] = 1
    return pred


def masked_loss(loss_fn, pred, trg):
    """
    :param loss_fn: original loss function to use
    :param pred: prediction tensor size of (n_batch x max_seq_len x event_size)
    :param trg:  target tensor size of (n_batch x max_seq_len x event_size)
    :param trg_len: target sequence sizes
    :return: loss value over batch elements
    """
    repac_flag = False
    if type(pred) == list and pred[1] is None:
        pred = pred[0]
        repac_flag = True

    pred = torch.reshape(pred, (-1, pred.size(2)))
    trg = torch.reshape(trg, (-1, 1)).squeeze()

    if trg.size() == ():
        trg = trg.unsqueeze(0)

    if repac_flag:
        pred = [pred, None]

    try:
        loss = loss_fn(pred, trg)
    except RuntimeError as e:
        logger.info('trg.size:{}'.format(trg.size()))
        logger.info('pred.size:{}'.format(pred.size()))
        raise e

    return loss


def masked_unroll_loss(loss_fn, pred, trg, lengths, mask_neg=False):
    if len(pred.size()) < 2:
        # logger.info('pred.size:{}'.format(pred.size()))
        # logger.info('trg.size:{}'.format(trg.size()))
        # logger.info('lengths:{}'.format(lengths))

        if pred.size() == ():
            pred = pred.unsqueeze(0)
        pred = pred[:lengths[0]]
        trg = trg.squeeze()
        if trg.size() == ():
            trg = trg.unsqueeze(0)
        trg = trg[:lengths[0]]
        return loss_fn(pred, trg)

    mask = torch.zeros(pred.size()).byte()

    if pred.is_cuda:
        device = pred.get_device()
        mask = mask.to(device)

    for i, l in enumerate(lengths):
        mask[i, : l] = 1

    # mask for -1 valued trg
    if mask_neg:
        mask_n = (trg >= 0)
        if pred.is_cuda:
            mask_n = mask_n.to(device)
        mask = mask * mask_n

    denom = torch.sum(mask).float()
    if pred.is_cuda:
        denom = denom.to(device)

    if torch.sum(mask) == 0:
        z = torch.zeros(1)
        if pred.is_cuda:
            z = z.to(device)
        return z, z

    try:
        # NOTE: behavior of torch.masked_select is to extract only elements
        # with value=1 in its mask tensor. If val=0, it will be disregarded.
        pred_1d = torch.masked_select(pred, mask)
        trg_1d = torch.masked_select(trg, mask)
    except RuntimeError as e:
        logger.info('pred.size(): {}'.format(pred.size()))
        logger.info('trg.size(): {}'.format(trg.size()))
        logger.info('error: {}'.format(e))
        raise e

    return loss_fn(pred_1d, trg_1d, denom=denom)


def set_event_mask(event_size, param_size):
    mask = torch.zeros(event_size, event_size * param_size)
    for i in range(event_size):
        for p in range(param_size):
            mask[i, i + event_size * p] = 1
    return mask


def load_multitarget_data(data_name, set_type, event_size, data_filter=None,
                          base_path=None, x_hr=None, y_hr=None, test=False,
                          midnight=False, labrange=False,
                          excl_ablab=False,
                          excl_abchart=False,
                          split_id=None, icv_fold_ids=None, icv_numfolds=None,
                          remapped_data=False,
                          use_mimicid=False, option_str="", pred_labs=False,
                          pred_normal_labchart=False,
                          inv_id_mapping=None, target_size=None,
                          elapsed_time=False,
                          x_as_list=False,
                          ):
    d_path = get_data_file_path(base_path, data_name, x_hr, y_hr, data_filter,
                                set_type, midnight, labrange, excl_ablab, excl_abchart, test,
                                use_mimicid, option_str, elapsed_time, split_id)

    if remapped_data:
        remap_str = '_remapped'
    else:
        remap_str = ''

    if icv_fold_ids is not None and icv_numfolds is not None \
            and data_name in ['mimic3', 'tipas']:
        # internal cross validation
        bin_xs, bin_ys = {}, {}
        for icv_fold_id in icv_fold_ids:

            if data_name == 'mimic3':
                d_path_fold = d_path + '/cv_{}_fold_{}'.format(
                    icv_numfolds, icv_fold_id)
                logging.info("data path: {}".format(d_path_fold))
                x_path = '{}/hadm_bin_x{}.npy'.format(d_path_fold, remap_str)
                y_path = '{}/hadm_bin_y{}.npy'.format(d_path_fold, remap_str)

            elif data_name == 'tipas':
                x_path = '{}_bin_x_fold_{}.npy'.format(d_path, icv_fold_id)
                y_path = '{}_bin_y_fold_{}.npy'.format(d_path, icv_fold_id)

            with Timing('load {} ... '.format(x_path), logger=logger):
                bin_xs.update(cp_load_obj(x_path))
            with Timing('load {} ... '.format(y_path), logger=logger):
                bin_ys.update(cp_load_obj(y_path))
        bin_x, bin_y = bin_xs, bin_ys

        logger.info('elapsed_time:{}'.format(elapsed_time))
        dataset = to_multihot_sorted_vectors(bin_x, bin_y,
                                             input_size=event_size,
                                             elapsed_time=elapsed_time,
                                             pred_labs=pred_labs,
                                             pred_normal_labchart=pred_normal_labchart,
                                             inv_id_mapping=inv_id_mapping,
                                             target_size=target_size,
                                             x_as_list=x_as_list,
                                             )
    else:
        # non internal cv data
        logging.info("data path: {}".format(d_path))

        logger.info("data path: {}".format(d_path))
        if data_name == 'mimic3':
            x_path = '{}/hadm_bin_x{}.npy'.format(d_path, remap_str)
            y_path = '{}/hadm_bin_y{}.npy'.format(d_path, remap_str)

        elif data_name == 'instacart':
            x_path = '{}/user_bin_x_{}.npy'.format(d_path, set_type)
            y_path = '{}/user_bin_y_{}.npy'.format(d_path, set_type)
        elif data_name == 'tipas':
            x_path = '{}_bin_x.npy'.format(d_path)
            y_path = '{}_bin_y.npy'.format(d_path)
        else:
            raise NotImplementedError

        dataset_path = x_path.replace('.npy', '.dataset')

        # force_load_data=False

        # if os.path.exists(dataset_path) and not force_load_data:
        #     logger.info("found computed multihot vectors. let's load.")
        #     with Timing('load {} ... '.format(dataset_path)):
        #         dataset = cp_load_obj(dataset_path)
        # else:
        logger.info("not found computed multihot vectors. let's processing.")
        with Timing('load {} ... '.format(x_path), logger=logger):
            bin_x = cp_load_obj(x_path)

        with Timing('load {} ... '.format(y_path), logger=logger):
            bin_y = cp_load_obj(y_path)

        dataset = to_multihot_sorted_vectors(bin_x, bin_y, input_size=event_size,
                                             elapsed_time=elapsed_time,
                                             pred_labs=pred_labs,
                                             pred_normal_labchart=pred_normal_labchart,
                                             inv_id_mapping=inv_id_mapping,
                                             target_size=target_size,
                                             x_as_list=x_as_list,
                                             )
        # if not force_load_data:
        #     cp_save_obj(dataset, dataset_path)

    def add_hadmid_to_dataset(dataset, bin_x):
        return tuple(list(dataset) + [list(bin_x.keys())])

    return add_hadmid_to_dataset(dataset, bin_x), d_path


def pack_n_sort(instances):
    # logger.info('instances:{}'.format(instances))
    instances = sorted(instances, key=lambda x: x[0].size(0))
    x_bin_lens = [instance[0].size(0) for instance in instances]
    y_bin_lens = [instance[1].size(0) for instance in instances]
    return (instances, x_bin_lens, y_bin_lens)


def load_simulated_data(f_name_base, set_type, split_id=None,
                        icv_fold_ids=None, icv_numfolds=None,):

    base_path = "/afs/cs.pitt.edu/usr0/jel158/public/project/ts_dataset/sim_data/"

    if icv_fold_ids is not None and icv_numfolds is not None:

        # cross validation data

        instances = []
        for icv_fold_id in icv_fold_ids:

            d_path_fold = '{}/{}_split_{}_train_fold_{}.npy'.format(
                base_path, f_name_base, split_id, icv_fold_id)
            logging.info("data path: {}".format(d_path_fold))

            with Timing(f'load {d_path_fold} ... ', logger=logger):
                instances += cp_load_obj(d_path_fold)

    else:
        d_path = '{}/{}_split_{}_{}.npy'.format(
            base_path, f_name_base, split_id, set_type)
        logging.info("data path: {}".format(d_path))

        with Timing(f'load {d_path} ... ', logger=logger):
            instances = cp_load_obj(d_path)

    train_path = '{}/sim_{}_split_{}/'.format(base_path, f_name_base, split_id)
    os.system("mkdir -p {}".format(train_path))
    instances = pack_n_sort(instances)

    return instances, train_path


def load_multitarget_dic(base_path, data_name, x_hr, y_hr, data_filter=None,
                         set_type=None, midnight=False, labrange=False,
                         excl_ablab=False, excl_abchart=False,
                         test=False, split_id=None, remapped_data=False,
                         use_mimicid=False, option_str="", elapsed_time=False,
                         get_vec2mimic=False
                         ):

    if data_name == 'tipas':
        event_size = 10
        labels = {
            1: 'drink',
            2: 'sleep (wake-up)',
            3: 'heart rate',
            4: 'running',
            5: 'weight',
            6: 'food',
            7: 'walking',
            8: 'biking',
            9: 'workout',
            10: 'stretching'
        }
        vecidx2label = {i: {'category': 'cat'+str(i), 'label': labels[i]}
                        for i in range(1, event_size + 1)}
        return vecidx2label, event_size

    elif data_name == 'simdata':
        event_size = 100
        vecidx2label = {i: {'category': 'cat'+str(i), 'label': 'label'+str(i)}
                        for i in range(event_size)}
        return vecidx2label, event_size

    elif data_name == 'mimic3':
        fname = 'vec_idx_2_label_info'
        fname += '_labrange' if labrange else ''
        fname += '_exclablab' if excl_ablab else ''
        fname += '_exclabchart' if excl_abchart else ''
        if test:
            fname += '_TEST'
        if split_id is not None:
            fname += '_split_{}'.format(split_id)

        if remapped_data:
            fname += '_remapped_dic'
            set_type = 'train'
            d_path = get_data_file_path(
                base_path, data_name, x_hr, y_hr, data_filter, set_type,
                midnight, labrange, excl_ablab, excl_abchart, test, use_mimicid, option_str,
                elapsed_time
            )
            d_path += '/split_{}'.format(split_id)

        else:
            d_path = base_path

    elif data_name == 'instacart':
        d_path = get_data_file_path(
            base_path, data_name, x_hr, y_hr, data_filter, set_type, midnight,
            labrange, excl_ablab, excl_abchart, test, use_mimicid, option_str, elapsed_time
        )

        fname = 'vectorid_2_iteminfo'

    else:
        raise RuntimeError('wrong data name')

    dicfile = '{}/{}.npy'.format(d_path, fname)

    with Timing(f'read {dicfile} ... ', logger=logger):
        vecidx2label = np.load(dicfile).item()

    event_size = len(vecidx2label)

    if get_vec2mimic:
        vec2mimic_f = dicfile.replace('remapped_dic', 'vecidx2mimic')

        with Timing(f'read {vec2mimic_f} ... ', logger=logger):
            vecidx2mimic = np.load(vec2mimic_f).item()

        vecidx2label = (vecidx2label, vecidx2mimic)

    # NOTE: (Sep 11 2019) remove event_size +=1.
    # event_size += 1  # NOTE: accomodate padding index  Aug28

    return vecidx2label, event_size


def get_data_file_path(base_path, data_name, x_hr, y_hr, data_filter=None,
                       set_type=None, midnight=False, labrange=False,
                       excl_ablab=False, excl_abchart=False,
                       test=False, use_mimicid=False, option_str="",
                       elapsed_time=False, split_id=None):
    test_str = '_TEST' if test else ''
    opt_str = '_midnight' if midnight else ''

    if use_mimicid:
        opt_str += '_mimicid'

    if elapsed_time:
        opt_str += '_elapsedt'

    opt_str += option_str

    lr_str = '_labrange' if labrange else ''
    lr_str += '_exclablab' if excl_ablab else ''
    lr_str += '_exclabchart' if excl_abchart else ''
    logger.info("lr_str: {}".format(lr_str))
    if data_name == 'mimic3':
        d_path = '{}/mimic_{}_xhr_{}_yhr_{}_ytype_multi_event{}' \
                 '{}_singleseq{}'.format(base_path,
                                         set_type,
                                         x_hr, y_hr,
                                         lr_str, test_str,
                                         opt_str)
    elif data_name == 'instacart':
        d_path = '{}/{}_xhr_{}_yhr_{}_ytype_multi_event{}' \
                 '{}_singleseq{}/{}'.format(base_path, data_name, x_hr,
                                            y_hr, lr_str, test_str,
                                            opt_str, data_filter)
    elif data_name == 'tipas':
        d_path = '{}/{}_w{}_{}_split_{}_{}'.format(
            base_path, data_name, y_hr, data_filter, split_id, set_type
        )
    else:
        raise RuntimeError

    if data_name in ['mimic3', 'instacart']:
        if split_id is not None:
            d_path += '/split_{}'.format(split_id)

    return d_path


def main():
    pass


if __name__ == '__main__':
    main()

