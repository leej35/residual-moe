# -*- coding: utf-8 -*-

import torch
import torch.utils.data
from torch.autograd import Variable
from progress.bar import IncrementalBar
from torch.nn.utils.rnn import pad_sequence
from multiprocessing.reduction import ForkingPickler
from torch.multiprocessing import reductions
from torch.utils.data import dataloader
import sys
import pickle
import os
import pandas as pd

import multiprocessing
# multiprocessing.set_start_method('spawn', True)

# from scipy.stats import rankdata


default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)


setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]






def to_numpy(inp):
    if inp.is_cuda:
        inp = inp.cpu()

    if isinstance(inp, Variable):
        inp = inp.data

    if isinstance(inp, torch.FloatTensor) or \
            isinstance(inp, torch.LongTensor):
        inp = inp.numpy()

    return inp

def sort_minibatch_single(events, times, lengths):
    """Sort minibatch length in decreasing order for original RNN
    """
    sorted_idx = sorted(enumerate(lengths), key=lambda x:x[1], reverse=True)
    sorted_idx, lengths = list(zip(*sorted_idx))
    sorted_idx = torch.LongTensor(list(sorted_idx))
    events = torch.index_select(events, 0, sorted_idx)
    times = torch.index_select(times, 0, sorted_idx)

    # if debug: print('len_inp:{}\ninp:{}'.format(len_inp, inp.size()))
    return events, times, list(lengths)


def sort_minibatch_multi(inp, trg, len_inp, len_trg, trg_time=None, inp_time=None, hadmids=None):
    """Sort minibatch length in decreasing order for original RNN"""
    sorted_idx = sorted(enumerate(len_inp), key=lambda x:x[1], reverse=True)
    sorted_idx, len_inp = list(zip(*sorted_idx))
    len_trg = [len_trg[x] for x in list(sorted_idx)]
    hadmids = [hadmids[x] for x in list(sorted_idx)]
    sorted_idx = torch.LongTensor(list(sorted_idx))
    inp = torch.index_select(inp, 0, sorted_idx)
    trg = torch.index_select(trg, 0, sorted_idx)
    if trg_time is not None:
        trg_time = torch.index_select(trg_time, 0, sorted_idx)
    if inp_time is not None:
        inp_time = torch.index_select(inp_time, 0, sorted_idx)

    # if debug: print('len_inp:{}\ninp:{}'.format(len_inp, inp.size()))
    return inp, trg, list(len_inp), list(len_trg), trg_time, inp_time, hadmids


def to_multihot_sorted_vectors_jsb(jsb_seq):
    instances = []
    for hid in range(jsb_seq.size()[0]):
        vectors_x = []
        vectors_y = []
        for t_step in range(jsb_seq.size()[1] - 1):
            vec_x = jsb_seq[hid][t_step]
            vec_y = jsb_seq[hid][t_step + 1]

            vectors_x.append(vec_x)
            vectors_y.append(vec_y)
        instances.append((vectors_x, vectors_y))

    instances = sorted(instances, key=lambda x: len(x[0]))
    x_bin_lens = [len(instance[0]) for instance in instances]
    y_bin_lens = [len(instance[1]) for instance in instances]
    return instances, x_bin_lens, y_bin_lens


def to_multihot_sorted_vectors(bin_x, bin_y, input_size,
                               elapsed_time=False,
                               pred_labs=False,
                               pred_normal_labchart=False,
                               inv_id_mapping=None,
                               target_size=None,
                               x_as_list=False):
    hadmids = list(bin_x.keys())

    if target_size is None:
        target_size = input_size

    instances = []
    for hid in hadmids:
        for bp in range(len(bin_x[hid])):
            vectors_x = []
            timevec_x = []
            vectors_y = []
            timevec_y = []
            # ordervec_y = []

            # process x
            for b_idx in range(len(bin_x[hid][bp])):
                if elapsed_time:
                    items_x = list(bin_x[hid][bp][b_idx]['events'].keys())
                else:
                    items_x = list(set(bin_x[hid][bp][b_idx]['events']))
                
                if x_as_list:
                    vec_x = tuple(items_x)
                else:
                    vec_x = torch.zeros(input_size).long()
                    for item in items_x:
                        
                        # NOTE: itemid from prep_data starts from 1.
                        # in trainer, the input multivariate vector starts 
                        # the itemid from 0 (Sep 11 2019)

                        item = item - 1
                        if item < input_size:
                            vec_x[int(item)] = 1
                        else:
                            print("!!! OOV: item:{}".format(item))

                vectors_x.append(vec_x)

                if elapsed_time:

                    if x_as_list:
                        time_vec = tuple(bin_x[hid][bp][b_idx]['events'].values())
                    else:
                        time_vec = torch.zeros(input_size)
                        time_vec -= 1  # initial value

                        for item, time in bin_x[hid][bp][b_idx]['events'].items():
                            item = item - 1
                            if item < input_size:
                                time_vec[int(item)] = time
                            else:
                                print("!!! OOV: item:{}".format(item))

                    timevec_x.append(time_vec)

            # process y
            for b_idx in range(len(bin_y[hid][bp])):
                if elapsed_time:
                    items_y = list(bin_y[hid][bp][b_idx]['events'].keys())
                else:
                    items_y = list(set(bin_y[hid][bp][b_idx]['events']))
                vec_y = torch.zeros(target_size).long()
                for item in items_y:

                    if pred_labs or pred_normal_labchart:
                        if item in list(inv_id_mapping.keys()):
                            item = inv_id_mapping[item]
                        else:
                            continue

                    item = item - 1

                    if item < input_size:
                        vec_y[int(item)] = 1
                    else:
                        print("!!! OOV: item:{}".format(item))


                vectors_y.append(vec_y)

                if elapsed_time:
                    time_vec = torch.zeros(target_size)
                    time_vec -= 1

                    for item, time in bin_y[hid][bp][b_idx]['events'].items():

                        if pred_labs or pred_normal_labchart:
                            if item in list(inv_id_mapping.keys()):
                                item = inv_id_mapping[item]
                            else:
                                continue

                        item = item - 1

                        if item < input_size:
                            time_vec[int(item)] = time
                        else:
                            print("!!! OOV: item:{}".format(item))

                    timevec_y.append(time_vec)

            if elapsed_time:
                instances.append((vectors_x, timevec_x, vectors_y, timevec_y))
            else:
                instances.append((vectors_x, vectors_y))
    #     bar.next()
    # bar.finish()
    instances = sorted(instances, key=lambda x: len(x[0]))
    x_bin_lens = [len(instance[0]) for instance in instances]
    y_bin_lens = [len(instance[1]) for instance in instances]
    return instances, x_bin_lens, y_bin_lens


def padded_collate_single(batch):
    """Collate the batch and adding padding as necessary.
    """
    instances, seq_len, n_events, is_read_once = list(zip(*batch))
    seq_events, seq_times = list(zip(*instances))
    is_read_once = is_read_once[0]

    if not is_read_once:
        seq_events = [x for seq in seq_events for x in seq]
        seq_times = [x for seq in seq_times for x in seq]
        seq_len = [len(x) for x in seq_events]
    max_len = max(seq_len)

    padded_events = []
    padded_times = []

    for events, events_len in zip(seq_events, seq_len):
        if max_len - events_len > 0:
            events = torch.cat((
                events, torch.zeros(max_len - events_len).long()))
        padded_events.append(events)

    for times, times_len in zip(seq_times, seq_len):
        if max_len - times_len > 0:
            times = torch.cat((
                times, torch.zeros(max_len - times_len).float()))
        padded_times.append(times)

    return (torch.stack(padded_events, dim=0),
            torch.stack(padded_times, dim=0),
            list(seq_len))


def fit_input_to_output(_input, _mapping, dim=2):
    # remove non-lab events for periodicity module's input
    index = torch.LongTensor(
        list(_mapping)).to(_input.device) - 1
    output = torch.index_select(_input, dim=dim, index=index).to(_input.device)
    return output


def padded_collate_multi(batch):
    """Collate the batch and adding padding as necessary.

    Assumes that batch contains a list of (doc, label, doc_len) tuple
    Assumes that doc is a list of int, label is a multi-hot FloatTensor, doc_len is int
    This is used when calling DataLoader class
    """
    instances, seq_len_x, seq_len_y, seq_hadmid = list(zip(*batch))
    
    if len(instances[0]) == 4:
        elapsed_time = True
        instances_x, timevecs_x, instances_y, timevecs_y = list(zip(*instances))
    else:
        elapsed_time = False
        instances_x, instances_y = list(zip(*instances))

    # To support x-as-list, comment next two lines:
    if type(instances_x[0]) != torch.Tensor:
        instances_x = list(map(lambda x: torch.stack(x), instances_x))
    
    if type(instances_y[0]) != torch.Tensor:
        instances_y = list(map(lambda x: torch.stack(x), instances_y))

    # print('instances_x[0]: {}'.format(instances_x[0].size()))
    # print('instances_x[1]: {}'.format(instances_x[1].size()))
    # print('instances_x[2]: {}'.format(instances_x[2].size()))
    # print('instances_x[3]: {}'.format(instances_x[3].size()))
    # print('seq_len_x[:4]: {}'.format(seq_len_x[:4]))
    # print('instances_x[0]: {}'.format(instances_x[2].size()))
    # print('instances_x[0]: {}'.format(instances_x[3].size()))

    # print('instances_x[0]: {}'.format(instances_x[0].size()))
    # print('instances_y[0]: {}'.format(instances_y[0].size()))
    # print('instances_y: {}'.format(instances_y))

    if type(instances_x[0]) == torch.Tensor:
        instances_x = pad_sequence(instances_x, batch_first=True)
    if type(instances_y[0]) == torch.Tensor:
        instances_y = pad_sequence(instances_y, batch_first=True)

    padded_out_time = []
    max_len_y = max(seq_len_y)

    if elapsed_time:

        if type(instances_x[0]) == torch.Tensor:
            timevecs_x = list(map(lambda x: torch.stack(x), timevecs_x))
            timevecs_x = pad_sequence(timevecs_x, batch_first=True)

        if type(instances_y[0]) == torch.Tensor:
            timevecs_y = list(map(lambda x: torch.stack(x), timevecs_y))
            timevecs_y = pad_sequence(timevecs_y, batch_first=True)

        result = (instances_x,
                  instances_y,
                  timevecs_x,
                  timevecs_y,
                  list(seq_len_x),
                  list(seq_len_y),
                  list(seq_hadmid),
        )

    else:
        result = (instances_x, instances_y,
                  list(seq_len_x),
                  list(seq_len_y),
                  list(seq_hadmid),
        )

    # print('padded_x: {} padded_y: {}'.format(padded_x.size(),padded_y.size()))

    return result


def get_dataset(base_path, data_name, cv_fold=None):

    if cv_fold:
        data = {'train': {}, 'test': {}}
    else:
        data = {'train': {}, 'test': {}, 'valid': {}}

    for set in list(data.keys()):

        if cv_fold:
            data_ui = '{}/event-{}-{}.txt'.format(
                base_path, cv_fold, set)
            data_uidt = '{}/time-{}-{}.txt'.format(
                base_path, cv_fold, set)
        else:
            data_ui = '{}/{}_user_item_{}.csv'.format(
                base_path, data_name, set)
            data_uidt = '{}/{}_user_item_delta_time_{}.csv'.format(
                base_path, data_name, set)

        f = open(data_ui, 'rb')
        for line in f:
            no_user = True if ',' not in line else False
            break
        f.close()

        if not no_user:
            data[set]['user_item'] = pd.read_csv(data_ui,
                sep=',', names=['user','item'], index_col=False)

            data[set]['user_item_time_delta'] = pd.read_csv(data_uidt,
                sep=',', names=['user', 'item'], index_col=False)
        else:
            data[set]['user_item'] = pd.read_csv(data_ui, names=['item'],
                index_col=False, encoding='utf-8', skipinitialspace=True)

            data[set]['user_item_time_delta'] = pd.read_csv(data_uidt,
                names=['item'], index_col=False, encoding='utf-8',
                skipinitialspace=True)

        # data_uiat = '{}/{}_user_item_accumulate_time_{}.csv'.format(
        #     base_path, data_name, set)
        # data[set]['user_item_time_accumulate'] = pd.read_csv(data_uiat,
        #                                                      sep=',',
        #                                                      names=['user',
        #                                                             'item'],
        #                                                      index_col=False)

    fname = '{}/{}_index2item.pkl'.format(base_path, data_name)
    if os.path.isfile(fname):
        index2item = pickle.load(open(fname, 'rb'))
        n_events = len(index2item)
    else:
        # compute number of events
        all_events = []
        for _data in list(data.values()):
            for x in _data['user_item']['item'].tolist():
                all_events += x.split(' ')

        all_events = [x.encode('utf-8') for x in all_events]
        all_events = [int(x) for x in all_events if x != '']

        final_list = []
        for num in all_events:
            if num not in final_list:
                final_list.append(num)
        n_events = len(final_list)+1

    data_tensors = {dname: to_instance_tensor(data, n_events, is_read_once=True)
                    for dname, data in data.items()}

    if cv_fold:
        data_tensors['valid'] = data_tensors['test']

    return data_tensors, n_events,

def to_instance_tensor(data, n_events, is_read_once=False):
    # input: {'user_item': [ ... ], 'user_item_time_delta': [ ... ], }
    try:
        ui_tensor = [torch.LongTensor([int(y) for y in x]) for x in [x.split(' ') for x in data['user_item']['item'].tolist()]]
        uit_tensor = [torch.FloatTensor([float(y) for y in x]) for x in [x.split(' ') for x in data['user_item_time_delta']['item'].tolist()]]
    except ValueError:
        ui_tensor = [torch.LongTensor([int(y) for y in x]) for x in [x.encode('utf-8').strip().split(' ') for x in data['user_item']['item'].tolist()]]
        uit_tensor = [torch.FloatTensor([float(y) for y in x]) for x in [x.encode('utf-8').strip().split(' ') for x in data['user_item_time_delta']['item'].tolist()]]

    lengths = [len(x) for x in ui_tensor]

    if is_read_once:
        instances = list(zip(ui_tensor, uit_tensor))
    else:
        instances = (ui_tensor, uit_tensor)
    return (instances, lengths, n_events)


class DatasetWithLength_multi(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.instances, self.seq_len_inp, self.seq_len_out, self.hadm_ids = dataset
        assert len(self.instances) == len(self.seq_len_inp) == len(self.seq_len_out) == len(self.hadm_ids)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return (self.instances[index], self.seq_len_inp[index],
                self.seq_len_out[index], self.hadm_ids[index])


class DatasetWithLength_single(torch.utils.data.Dataset):
    def __init__(self, dataset, is_read_once=True):
        self.instances, self.seq_len, self.n_events = dataset
        self.is_read_once = is_read_once
        assert len(self.instances) == len(self.seq_len)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return (self.instances[index], self.seq_len[index],
                self.n_events, self.is_read_once)

#
def main():
    pass


if __name__ == '__main__':
    main()

