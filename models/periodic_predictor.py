

import os
import hashlib
import pickle as pickle

import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.project_utils import Timing, cp_save_obj, cp_load_obj
from utils.tensor_utils import fit_input_to_output

from utils.tensor_utils import \
    DatasetWithLength_multi, \
    DatasetWithLength_single, \
    padded_collate_multi, \
    padded_collate_single


class AdaptivePeriodicPredictor(nn.Module):
    def __init__(self, event_size, window_size_y, use_cuda=False, device=None,
                 target_type='multi', weight_scheme='default', remap=False,
                 f_exp=False, f_window=False,
                 inv_id_mapping=None, elapsed_time=False, pp_merge_signal=None,
                 hidden_dim=None, rb_concat=False, as_counts=False
                 ):
        super(AdaptivePeriodicPredictor, self).__init__()
        self.use_cuda, self.device = use_cuda, device
        self.event_size = event_size
        self.target_type = target_type
        self.prior = None
        self.pp = PeriodicPredictor(window_size_y, use_cuda, device, f_exp, f_window)
        self.weight_scheme = weight_scheme
        self.remap = remap
        self.inv_id_mapping = inv_id_mapping
        self.window_size_y = window_size_y
        self.elapsed_time = elapsed_time
        self.pp_merge_signal = pp_merge_signal
        self.as_counts = as_counts
        self.fc_out = None
        self.histogram = None

        if weight_scheme == 'feednn':
            self.alpha = nn.Linear(event_size * 3, event_size)
            self.alpha.weight.data.uniform_(-1e-08, 1e-08)
            self.alpha.bias.data.uniform_(-1e-08, 1e-08)
        
        elif weight_scheme == 'feedmlp':
            self.alpha_l1 = nn.Linear(event_size * 3, event_size)
            self.alpha_l2 = nn.Linear(event_size, event_size)
            self.alpha_l3 = nn.Linear(event_size, event_size)
            
        elif weight_scheme == 'pppe':
            self.alpha = nn.Linear(event_size * 2, event_size)
            self.alpha.weight.data.uniform_(-1e-08, 1e-08)
            self.alpha.bias.data.uniform_(-1e-08, 1e-08)

        elif weight_scheme == 'default':
            self.alpha = nn.Parameter(torch.ones(event_size) * 0.9)

        else:
            raise NotImplementedError
        
        if self.pp_merge_signal == "add-by-gate":

            #  - histogram (n_batch * event_size)
            #  - recent interval (n_batch * event_size)
            #  - elapsed counter (n_batch * event_size)
            #  - count sum over bins
            #  - mode hr over bins
            #  - mean hr over bins
            param_size = 3
            self.set_event_mask(param_size)
            # self.alpha_prior_l1 = nn.Linear(event_size * param_size, event_size)
            self.alpha_gate_l1 = nn.Linear(event_size * param_size, event_size)

            self.pp_weight = nn.Parameter(
                torch.FloatTensor(event_size).uniform_(-0.001, 0.001).to(device))

        elif self.pp_merge_signal in ['concat', 'prior-only', 'seq-only']:
            
            if self.pp_merge_signal in ['prior-only', 'seq-only']:
                mult_factor = 1
            else:
                mult_factor = 2

            self.set_event_mask(param_size=mult_factor)
            mask_input_size = hidden_dim
            if rb_concat:
                mask_input_size += event_size 
            one_matrix = torch.ones(event_size, mask_input_size).to(device)
            
            self.event_mask = torch.cat((one_matrix, self.event_mask), dim=1)


    def forward(self, input_step, step_idx, elapsed_time_step=None, debug=False):
        """
        Compute adaptive periodicity-based prediction for next event.

        One source of the prediction is coming from PeriodicPredictor
        and antoher source of prediction is coming from the prior statistics.

        Prior-based prediction is computed from querying the prior probability
        of the time gap betweeen current time step and recent event occurrence.

        The two sources are adaptive selected:
            output = alpha * prior_query + (1 - alpha) * periodic_pred

        There can be multiple strategies to learn alpha. To name a few:
        - decaying over time
        - decaying w.r.t. accumulate num of occurrences
        - learnable parameter (as a feed-forward nn)

        Prior: 
        - size: event_size, max_interval + 1

        :param input_seq_step: n_batch x 1 x n_events
        :param step_idx: index of current step (starts from 1)
        :return: pred : n_batch x (n_seqlen-1) x n_events
        """

        self.pp._to(self.device)

        if self.prior is None:
            raise RuntimeError('Call self.load_prior() function at the '
                               'beginning.')
        
        seq_period_signal = self.pp(input_step, step_idx, elapsed_time_step)
        batch_size = input_step.size(0)
        prior = self.prior.expand(batch_size, -1, -1).to(self.device)
        
        ec = torch.max(self.pp.ec, torch.zeros(1).to(self.device))

        # NOTE: assign one for those event has no prior 
        # (elasped counter is larger than prior's max interval)
        no_priors = (ec >= prior.size(2))

        # fill 0 for no_priors (these will not be used)
        ec.masked_fill_(no_priors, 0)

        prior_signal = self.query_prior(ec, prior, self.window_size_y, self.as_counts)

        # NOTE: shift -1 for prior's ec start from 0 and periodic predictor
        # starts from 1. AS we query for prior, we shift

        # in ec (elapsed counter), 0 means we observed the event in current
        # timestep. 1 means we observed in one step before.

        if self.weight_scheme == 'default':
            alpha = F.sigmoid(self.alpha.expand_as(no_priors))
        elif self.weight_scheme == 'feednn':
            w_infos = torch.cat((self.pp.ri, self.pp.ec, self.pp.ac), dim=1)
            alpha = F.sigmoid(self.alpha(w_infos))
        elif self.weight_scheme == 'feedmlp':
            w_infos = torch.cat((self.pp.ri, self.pp.ec, self.pp.ac), dim=1)
            w_infos = F.tanh(self.alpha_l1(w_infos))
            w_infos = F.tanh(self.alpha_l2(w_infos))
            alpha = F.sigmoid(self.alpha_l3(w_infos))
        elif self.weight_scheme == 'pppe':
            w_infos = torch.cat((seq_period_signal, prior_signal), dim=1)
            alpha = F.sigmoid(self.alpha(w_infos))
        else:
            raise NotImplementedError

        self.alpha_val = alpha

        self.prior_events = prior_signal
        
        if debug:
            print('')
            print('*'*32)
            print('step: {}'.format(step_idx))
            print('input_step: {}'.format(input_step))

            print('')
            print('ec: {}'.format(self.pp.ec))
            print('ac: {}'.format(self.pp.ac))
            print('ri: {}'.format(self.pp.ri))
            print('')
            print('gap: {}'.format(ec))
            print('seq_period_signal: {}'.format(seq_period_signal))
            print('prior_events: {}'.format(prior_signal))
            print('alpha: {}'.format(alpha))

        if self.pp_merge_signal == "prior-only":
            seq_period_signal.requires_grad = False
            alpha.detach()
            period_signal = prior_signal

        elif self.pp_merge_signal == "seq-only":
            prior_signal.requires_grad = False
            alpha.detach()
            period_signal = seq_period_signal

        elif self.pp_merge_signal == "add":
            period_signal = seq_period_signal + prior_signal

        elif self.pp_merge_signal == "add-by-gate":

            #  - histogram (n_batch * event_size)
            #  - recent interval (n_batch * event_size)
            #  - elapsed counter (n_batch * event_size)

            histogram = self.histogram.expand(batch_size, -1, -1).to(self.device)
            pp_weight = self.pp_weight.expand(batch_size, -1)
            # histogram_signal = self.query_prior(
            #     ec, histogram, self.window_size_y, as_counts=True)

            counts = histogram.sum(2).float()  # sum over bins => make it prior counts
            counts /= counts.max()
            _, mode_hr = torch.max(histogram, dim=2)
            mean_hr = torch.mean(histogram, dim=2)
            mode_hr = mode_hr.float()
            mode_hr /= mode_hr.max()

            # self.alpha_seq_l1.weight.data.mul_(self.event_mask)
            self.alpha_gate_l1.weight.data.mul_(self.event_mask)

            # histogram_signal, self.pp.ri, self.pp.ec, 
            w_infos = torch.cat(
                (counts, mode_hr, pp_weight), dim=1)
            # alpha_gate = self.alpha_prior_l1(w_infos)
            alpha_gate = self.alpha_gate_l1(w_infos)

            period_signal = alpha_gate * (seq_period_signal + prior_signal)

        elif self.pp_merge_signal == "concat":
            period_signal = torch.cat((seq_period_signal, prior_signal), dim=1)

        else:
            period_signal = alpha * seq_period_signal + (1 - alpha) * prior_signal

        return period_signal

    def set_event_mask(self, param_size):
        mask = torch.zeros(self.event_size, self.event_size * param_size)
        for i in range(self.event_size):
            for p in range(param_size):
                mask[i, i + self.event_size * p] = 1
        self.event_mask = mask.to(self.device)

    def query_prior(self, ec, prior, window_size_y=1, as_counts=False):
        ec = ec.unsqueeze(2).long()
        eps = 1e-18

        if window_size_y == 1:
            signal = torch.gather(input=prior, dim=2, index=ec).squeeze(2)
        else:
            # n_range: number range start from 0 to the len of prior time steps
            n_range = torch.arange(1, prior.size(2) + 1).expand(prior.size(1),-1)\
                .expand(prior.size(0),-1,-1).to(self.device) 
            
            # _ec_start and _ec_end determine boudary for window
            _ec_start = ec.expand(-1, -1, prior.size(2))
            _ec_end = _ec_start + window_size_y

            # mask val == 1 means it will be selected
            mask_pre_window = (_ec_start > n_range)
            mask_window = (_ec_start <= n_range) * (_ec_end > n_range)
            mask_post_window = (_ec_end <= n_range)

            # compute probabilities
            prob_in_window = (mask_window * prior).sum(2)
            prob_post_window = (mask_post_window * prior).sum(2)
            prob_pre_window = (mask_pre_window * prior).sum(2)

            if as_counts:
                signal = prob_in_window
            else:
                signal = (prob_in_window) / (prob_in_window + prob_post_window + eps)

        return signal

    def load_prior(self, train_data=None, d_path=None, prep_loader=False, 
                   force_reset=False, not_save=False, prior_from_mimic=False, vecidx2mimic=None):
        if d_path is None:
            d_hash = hashlib.md5(pickle.dumps(train_data)).hexdigest()
            d_path = "./tmp_prior_occurs/{}".format(d_hash)
            os.system("mkdir -p {}".format(d_path))

        # fix path on 1hr (default)
        d_path = self.fix_dpath_to_hr(d_path, target_hr_x=1, target_hr_y=1)
        
        if not not_save:
            os.system("mkdir -p {}".format(d_path))
        remap_str = '' if not self.remap else '_remap'
        remap_str += '_lab_only' if (self.inv_id_mapping is not None) else ''
        
        f_prior = '{}/prior_occur{}.pkl'.format(d_path, remap_str)
        f_prior_dic = '{}/prior_occur_dic{}.pkl'.format(d_path, remap_str)
        f_histogram = '{}/prior_occur_histogram{}.pkl'.format(d_path, remap_str)
        f_histogram_dic = '{}/prior_occur_histogram_dic{}.pkl'.format(d_path, remap_str)

        if not prior_from_mimic and not force_reset and np.prod([os.path.isfile(f) for f in \
                    [f_prior, f_prior_dic, f_histogram, f_histogram_dic]]):
            with Timing('file exists. load file: {} ... '.format(f_prior)):
                prior = cp_load_obj(f_prior)
                prior_dic = cp_load_obj(f_prior_dic)
            with Timing('file exists. load file: {} ... '.format(f_histogram)):
                histogram = cp_load_obj(f_histogram)
                histogram_dic = cp_load_obj(f_histogram_dic)
        else:
            with Timing('Not loading files. Compute new prior statistics ... '):
                (prior, prior_dic, histogram, histogram_dic) = \
                    self.compute_prior(
                        train_data, prep_loader, self.inv_id_mapping, prior_from_mimic, vecidx2mimic, d_path)
            if not not_save and not prior_from_mimic:
                with Timing('save the prior statistics to file {} ... '.format(f_prior)):
                    cp_save_obj(prior, f_prior)
                    cp_save_obj(prior_dic, f_prior_dic)
                    cp_save_obj(histogram, f_histogram)
                    cp_save_obj(histogram_dic, f_histogram_dic)

        self.prior = prior.to(self.device)
        self.histogram = histogram.to(self.device)
        self.histogram_dic = histogram_dic
        return prior, histogram, prior_dic, histogram_dic

    def fix_dpath_to_hr(self, d_path, target_hr_x=1, target_hr_y=1):
        pos_xhr_begin = d_path.find('_xhr_') + len('_xhr_')
        pos_xhr_end = d_path.find('_yhr_')
        pos_yhr_end = d_path.find('_ytype')
        prefix = d_path[:pos_xhr_begin]
        midfix = '_yhr_'
        postfix = d_path[pos_yhr_end:]
        return prefix + str(target_hr_x) + midfix + str(target_hr_y) + postfix

    def prep_data_loader(self, train_data, batch_size=512, num_workers=0):

        if isinstance(train_data, tuple):
            if self.target_type == 'multi':
                train = DatasetWithLength_multi(train_data)
            elif self.target_type == 'single':
                train = DatasetWithLength_single(train_data)
            else:
                raise NotImplementedError
        else:
            train = train_data

        dataloader = torch.utils.data.DataLoader(train,
             batch_size=batch_size,
             shuffle=False,
             num_workers=num_workers,
             drop_last=False,
             pin_memory=True,
             collate_fn=padded_collate_multi if self.target_type == 'multi' \
                 else padded_collate_single)

        return dataloader

    def compute_prior_from_traindata(self, train_data, histograms, inv_id_mapping):
        for data in train_data:
            if self.target_type == 'multi':

                if self.elapsed_time:
                    inp, trg, inp_time, trg_time, len_inp, len_trg = data
                else:
                    inp, trg, len_inp, len_trg = data
                trg_time = None
                """
                inp: batch_size x max_seq_len x n_events
                len_inp: batch_size
                """

            elif self.target_type == 'single':
                events, times, lengths = data
                inp = events[:, :-1]
                trg = events[:, 1:]
                inp_time = times[:, :-1]
                trg_time = times[:, 1:]
                len_inp = torch.LongTensor(lengths) - 1
                len_trg = torch.LongTensor(lengths) - 1
                raise NotImplementedError
            else:
                raise NotImplementedError

            if inv_id_mapping is not None:
                inp = fit_input_to_output(
                    inp, inv_id_mapping.keys())

            for patient_in_batch in range(inp.size(0)):
                intervals = {i: 0 for i in range(self.event_size)}
                count_occurs = [0] * self.event_size

                for step in range(inp.size(1)):
                    occurs = (inp[patient_in_batch][step] ==
                                1).nonzero()  # idx of occurred items
                    occurs = occurs.squeeze().tolist()
                    occurs = [occurs] if type(occurs) == int else occurs

                    non_occurs = list(
                        set(range(self.event_size)) - set(occurs))

                    for event in occurs:
                        count_occurs[event] += 1

                        if count_occurs[event] > 1:
                            event_interval = intervals[event]

                            if event_interval not in histograms[event]:
                                histograms[event][event_interval] = 0

                            histograms[event][event_interval] += 1
                            intervals[event] = 0  # reset

                    for event in non_occurs:
                        if count_occurs[event] > 0:
                            intervals[event] += 1
        
        return histograms

    def compute_prior_from_mimic(self, d_path, vecidx2mimic, histograms, max_time_cut=150):
        """
        NOTE: this setting is used for AIM-Journal and Thesis Proposal Experiment.
        The prior data is prepared  and compiled at utils/compile_interval_priors.py
        # The prior data (histogram_lab_dict.npy, etc.) can be exported to CSV through utils/prior_to_csv_mimic.py
        """
        d_lab = np.load("{d_path}/histogram_lab_dict.npy".format(**locals())).item()
        d_chart = np.load("{d_path}/histogram_chart_dict.npy".format(**locals())).item()
        d_drug = np.load(
            "{d_path}/histogram_drug_dict.npy".format(**locals())).item()
        d_proc = np.load(
            "{d_path}/histogram_procedure_dict.npy".format(**locals())).item()
        
        d_all = {**d_lab, **d_chart, **d_drug, **d_proc}

        # NOTE: vecidx2mimic starts from 1. k starts from 0.
        
        for i, k in enumerate(histograms.keys()):
            d_event = d_all[int(vecidx2mimic[k + 1])]
            histograms[i] = {k_: v_ for k_, v_ in d_event.items() \
                if type(k_) == int and int(k_) < max_time_cut}

        return histograms

    def compute_prior(self, train_data=None, prep_loader=True, 
                      inv_id_mapping=None, prior_from_mimic=False, 
                      vecidx2mimic=None, d_path=None):

        histograms = {i:{} for i in range(self.event_size)}

        if prep_loader:
            train_data = self.prep_data_loader(train_data)

        if prior_from_mimic:
            histograms = self.compute_prior_from_mimic(d_path, vecidx2mimic, histograms)
        else:
            histograms = self.compute_prior_from_traindata(train_data, histograms, inv_id_mapping)

        # normalize (histogram -> prior by nomarlizing)
        prior = {}
        for event, histogram in histograms.items():
            denom = np.sum(list(histogram.values()))
            prior[event] = {k: v / denom for k, v in histogram.items()}

        max_interval = max([k for v in list(prior.values()) for k in list(v.keys())])
        prior_tensor = torch.zeros(self.event_size, max_interval + 1).float()
        histogram_tensor = torch.zeros(self.event_size, max_interval + 1).float()
        for event, histogram in prior.items():
            for interal, prob in histogram.items():
                prior_tensor[event, interal] = prob  

        # NOTE to return a dict prior, return 'prior' instead of 'prior_tensor'
        # return_val = (prior_tensor, histograms) if get_histogram else prior_tensor

        for event, histogram in histograms.items():
            for interval, count in histogram.items():
                histogram_tensor[event, interval] = count

        """NOTE
        prior_tensor and histogram_tensor's event ids are  should be think of
        they are shifted -1 as each tensor's starting index is 0. 
        
        for dict based "prior" and "histogram", their event ids were not shifted.
        (they should be used as it is)  
        """

        return prior_tensor, prior, histogram_tensor, histograms

    def initialize_batch(self, batch_size, event_size, device=None):
        self.pp.initialize_batch(batch_size, event_size, device)


class PeriodicPredictor(nn.Module):
    def __init__(self, window_size_y, use_cuda=False, device=None, f_exp=False, f_window=False):
        super(PeriodicPredictor, self).__init__()
        # self.use_cuda, self.device = use_cuda, device
        self.ri = None
        self.f_exp = f_exp
        self.f_window = f_window
        self.window_size_y = window_size_y

    def forward(self, input_step, step_idx, elapsed_time_step=None):
        """
        Compute recent interval (ri) and elapsed counter (ec) to generate
        event prediction based on peridocity.

        For efficient computation, it supports batch computation.

        In intermediate, it has these internal variables

        tau1 : time index of recent occurrence
        tau2 : time index of second recent occurence
        (time index starts from 1 to increase towards future)

        ac : accumulate counter that adds up number of occurrence in the seq

        :param input_seq_step: n_batch x 1 x n_events
        :param step_idx: index of current step (starts from 1)
        :return: pred : n_batch x (n_seqlen-1) x n_events
        """
        eps = 0

        if elapsed_time_step is None:
            elapsed_time_step = 0
        else:
            elapsed_time_step = F.relu(elapsed_time_step)
        
        if self.window_size_y == 1:
            elapsed_time_step = 0

        # NOTE: tau2 should be computed before; otherwise use tau1_old.
        self.ac = self.ac + input_step
        self.tau2 = torch.max(self.tau2, input_step * self.tau1)
        self.tau1 = torch.max(self.tau1, input_step * (
            step_idx * self.window_size_y - elapsed_time_step))
        self.ri = (self.ac > 1).float() * \
            torch.max(self.tau1 - self.tau2, self.zten)
        self.ec = (self.ac > 0).float() * (
            (self.ec + self.window_size_y) * (input_step == 0).float() 
            + (input_step == 1).float() * elapsed_time_step)

        if self.f_exp:
            pred = (self.ri > 0).float() * \
                torch.exp(-torch.abs(self.ri - (self.ec + 1)))
        elif self.f_window:
            # NOTE: when pred is 0, instead of output 0, let's ouptut very small number (eps)
            pred = (self.ri > 0) * ((self.ri >= self.ec) * \
                                    (self.ri <= (self.ec + self.window_size_y)))
            # pred = (pred == 0) * eps + pred                                    
        else:
            pred = (self.ri > 0) * (self.ri == (self.ec + self.window_size_y))

        return pred.float()

    def _to(self, device):
        
        self.tau1 = self.tau1.to(device)
        self.tau2 = self.tau2.to(device)
        self.ri = self.ri.to(device)
        self.ec = self.ec.to(device)
        self.ac = self.ac.to(device)
        self.zten = self.zten.to(device)


    def initialize_batch(self, batch_size, event_size, device=None):
        self.register_buffer('tau1', torch.zeros(batch_size, event_size))
        self.register_buffer('tau2', torch.zeros(batch_size, event_size))
        self.register_buffer('ec', torch.zeros(batch_size, event_size))
        self.register_buffer('ac', torch.zeros(batch_size, event_size))
        self.register_buffer('zten', torch.zeros(batch_size, event_size))
        if device is not None:
            self.tau1 = self.tau1.to(device)
            self.tau2 = self.tau1.to(device)
            self.ri = self.tau1.to(device)
            self.ec = self.tau1.to(device)
            self.ac = self.tau1.to(device)
            self.zten = self.zten.to(device)
        self.tau1.detach_()
        self.tau2.detach_()
        self.ec.detach_()
        self.ac.detach_()
        self.zten.detach_()

    def deinitialize(self):
        del self.tau1
        del self.tau2
        del self.ri
        del self.ec
        del self.ac
        del self.zten



def test_adaptive_pp():

    # data = torch.randint(0, 2, (1,10,2))
    seqs = [
        # [1.,0.,1.,0.,1.,0]*3,
        # [1.,0.,0.,0.,0.,0]*3,
        # [0.,1.,1.,1.,1.,0]*3,
        [0, 0, 0, 1] * 6
    ]

    data = torch.tensor([seqs]).permute(0,2,1).float()
    
    app = AdaptivePeriodicPredictor(event_size=len(
        seqs), window_size_y=1, weight_scheme='default', f_window=True, f_exp=False,
        pp_merge_signal='seq-only', hidden_dim=0)

    train_data = [[data, 1, 1, 1]] * 1
    prior, histogram, prior_dic, histogram_dic = app.load_prior(
        train_data, d_path="./", prep_loader=False, force_reset=True)

    print('prior:')
    print(prior)

    print('histogram:')
    print(histogram)
    
    print('prior_dic:')
    print(prior_dic)
    
    print('histogram_dic:')
    print(histogram_dic)

    criterion = torch.nn.BCELoss()
    app.fc_out = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(app.parameters(), lr=0.5)
    epochs = 200
    for epoch in range(epochs):
        app.pp.initialize_batch(data.size(0), data.size(2))

        signals = []
        for step_idx in range(0, data.size(1)):
            input_step = data[:, step_idx, :]
            elapsed_time_step = input_step * 2
            period_signal = app(input_step, step_idx, elapsed_time_step)
            signals.append(period_signal)

            # print('step:{} / input_step:{} / elapsed_step:{} / ec:{} / ac:{} / ri:{} / pred:{} / prior_signal:{}'.format(
            #     step_idx, input_step.item(), elapsed_time_step.item() , app.pp.ec.item(
            #     ), app.pp.ac.item(), app.pp.ri.item(), period_signal.item(), app.prior_events.item()
            # ))
            # #app.alpha_val.item()

        period_signal = torch.stack(signals, dim=1)
        period_signal = period_signal[:, :-1, :]
        period_signal = app.fc_out(period_signal)
        prob = torch.sigmoid(period_signal)
        pred = prob > 0.5
        target = data[:, 1:, :]
        # print('pred.size():{}'.format(pred.size()))
        # print('target.size():{}'.format(target.size()))
        # # histograms
        # for e, pr in histograms.iteritems():
        #     print('{} : {}'.format(e, pr))

        # priors (NOTE: print dict)
        # for e, pr in prior.iteritems():
        #     print('{} : {}'.format(e,pr))

        # priors

        acc = torch.sum(target == pred).float() / pred.numel()
        # print('epoch: {} .format(epoch, acc))

        loss = criterion(prob, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 40 == 0:
            print('epoch: {} loss: {:.4f} acc: {:.4f}'.format(epoch, loss, acc))
        

    print('input:')
    print(data.permute(0, 2, 1))

    print('target:')
    print(target.permute(0, 2, 1))
    print('prob:')
    print(prob.permute(0, 2, 1))


# def test_seq_fixed(interval):
#     # test: use fixed data

#     data = torch.FloatTensor(interval)
#     data = data.unsqueeze(0).unsqueeze(2)

#     n_batch, n_seq, n_events = data.size()

#     pp = PeriodicPredictor(True, torch.device("cuda: 0"))
#     pp.initialize_batch(n_batch, n_events)

#     pred_steps = []
#     for step in range(n_seq):
#         pred = pp(data[:, step, :], step + 1)
#         pred_steps.append(pred)

#     pred = torch.stack(pred_steps, 1)
#     pred = pred[:, :-1, :] ## Should discard last pred.
#     target = data[:, 1:, :]

#     print('target:\n{}'.format(target.permute(0,2,1)))
#     print('pred:\n{}'.format(pred.permute(0,2,1)))

#     acc = torch.sum(pred == target).float() / pred.numel()
#     print('acc: {}'.format(acc))


def test_seq_fixed(interval):

    # test: use fixed data
    print("input: {}".format(interval))
    data = torch.FloatTensor(interval)
    data = data.unsqueeze(0).unsqueeze(2)

    n_batch, n_seq, n_events = data.size()

    pp = PeriodicPredictor(True, torch.device("cpu"), f_exp=False, f_window=True)
    pp.initialize_batch(n_batch, n_events)
    pp.window_size_y = 1

    pred_steps = []
    for step in range(n_seq):
        input_step = data[:, step, :]
        elapsed_time_step = input_step * 2
        pred = pp(input_step, step + 1, elapsed_time_step)
        pred_steps.append(pred)

        print('step:{} / input_step:{} / elapsed_time_step:{} / ec:{} / ac:{} /  tau1:{} / tau2:{} / ri:{} / pred:{} '.format(
            step, input_step.item(), elapsed_time_step.item(), pp.ec.item(), pp.ac.item(), pp.tau1.item(), pp.tau2.item() ,pp.ri.item(), pred.item(),
            )
        
        )

    pred = torch.stack(pred_steps, 1)
    pred = pred[:, :-1, :] ## Should discard last pred.
    target = data[:, 1:, :]

    print('target:\n{}'.format(target.permute(0,2,1)))
    print('pred:\n{}'.format(pred.permute(0,2,1)))

    acc = torch.sum(pred == target).float() / pred.numel()
    print('acc: {}'.format(acc))


def test_seq_random():

    # test: use random data

    n_batch, n_seq, n_events = 1, 30, 1
    data = torch.randint(0, 2, (n_batch, n_seq, n_events)).float()

    pp = PeriodicPredictor(True, torch.device("cpu"), f_window=True)
    pp.initialize_batch(n_batch, n_events)

    pred_steps = []
    for step in range(n_seq):
        pred = pp(data[:, step, :], step + 1)
        pred_steps.append(pred)

    pred = torch.stack(pred_steps, 1)
    pred = pred[:, :-1, :] ## Should discard last pred.
    target = data[:, 1:, :]

    print('target:\n{}'.format(target.permute(0,2,1)))
    print('pred:\n{}'.format(pred.permute(0,2,1)))

    acc = torch.sum(pred == target).float() / pred.numel()
    print('acc: {}'.format(acc))

def test_pp():
    interval1 = [0, 0, 0, 0, 0, 1] * 6
    interval2 = [0, 1] * 4

    # interval1[-1] = 0
    test_seq_fixed(interval1)
    # test_seq_fixed(interval2)
    # test_seq_random()

if __name__ == '__main__':
    # test_adaptive_pp()
    test_pp()
