


import torch
import os

import numpy as np
from tabulate import tabulate

tabulate.PRESERVE_WHITESPACE = True


class MultiClassEval(object):
    def __init__(self, event_size, use_cuda=False, device=None, ks=[1]):
        ks.append(event_size)
        self.ks = ks
        self.maxk = max(ks)
        self.use_cuda = use_cuda
        self.device = device
        self._n_retrels = {k: 0 for k in ks}
        self._n_examples = 0
        self.event_size = event_size

        self.eval = {}
        self.eval['total_gold'] = 0
        self.eval['total_correct'] = 0
        self.eval['total_pred'] = 0
        self.eval['total_hamming_loss'] = 0
        self.eval['total_cnt'] = 0
        self.eval['ranks'] = []

    def update(self, pred=None, trg=None, len_inp=None, final=None,
               base_step=None,
               prob=None):
        """

        :param pred: used for usual precision and recall (F1) measure
        :param trg:
        :param len_inp:
        :param final:
        :param base_step:
        :param prob: used for @k measurements of precision and recall
        :return:
        """
        assert pred.size() == trg.size()

        # prec-rec update (multiclass)
        # self['total_match'] += (pred == trg).float().sum().data
        n_masked_items = sum(trg == 0).item()

        # TODO: how to.. multiclass case, handle.

        self.eval['total_gold'] += trg.numel() - n_masked_items  # recall
        # self['total_num'] += reduce(lambda x, y: x * y, trg.size())
        self.eval['total_correct'] += torch.sum(pred == trg).item()
        self.eval['total_pred'] += pred.numel() - n_masked_items  # prec
        # subtract n_masked_items from total_pred
        self.eval['total_hamming_loss'] += (pred != trg).float().sum().item()
        self.eval['total_cnt'] += 1

        # At K update
        if prob.dim() > 2:
            # TODO: FIX contiguous
            # prob = prob.contiguous().view(-1, prob.size(-1))
            prob = prob.flatten(0, 1)
        if trg.dim() > 1:
            # TODO: FIX contiguous
            # trg = trg.contiguous().view(trg.numel())
            trg = trg.flatten(0, 1)

        assert len(trg.size()) == 1

        # Mean Reciprocal Rank

        rs = get_ranks(prob, trg)
        self.eval['ranks'].append(rs)

        # P@K and R@K

        aug_rel = trg.expand(self.maxk, trg.size(0)).t()
        _, idx = prob.sort(dim=1, descending=True)

        """NOTE: Inconsistency between event id <---> pred idx
        - event id is start from 1 to (event_size +1), as 0 = padding_idx 
        - class prediction (idx) from pred.sort starts 0, as it is from
          column index.
        - so there is an inconsistency between actual target class id and
          a prediction made from pred.sort.
        - to resolve the proble, we need to offset +1 on predicted class 
        """
        idx += 1
        idx = idx[:, :self.maxk]

        retrel = idx == aug_rel  # element-wise multiplication

        for k, val in self._n_retrels.items():
            if prob.size(-1) < k:
                continue
            n_retrel = retrel[:, :k].sum().item()

            if n_retrel > trg.numel():
                raise RuntimeError()

            self._n_retrels[k] += n_retrel

        self._n_examples += trg.numel()

    def compute(self, epoch, test_name='', verbose=False, logger=None,
                tstep=False, option_str='', final=False):

        # At-K
        precs = {k: 0 for k in self.ks}
        recs = {k: 0 for k in self.ks}
        f1s = {k: 0 for k in self.ks}

        for k, val in self._n_retrels.items():
            precs[k] = val / (k * self._n_examples) * 100
            recs[k] = val / self._n_examples * 100
            f1s[k] = 2 * precs[k] * recs[k] / (precs[k] + recs[k])

        # multiclass
        total_correct = self.eval['total_correct']
        total_pred = self.eval['total_pred']
        total_gold = self.eval['total_gold']
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
        if total_pred > 0:
            acc = 100.0 * total_correct / total_pred
        else:
            acc = 0

        # MRR
        ranks = torch.cat(self.eval['ranks'], -1)
        mrr_avg = mean_reciprocal_rank(ranks).item() * 100

        if verbose:
            tbl = [['Prec', prec, 'Recall', recall, 'F1', f1]]
            for k in self.ks:
                tbl.append(['P@{}'.format(k), '{:5.2f}'.format(precs[k]),
                            'R@{}'.format(k), '{:5.2f}'.format(recs[k]),
                            'F1@{}'.format(k), '{:5.2f}'.format(f1s[k])])
            print((tabulate(tbl, tablefmt='psql', )))
            print(('MRR: {:5.2f}'.format(mrr_avg)))

        if logger is not None:
            avg = 'micro'
            cv_str = ''
            tstep_str = ''

            # At-K
            for k in self.ks:
                for met_name, metr in zip(['precision', 'recall', 'f1'],
                                          [precs, recs, f1s]):
                    score = metr[k]
                    # if k != self.event_size:
                    met_name = '{}@{}'.format(met_name, k)
                    logger.log_metric("{}: {}({}) {}{}{}".format(
                        test_name, met_name, avg, cv_str, tstep_str,
                        option_str),
                        score, step=epoch)

            # Multiclass
            for met_name, score in zip(
                    ['precision', 'recall', 'f1', 'acc', 'mrr'],
                    [prec, recall, f1, acc, mrr_avg]):
                logger.log_metric("{}: {}({}) {}{}{}".format(
                    test_name, met_name, avg, cv_str, tstep_str,
                    option_str), score, step=epoch)

        f1 = acc = f1s[self.event_size]

        return f1, acc


# prob = torch.nn.functional.softmax(prob, dim=-1)

def get_ranks(prob, trg):
    """
    Provide ranking of the true label (trg) in the probability of events (prob)

    :param prob: n_entry * n_events tensor containing probability of events
    along with each entry (note: n_entry containing entries as padding)
    :param trg: n_entry tensor containing true label (index; start from 1) of
    each entry
    :return: r_pos: n_entry (without padding; so it can be smaller than input)
    that contains position of actual true label in the ranking provided from prob
    """
    _, ranks = prob.sort(dim=-1, descending=True)
    ranks += 1  # NOTE: shift one to make it start from 1

    r_pos = (ranks == trg.repeat(ranks.size(1), 1).permute(1, 0)).nonzero()
    r_pos = r_pos[:, 1]
    return r_pos


def test_get_ranks():
    prb = torch.FloatTensor([[0, 1, 2], [1, 2, 0], [2, 1, 0]])
    trg = torch.LongTensor([3, 2, 1])
    rs = get_ranks(prb, trg)

    print(('ranks:', rs))

    mrr = mean_reciprocal_rank(rs) * 100

    print(('mrr:', mrr))


def mean_reciprocal_rank_np(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    mrr = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
    return mrr


def mean_reciprocal_rank(rs):
    return (1. / (rs.float() + 1)).mean()


class IR_Prec_Rec_atK(object):
    def __init__(self, num_classes, ks):
        self.num_classes = num_classes
        self.ks = ks
        self.precs = {k: [] for k in ks}
        self.recs = {k: [] for k in ks}
        self.w_sizes = []
        self.eps = 1e-10  # very small number for numerical stability

    def update(self, ys, pred):
        if ys.use_cuda == True:
            self.update_gpu(ys, pred)
        else:
            self.update_cpu(ys, pred)

    def update_gpu(self, ys, pred):
        """
        ys: a list containing index number of the True class for an instance
        pred: a pytorch list containing probabilities for each classes
        r: a numpy list containing element such that:
            1 if the element predicted within top k
            0 otherwise
        """
        if len(pred.size()) == 2:
            pred = pred.squeeze()
        _, idx = torch.sort(pred, dim=0, descending=True)
        idx_list = idx.squeeze().data.cpu().numpy().tolist()
        top_k = idx_list[:self.ks]
        # rel&ret
        rel_ret = list(set(top_k).intersection(ys))
        recall = len(rel_ret) * 1.0 / (len(ys) + self.eps)
        precision = len(rel_ret) * 1.0 / self.ks
        self.recs.append(recall)
        self.precs.append(precision)
        self.w_sizes.append(len(ys))

    def update_cpu(self, ys, pred):
        ranking = sorted(enumerate(pred), key=lambda k: k[1], reverse=True)
        idx_list = [r[0] for r in ranking]
        top_k = idx_list[:self.ks]
        # rel&ret
        rel_ret = list(set(top_k).intersection(ys))
        recall = len(rel_ret) * 1.0 / (len(ys) + self.eps)
        precision = len(rel_ret) * 1.0 / self.ks
        self.recs.append(recall)
        self.precs.append(precision)
        self.w_sizes.append(len(ys))

    def get_scores(self):
        rec = np.mean(self.recs)
        prec = np.mean(self.precs)
        f1 = 2 * rec * prec / (rec + prec)
        avg_wsize = np.mean(self.w_sizes)
        std_wsize = np.std(self.w_sizes)
        return prec, rec, f1, avg_wsize, std_wsize


if __name__ == '__main__':
    # test()
    test_get_ranks()
