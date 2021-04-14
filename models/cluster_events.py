

import itertools
import os
import hashlib
import pickle as pickle

import numpy as np
import torch
import torch.nn as nn

# from utils.kmeans_pytorch import lloyd

from utils.project_utils import Timing, cp_save_obj, cp_load_obj

from utils.tensor_utils import \
    DatasetWithLength_multi, \
    DatasetWithLength_single, \
    padded_collate_multi, \
    padded_collate_single, \
    sort_minibatch_multi, \
    sort_minibatch_single, \
    to_multihot_sorted_vectors

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
# from hdbscan import HDBSCAN
import numpy as np

class ClusterEvents(nn.Module):
    def __init__(self, event_size, cluster_size, train_data, batch_size,
                 target_type, num_workers=1):
        super(ClusterEvents, self).__init__()
        self.train_data = train_data
        self.batch_size = batch_size
        self.target_type = target_type
        self.num_workers = num_workers
        self.cluster_size = cluster_size
        self.event_size = event_size

    def get_co_occur_matrix(self, d_path=None):

        if d_path is None:
            d_hash = hashlib.md5(pickle.dumps(self.train_data)).hexdigest()
            d_path = "./tmp_cooccur_mat/{}/".format(d_hash)
            os.system("mkdir -p {}".format(d_path))

        f_name = '{}/cooccur_train.pkl'.format(d_path)
        if os.path.isfile(f_name):
            with Timing('file exists. load file: {} ... '.format(f_name)):
                coo_tensor = cp_load_obj(f_name)
        else:
            with Timing('file not exists. create co-occur matrix ... '):
                coo_tensor = self.create_co_occur_matrix()
            with Timing('save the matrix to file {} ... '.format(f_name)):
                cp_save_obj(coo_tensor, f_name)
        return coo_tensor

    def create_co_occur_matrix(self):

        coo = {}

        if isinstance(self.train_data, tuple):
            if self.target_type == 'multi':
                train = DatasetWithLength_multi(self.train_data)
            elif self.target_type == 'single':
                train = DatasetWithLength_single(self.train_data)
            else:
                raise NotImplementedError
        else:
            train = self.train_data

        dataloader = torch.utils.data.DataLoader(train,
             batch_size=self.batch_size,
             shuffle=True,
             num_workers=self.num_workers,
             drop_last=False,
             pin_memory=True,
             collate_fn=padded_collate_multi if self.target_type == 'multi' \
                 else padded_collate_single)


        def _update_coo(_d, _idxs):
            permuted_idxs = itertools.permutations(_idxs, 2)

            for _idx in permuted_idxs:
                _idx = sorted(_idx)

                if str(_idx) not in _d:
                    _d[str(_idx)] = 0

                _d[str(_idx)] += 1
            return _d


        for i, data in enumerate(dataloader):

            if self.target_type == 'multi':
                # inp, trg, len_inp, len_trg = data
                inp, trg, inp_time, trg_time, len_inp, len_trg = data
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

            for b in range(inp.size(0)):
                for s in range(inp.size(1)):
                    nz_idx = (inp[b][s] == 1).nonzero()
                    nz_idx = nz_idx.squeeze().tolist()
                    nz_idx = [nz_idx] if type(nz_idx) == int else nz_idx

                    if len(nz_idx) > 1:
                        coo = _update_coo(coo, nz_idx)

        coo_tensor = torch.FloatTensor(self.event_size, self.event_size)

        for coor, v in coo.items():
            co_x, co_y = [int(x) for x in coor[1:-1].split(',')]
            co_x, co_y = co_x - 1, co_y - 1  # shift to compensate padding_idx (0)
            coo_tensor[co_x, co_y] = v

        return coo_tensor


def find_elbow(arr):
    """
    Find a index that value of elements are drastically changing
    :param arr: python array, monotonically decreasing
    :return:
    """
    d = [a - b for a, b in zip(arr[:-1 ], arr[1:])]
    return d.index(max(d))


def get_clusters(event_size, cluster_size, train_data, batch_size,
                 target_type, do_svd=False, method='spectral', d_path=None,
                 event_dic=None, verbose=False, assign_labels='discretize',
                 embed_dim=None):
    init_random_state = 0
    ce = ClusterEvents(event_size, cluster_size, train_data, batch_size,
                       target_type)

    hash_str = method + assign_labels + str(event_size) + str(cluster_size) \
               + str(pickle.dumps(train_data))
    hash_key = hashlib.md5(hash_str.encode('utf-8')).hexdigest()

    with Timing('prepare co-occurrence matrix'):
        coo = ce.get_co_occur_matrix(d_path)
        coo += coo.t()

    embeding = None
    if do_svd:
        with Timing('decompose the matrix by svd'):

            if d_path is None:
                d_path = "./tmp_cluster_file/"
                os.system("mkdir -p {}".format(d_path))

            f_name_svd = '{}/svd_embed_{}.pkl'.format(d_path, hash_key)

            if os.path.isfile(f_name_svd):
                with Timing('{} file exists. lets load  ... '.format(f_name_svd)):
                    embed_svd = cp_load_obj(f_name_svd)
            else:

                coo[coo != coo] = 0

                u, s, v= coo.cpu().svd()
                if embed_dim is None:
                    embed_dim = 50
                embed_svd = u[:, :embed_dim]
                cp_save_obj(embed_svd, f_name_svd)
        embeding = embed_svd
    else:
        embed_svd = coo
    # elbow = find_elbow(s.tolist())

    with Timing('clustering by {} ... '.format(method)):

        if d_path is None:
            d_path = "./tmp_cluster_file/"
            os.system("mkdir -p {}".format(d_path))

        f_name = '{}/cluster_group_{}.pkl'.format(d_path, hash_key)
        f_name_emb = '{}/cluster_group_{}_embeds.pkl'.format(d_path, hash_key)

        if os.path.isfile(f_name):
            with Timing('file exists. load file: {} ... '.format(f_name)):
                groups = cp_load_obj(f_name)
            if method == 'KMeans':
                embeds = cp_load_obj(f_name_emb)
        else:
            with Timing('file not exists. run clustering algorithm ... '):
                groups, init_random_state, embeds = train_cluster(embed_svd,
                                                          method,
                                                          cluster_size,
                                                          init_random_state,
                                                          assign_labels,
                                                          do_svd=do_svd)
            with Timing('save the matrix to file {} ... '.format(f_name)):
                cp_save_obj(groups, f_name)
                if method == 'KMeans':
                    f_name_emb = '{}/cluster_group_{}_embeds.pkl'.format(
                        d_path, hash_key)
                    cp_save_obj(embeds, f_name_emb)

    if verbose:
        for gid, members in groups.items():
            print(('-'*80))
            print(('Group: {}  # memebers:{}'.format(gid, len(members))))
            for e in members:
                if event_dic is not None:
                    event_name = event_dic[e]['label']
                else:
                    event_name = ''
                print(('{} '.format(gid) + event_name))
        print(('=' * 80))
        print('Summary::')
        print(('k: {}'.format(cluster_size)))
        print(('method: {}'.format(method)))
        print(('assign_labels: {}'.format(assign_labels)))
        print(('num retry: {}'.format(init_random_state)))
        for gid, members in groups.items():
            print(("Group {} : n item: {}".format(gid, len(members))))

    return groups, embeding


def train_cluster(cooccur_mat, method, cluster_size, init_random_state,
                  assign_labels, do_svd=False):
    print('one of group has empty list or only one. Rerun clustering...')
    while True:
        if method == 'KMeans':
            f_method = KMeans(n_clusters=cluster_size,
                              random_state=init_random_state,
                              max_iter=50,
                              n_jobs=20,
                              verbose=0,
                              )
        elif method == 'spectral':
            f_method = SpectralClustering(n_clusters=cluster_size,
                                          n_jobs=20,
                                          random_state=init_random_state,
                                          affinity='precomputed' if do_svd is False else 'rbf',
                                          assign_labels=assign_labels)
        elif method == 'agglo':
            f_method = AgglomerativeClustering(n_clusters=cluster_size)
        elif method == 'hdbscan':
            f_method = HDBSCAN()
        else:
            raise NotImplementedError

        # just in case, process NANs to Zeros
        # cooccur_mat[torch.isnan(cooccur_mat)] = 0

        clusters_index = f_method.fit_predict(cooccur_mat.numpy())
        groups = {str(i): [] for i in range(cluster_size)}
        for i, v in enumerate(clusters_index):
            groups[str(v)].append(i)

        is_empty_or_one = [len(v) > 1 for v in list(groups.values())]

        if method in ['KMeans']:
            embeds = f_method.transform(cooccur_mat)
        else:
            embeds = None

        if np.prod(is_empty_or_one) == True:
            break
        else:
            init_random_state += 1

    return groups, train_cluster, embeds