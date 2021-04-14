# from https://github.com/overshiki/kmeans_pytorch

import torch
import numpy as np
from utils.pairwise import pairwise_distance


def forgy(X, n_clusters):
    _len = len(X)
    indices = np.random.choice(_len, n_clusters)
    initial_state = X[indices]
    return initial_state


def lloyd(X, n_clusters, device=0, tol=1e-4, verbose=False):
    # X = torch.from_numpy(X).float().cuda(device)

    initial_state = forgy(X, n_clusters)

    iter = 0
    while True:
        if verbose:
            print(('kmeans: start iter {}'.format(iter)))

        dis = pairwise_distance(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()


        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(torch.sqrt(
            torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

        if verbose:
            print(('kmeans: center_shift: {}'.format(center_shift ** 2)))

        if center_shift ** 2 < tol:
            break

        iter += 1

    return choice_cluster.cpu().numpy(), initial_state.cpu().numpy()