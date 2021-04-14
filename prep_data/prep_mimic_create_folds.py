import os
import copy
import socket
import random
from datetime import datetime
from datetime import timedelta
from argparse import ArgumentParser

import sys
import os

import numpy as np

from progress.bar import IncrementalBar

from project_utils import save_obj, load_obj, cp_save_obj, cp_load_obj

login = False

SEED = 5


def process(data_path, num_folds, remapped_data=False):

    print('load admission data file')
    
    if remapped_data:
        remap_str = '_remapped'
    else:
        remap_str = ''
    

    hadm_x = cp_load_obj(data_path + '/hadm_bin_x{}.npy'.format(remap_str))


    hadm_y = cp_load_obj(data_path + '/hadm_bin_y{}.npy'.format(remap_str))

    hadm_ids = hadm_x.keys()
    random.seed(SEED)
    random.shuffle(hadm_ids)
    folded_ids = np.array_split(hadm_ids, num_folds)

    for fold, fold_ids in enumerate(folded_ids):
        print('start fold {}'.format(fold))
        hadm_x_fold = {id: hadm_x[id] for id in fold_ids}
        hadm_y_fold = {id: hadm_y[id] for id in fold_ids}

        path_x_file = '{}/cv_{}_fold_{}/hadm_bin_x{}.npy'.format(
            data_path, num_folds, fold, remap_str)
        path_y_file = '{}/cv_{}_fold_{}/hadm_bin_y{}.npy'.format(
            data_path, num_folds, fold, remap_str)

        cp_save_obj(hadm_x_fold, path_x_file)

        cp_save_obj(hadm_y_fold, path_y_file)

        print('x path: {}\ny path:{}'.format(path_x_file, path_y_file))


def main():
    parser = ArgumentParser(description='create cross validation folds')
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--num-folds', type=int)
    parser.add_argument('--remapped-data', action='store_true', default=False,
                        dest='remapped_data')
    args = parser.parse_args()

    process(args.data_path, args.num_folds, args.remapped_data)


if __name__ == '__main__':
    main()
