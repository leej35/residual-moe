import os
import sys
import socket
import random
from argparse import ArgumentParser

from project_utils import cp_save_obj, cp_load_obj

login = False

SEED = 5


def process(data_path, valid_ratio):

    print('load admission data file')
    hadm_x = cp_load_obj(data_path + '/hadm_bin_x.npy')

    hadm_y = cp_load_obj(data_path + '/hadm_bin_y.npy')

    hadm_ids = hadm_x.keys()
    random.seed(SEED)
    random.shuffle(hadm_ids)

    pivot_id = int(valid_ratio * len(hadm_ids))
    valid_ids = hadm_ids[: pivot_id]
    test_ids = hadm_ids[pivot_id: ]

    print('start valid split')
    hadm_x_valid = {id: hadm_x[id] for id in valid_ids}
    hadm_y_valid = {id: hadm_y[id] for id in valid_ids}

    path_x_file = '{}/internal_valid/hadm_bin_x.npy'.format(data_path)
    path_y_file = '{}/internal_valid/hadm_bin_y.npy'.format(data_path)

    cp_save_obj(hadm_x_valid, path_x_file)
    cp_save_obj(hadm_y_valid, path_y_file)

    print('x path: {}\ny path:{}'.format(path_x_file, path_y_file))

    print('start test split')
    hadm_x_test = {id: hadm_x[id] for id in test_ids}
    hadm_y_test = {id: hadm_y[id] for id in test_ids}

    path_x_file = '{}/internal_test/hadm_bin_x.npy'.format(data_path)
    path_y_file = '{}/internal_test/hadm_bin_y.npy'.format(data_path)

    cp_save_obj(hadm_x_test, path_x_file)
    cp_save_obj(hadm_y_test, path_y_file)

    print('x path: {}\ny path:{}'.format(path_x_file, path_y_file))



def main():
    
    parser = ArgumentParser(description='split test into test and valid (BIBM19)')
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--valid-ratio', type=float)
    args = parser.parse_args()

    process(args.data_path, args.valid_ratio)


if __name__ == '__main__':
    main()
