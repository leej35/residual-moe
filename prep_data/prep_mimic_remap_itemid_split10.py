from __future__ import division
from __future__ import print_function

from prep_mimic_gen_seq import decorate_fname
import os
import sys
from argparse import ArgumentParser

from progress.bar import IncrementalBar
import numpy as np

from project_utils import save_obj, load_obj, cp_save_obj, cp_load_obj, Timing

def main():
    
    parser = ArgumentParser()


    parser.add_argument('--single-seq', dest='single_seq', action='store_true',
                        default=False,
                        help='generate single sequence instance from an admission')
    parser.add_argument('--testmode', dest='testmode', action='store_true',
                        default=False)
    parser.add_argument('--align-y-seq', dest='align_y_seq', action='store_true',
                        default=False,
                        help='align y sequences along with x sequence (for cs3750)')
    parser.add_argument('--start-midnight', dest='start_from_midnight', action='store_true',
                        default=False,
                        help='start from midnight')

    parser.add_argument('--window-hr-x', dest='window_x_hr', type=int,
                        default=1,
                        help='window x size in hour')
    parser.add_argument('--window-hr-y', dest='window_y_hr', type=int,
                        default=1,
                        help='window y size in hour')
    parser.add_argument('--y-type', dest='y_type', type=str,
                        default='multi_step',
                        help='type of y-side to be segmented; one of '
                             '[single_event / multi_event / multi_step (Default)] \n'
                             'single_event: single closest event from the horizon \n'
                             'multi_event: multiple events as multi-hot vector \n '
                             'multi_step: sequence of multi-hot vector')
    parser.add_argument('--multistep-size', dest='y_multistep_size_hr',
                        type=int, default=48,
                        help='full width (in hour) of multi-step y. '
                             'It will be segmented by window-y size')

    parser.add_argument('--step-size', dest='step_size_hr', type=int,
                        default=12,
                        help='step size in hour. For single_seq, it is useless')

    parser.add_argument('--base-path', dest='base_path', type=str,
                        default='../../data/mimic_cs3750.sequence')

    parser.add_argument('--data-type', dest='data_type', type=str,
                        default='mimic',
                        help='name of the dataset (will be used to set folder name)')

    parser.add_argument('--lab-range', dest='lab_range', action="store_true",
                        default=False)
    parser.add_argument('--split-id', dest='split_id', type=int, default=None)
    parser.add_argument('--with-valid', dest='with_valid', action="store_true",
                        default=False)
    parser.add_argument('--use-mimicid', dest='use_mimicid', action="store_true",
                        default=False,
                        help='instead of mapped vec_index, use mimic itemid with'
                        'value string (abnormal/normal/etc.)')
    parser.add_argument('--opt-str', dest='opt_str', type=str,
                        help='optional string attached to output path name')
    parser.add_argument('--itemdic', dest='itemdic_path', type=str,
                        help='path to itemid dic (For use-mimicid)')
    parser.add_argument('--excl-lab-abnormal', dest='excl_lab_abnormal', action="store_true",
                        default=False)
    parser.add_argument('--excl-chart-abnormal', dest='excl_chart_abnormal', action="store_true",
                        default=False)
    parser.add_argument('--elapsed-time', dest='elapsed_time',
                        action="store_true", default=False)

    args = parser.parse_args()
    args.login = True

    print('-----------------------')
    for arg in sorted(vars(args)):  # print all args
        itm = str(getattr(args, arg))
        print('{0: <20}: {1}'.format(arg, itm))  #

    if not args.use_mimicid:
        dic = load_dict(args)
    else:
        dic = np.load(args.itemdic_path).item()

    count_entry = ['train_x', 'train_y', 'test_x', 'test_y']

    if args.with_valid:
        count_entry.append('valid_x')
        count_entry.append('valid_y')

    count_dic = create_count_dic(dic, count_entry)

    # load data file
    train_hadm_x, train_hadm_y = load_data(args, 'train')
    test_hadm_x, test_hadm_y = load_data(args, 'test')
    if args.with_valid:
        valid_hadm_x, valid_hadm_y = load_data(args, 'valid')

    # sweep item counts 
    count_dic = sweep_counts(
        count_dic, train_hadm_x, 'train_x', args.elapsed_time)
    count_dic = sweep_counts(
        count_dic, train_hadm_y, 'train_y', args.elapsed_time)
    count_dic = sweep_counts(
        count_dic, test_hadm_x, 'test_x', args.elapsed_time)
    count_dic = sweep_counts(
        count_dic, test_hadm_y, 'test_y', args.elapsed_time)

    if args.with_valid:
        count_dic = sweep_counts(
            count_dic, valid_hadm_y, 'valid_y', args.elapsed_time)
        count_dic = sweep_counts(
            count_dic, valid_hadm_x, 'valid_x', args.elapsed_time)

    items_in_all_bins = squeeze_dic(count_dic)
    remap_old_2_new, mapped_dic = create_remap_and_dic(items_in_all_bins, dic, args)
    
    # remap hadm_x,y for test and train
    train_hadm_x = remap_bins(
        train_hadm_x, remap_old_2_new, 'train_x', args.elapsed_time)
    train_hadm_y = remap_bins(
        train_hadm_y, remap_old_2_new, 'train_y', args.elapsed_time)
    test_hadm_x = remap_bins(
        test_hadm_x, remap_old_2_new, 'test_x', args.elapsed_time)
    test_hadm_y = remap_bins(
        test_hadm_y, remap_old_2_new, 'test_y', args.elapsed_time)

    if args.with_valid:
        valid_hadm_x = remap_bins(
            valid_hadm_x, remap_old_2_new, 'valid_x', args.elapsed_time
        )
        valid_hadm_y = remap_bins(
            valid_hadm_y, remap_old_2_new, 'valid_y', args.elapsed_time
        )

    save_data(args, 'train', 'hadm_bin_x_remapped.npy', train_hadm_x)
    save_data(args, 'train', 'hadm_bin_y_remapped.npy', train_hadm_y)
    save_data(args, 'test', 'hadm_bin_x_remapped.npy', test_hadm_x)
    save_data(args, 'test', 'hadm_bin_y_remapped.npy', test_hadm_y)
    
    if args.with_valid:
        save_data(args, 'valid', 'hadm_bin_x_remapped.npy', valid_hadm_x)
        save_data(args, 'valid', 'hadm_bin_y_remapped.npy', valid_hadm_y)

    print('done.')


def create_remap_and_dic(items_in_all_bins, dic, args):
    """
    Map original itemid in dic to ids in all bins
    """
    if args.use_mimicid:
        # create new itemid dic that maps mimic itemid to vec_idx
        vecidx_to_mimicid = {vecidx + 1: mimic_id for vecidx, mimic_id \
            in enumerate(items_in_all_bins)}
        mimicid_to_vecidx = {v:k for k,v in vecidx_to_mimicid.iteritems()}

        # vecidx to item info
        mapped_dic = {mimicid_to_vecidx[mimic_id]: item_info \
                      for mimic_id, item_info in dic.iteritems() if mimicid_to_vecidx.has_key(mimic_id)}
        remap_old_2_new = mimicid_to_vecidx
    else:
        # remap: new item id -> orig item id
        remap_new_2_old = {(idx + 1): orig_idx for idx, orig_idx in enumerate(items_in_all_bins)}
        remap_old_2_new = {v: k for k, v in remap_new_2_old.iteritems()}  # new item id -> orig item id
        mapped_dic = {remap_old_2_new[orig_itemid]: item_info
            for orig_itemid, item_info in dic.iteritems() if remap_old_2_new.has_key(orig_itemid)}


    # save dict
    
    if args.with_valid:
        set_types = ['train', 'test', 'valid']
    else:
        set_types = ['train', 'test']

    for set_type in set_types:
        direc_path = get_data_path(args, set_type)

        fname = '{}/vec_idx_2_label_info'.format(direc_path)
        fname = decorate_fname(fname, args)
        fname = fname.replace('.npy', '')
        map_dic_fname = fname +'_remapped_dic.npy'

        if args.use_mimicid:
            v2m = fname + '_vecidx2mimic.npy'
            m2v = fname + '_mimic2vecidx.npy'
            np.save(v2m, vecidx_to_mimicid)
            np.save(m2v, mimicid_to_vecidx)
        else:
            remap_n2o_fname = fname + '_remap_new_2_old.npy'
            remap_o2n_fname = fname + '_remap_old_2_new.npy'
            np.save(remap_n2o_fname, remap_new_2_old)
            np.save(remap_o2n_fname, remap_old_2_new)

        np.save(map_dic_fname, mapped_dic)
        
    return remap_old_2_new, mapped_dic


def squeeze_dic(count_dic):
    items_occur_all = []
    for itemid, counter in count_dic.iteritems():
        if np.prod(counter.values()) > 0:
            items_occur_all.append(itemid)
    
    print('all # items: {}\n after filter:{}'.format(
        len(count_dic), len(items_occur_all)
    ))
    
    return items_occur_all


def create_count_dic(item_dic, count_entry):
    dic = {itemid: {x: 0 for x in count_entry} 
        for itemid in item_dic.keys()}
        
    return dic


def sweep_counts(count_dic, ts_bin, entry_name, elapsed_time=False):
    
    hadmids = ts_bin.keys()
    bar = IncrementalBar('sweeping {} ... '.format(entry_name),
                         max=len(hadmids),
                         suffix='%(index)d/%(max)d - '
                                '%(percent).1f%% - %(eta)ds')
    for hid in hadmids:
        for bp in range(len(ts_bin[hid])):
            for b_idx in range(len(ts_bin[hid][bp])):
                if elapsed_time:
                    items_x = ts_bin[hid][bp][b_idx]['events'].keys()
                else:
                    items_x = list(set(ts_bin[hid][bp][b_idx]['events']))                
                for item in items_x:
                    count_dic[item][entry_name] += 1
        bar.next()
    bar.finish()    

    return count_dic


def remap_bins(ts_bin, remap_old_2_new, entry_name, elapsed_time=False):
    hadmids = ts_bin.keys()
    bar = IncrementalBar('sweeping {} ... '.format(entry_name),
                         max=len(hadmids),
                         suffix='%(index)d/%(max)d - '
                                '%(percent).1f%% - %(eta)ds')
    new_bin = {}
    for hid in hadmids:
        new_bin[hid] = []
        for bp in range(len(ts_bin[hid])):
            new_bp = []
            for b_idx in range(len(ts_bin[hid][bp])):

                if elapsed_time:
                    orig_items = ts_bin[hid][bp][b_idx]['events'].keys()

                    items = {remap_old_2_new[item]: \
                        ts_bin[hid][bp][b_idx]['events'][item] \
                            for item in orig_items \
                                if remap_old_2_new.has_key(item)}
                    b_idx_items = {'events': items}
                    new_bp.append(b_idx_items)

                else:
                    orig_items = list(set(ts_bin[hid][bp][b_idx]['events']))
                    items = [remap_old_2_new[item] for item in orig_items \
                        if remap_old_2_new.has_key(item)]
                    b_idx_items = {'events': items}
                    new_bp.append(b_idx_items)
            new_bin[hid].append(new_bp)
        bar.next()
    bar.finish()    

    return new_bin


# def decorate_fname(fname, args):
#     if args.lab_range:
#         fname += '_labrange'
#     if args.excl_lab_abnormal:
#         fname += ''
#     if args.testmode:
#         fname += '_TEST'
#     if args.split_id is not None:
#         fname += '_split_{}'.format(args.split_id)
#     return fname


def load_dict(args):

    # load dict
    fname = '{}/archive_vec_idx_splits/vec_idx_2_label_info'.format(args.base_path)
    fname = decorate_fname(fname, args)
    fname += '.npy'

    with Timing('load dict file: {} ... '.format(fname)):
        v2d = np.load(fname).item()
    return v2d


def get_data_path(args, set_type):
    option_str = ''

    if args.lab_range:
        option_str += '_labrange'

    if args.excl_lab_abnormal:
        option_str += '_exclablab'

    if args.excl_chart_abnormal:
        option_str += '_exclabchart'

    if args.testmode:
        option_str += '_TEST'

    if args.single_seq:
        option_str += '_singleseq'
    else:
        option_str += '_step_{}'.format(args.step_size_hr)

    if args.start_from_midnight:
        option_str += '_midnight'
        
    if args.use_mimicid:
        option_str += '_mimicid'

    if args.elapsed_time:
        option_str += '_elapsedt'

    option_str += args.opt_str

    multi_step_str = '_mlthr_{}'.format(
        args.y_multistep_size_hr) if args.y_type == 'multi_step' else ''

    direc_path = '{}/{}_{}_xhr_{}_yhr_{}_ytype_{}{}{}/'.format(
        args.base_path, args.data_type, set_type, args.window_x_hr,
        args.window_y_hr,
        args.y_type, multi_step_str, option_str)

    direc_path += 'split_{}/'.format(args.split_id)

    return direc_path


def load_data(args, set_type):
    direc_path = get_data_path(args, set_type)
    
    
    with Timing('Load hadm_x and hadm_y files: {}\n'.format(direc_path)): 
        hadm_bin_y = cp_load_obj(direc_path + 'hadm_bin_y.npy')
        hadm_bin_x = cp_load_obj(direc_path + 'hadm_bin_x.npy')

    return hadm_bin_x, hadm_bin_y


def save_data(args, set_type, file_name, data):
    direc_path = get_data_path(args, set_type)
    
    file_path = '{}/{}'.format(direc_path, file_name)

    with Timing('Save {} ... '.format(file_path)):
        cp_save_obj(data, file_path)


if __name__ == '__main__':
    main()
