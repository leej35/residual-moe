import os
import copy
import socket
from datetime import datetime
from datetime import timedelta
from argparse import ArgumentParser

import numpy as np
from progress.bar import IncrementalBar

from project_utils import save_obj, load_obj, cp_save_obj, cp_load_obj


SEED = 5
np.random.seed(SEED)


# ==========================
# Context generator class
# ==========================

class RNN_Instance_Generator(object):
    """
    This class generates time-segmented event sequence object
    from medTS(time-series) object.
    The medTS object is generated from med_timeseries_extractor.py
    This class produces hadm_bin_x and hadm_bin_y,


    1) visual explanation of input parameters and how these shape time-series object

        (t=1)      X       b     Y         # b (breakpoint)  :
                                                this divides X and Y window
            |------|------||-----|-----|
            ..w_x..               .w_y.    # w_x : (window_size_x),  w_y: (window_size_y)
                                               these makes width of a bin
                                               for y_type = 'multi_step',
                                               the multistep size (y_multistep_size_hr)
                                               defines full width of Y window,
                                               segmented by window_size_y
                            <----------> : (y_multistep_size_hr)

        (t=2)      X`            b`      Y`
            |------|------|------||-----|-----|


        (delta)           b      b`
            |------|------||-----||-----|-----|   # step_size: time gap between
                                                    current and next breakpoint
                           .step.                   step_size = b` - b


    2) To access an actual list of event-ids:

            hadm_bin_x.values()[0][0][0]['events'].keys()
                                |  |  |      |      |
                                |  |  |      |      it returns list of event_ids.
                                |  |  |      |       .values() returns list of dicts
                                |  |  |      |         which each dict of
                                |  |  |      |          'value' and 'timedelta'
                                |  |  |      |
                                |  |  |     also, ['bin_start_t'] and ['bin_end_t']
                                |  |  |                (end and start time) is available
                                |  |  |
                                |  |  a window's index
                                |  |
                                |  breakpoint's index
                                |
                                hadm's index


    """

    def __init__(self,
                 window_size_x_hr,
                 window_size_y_hr,
                 step_size_hr,
                 medTSfile,
                 path='',
                 data_type='mimic',
                 set_type=None,
                 y_type=None,
                 y_multistep_size_hr=None,
                 testmode=False,
                 small=None,
                 elapsed_time=False,
                 single_seq=False,
                 random_breakpoint=False,
                 y_start_from_end_of_x=True,
                 re_map_path=None,
                 align_y_seq=False,
                 start_from_midnight=False,
                 is_lab_range=False,
                 split_id=None,
                 use_mimicid=False,
                 opt_str="",
                 excl_lab_abnormal=False,
                 excl_chart_abnormal=False,
                 force_create=False,
                 ):

        self.data_type = data_type  # mimic
        self.set_type = set_type  # train/test/valid

        self.window_size_x_hr = window_size_x_hr
        self.window_size_y_hr = window_size_y_hr
        self.step_size_hr = step_size_hr

        self.window_size_x = timedelta(0, window_size_x_hr * 3600)
        self.window_size_y = timedelta(0, window_size_y_hr * 3600)
        self.step_size = timedelta(0, step_size_hr * 3600)
        self.elapsed_time = elapsed_time
        self.y_multistep_size_hr = y_multistep_size_hr
        self.y_type = y_type
        self.random_breakpoint = random_breakpoint
        # start y bin from the end of x bin
        self.y_start_from_end_of_x = y_start_from_end_of_x  # (only for single_seq)

        if self.y_type == 'multi_step':
            assert y_multistep_size_hr != None
            self.y_multistep_size = timedelta(0, self.y_multistep_size_hr * 3600)

        self.single_seq = single_seq
        self.medTS = {}
        self.path = path
        self.medTSfile = medTSfile
        self.testmode = testmode
        self.small = small
        self.align_y_seq = align_y_seq # align y-sequence with x-sequence (cs3750)
        self.re_map_path = re_map_path
        self.start_from_midnight = start_from_midnight
        self.is_lab_range = is_lab_range
        self.excl_lab_abnormal = excl_lab_abnormal
        self.excl_chart_abnormal = excl_chart_abnormal
        self.split_id = split_id
        self.use_mimicid = use_mimicid
        self.force_create = force_create

        if self.re_map_path:
            self.re_map = np.load(re_map_path).item()
        else:
            self.re_map = None

        option_str = ''

        if self.is_lab_range:
            option_str += '_labrange'

        if self.excl_lab_abnormal:
            option_str += '_exclablab'

        if self.excl_chart_abnormal:
            option_str += '_exclabchart'

        if self.testmode:
            option_str += '_TEST'

        if self.small:
            option_str += '_small{}'.format(self.small)

        if self.single_seq:
            option_str += '_singleseq'
        else:
            option_str += '_step_{}'.format(self.step_size_hr)

        if self.random_breakpoint:
            option_str += '_randombp'

        if start_from_midnight:
            option_str += '_midnight'

        if self.use_mimicid:
            option_str += '_mimicid'

        if self.elapsed_time:
            option_str += '_elapsedt'

        option_str += opt_str

        multi_step_str = '_mlthr_{}'.format(
            self.y_multistep_size_hr) if self.y_type == 'multi_step' else ''

        self.direc_path = '{}/{}_{}_xhr_{}_yhr_{}_ytype_{}{}{}/'.format(
            self.path, self.data_type, self.set_type, self.window_size_x_hr,
            self.window_size_y_hr,
            self.y_type, multi_step_str, option_str)

        if split_id is not None:
            self.direc_path += '/split_{}/'.format(split_id)


        assert y_type != None \
            and (y_type in ['single_event', 'multi_event', 'multi_step']) \
            and (set_type in ['train', 'test', 'valid'])

        self.hadm_bin_x, self.hadm_bin_y = self.generate_data_instances()

    def get_data(self):
        return self.hadm_bin_x, self.hadm_bin_y

    def generate_data_instances(self):

        direc_path = self.direc_path

        print('generate_data_instances')
        if os.path.exists(direc_path) and not self.force_create:
            print('=====================')
            print(
                'Load RNN_Instance_Generator Objects : {}'.format(direc_path))
            print('=====================')
            print('start to loading objects.')
            hadm_bin_x = cp_load_obj(direc_path + 'hadm_bin_x.npy')
            hadm_bin_y = cp_load_obj(direc_path + 'hadm_bin_y.npy')
            print('done loading objects.')
        else:
            print('=====================')
            print(
                'WARNING: RNN_Instance_Generator Objects were not '
                'found.\n new path:{}'.format(
                    direc_path))
            print('=====================')
            os.system("mkdir -p {}".format(direc_path))
            print("create new directory done.")
            # load medTS object
            print("start to load {}".format(self.medTSfile))
            self.medTS = np.load(self.medTSfile).item()
            print("load {} done.".format(self.medTSfile))

            hadm_list = self.medTS.keys()

            if self.small:
                hadm_list = hadm_list[:int(len(hadm_list) * self.small)]

            if self.testmode:
                hadm_list = hadm_list[:10]

            _hadm_list = copy.deepcopy(hadm_list)

            hadm_bin_x = {}
            hadm_bin_y = {}
            span_lengths = []

            # 1. create bins

            # hadm_bin_x[hadm_id][i][0] : list of bin start and end time
            # hadm_bin_x[hadm_id][i][1] : list of event indices
            # hadm_bin_x[hadm_id][i][2] : dictionary of elapsed time of each events
            #                                 (key: event-id, value: elasped time).
            #                             Normalized with window_x or window_y size.
            bar = IncrementalBar('1. Create bins', max=len(hadm_list),
                 suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

            for hadm_id in hadm_list:
                bar.next()
                start_time = self.medTS[hadm_id][0]['start_time']

                if self.start_from_midnight:
                    start_time = datetime(year=start_time.year,
                                          month=start_time.month,
                                          day=start_time.day,
                                          hour=0, minute=0) + timedelta(days=1)

                end_time = self.medTS[hadm_id][-1]['end_time']
                span = (end_time - start_time).total_seconds()

                span_lengths.append(span)

                hadm_bin_x[hadm_id] = []
                hadm_bin_y[hadm_id] = []

                x_shift = self.window_size_x

                if self.single_seq:

                    if self.y_type == 'single_event':
                        raise NotImplementedError('for single_seq option,\
                            y_type=single_event is not implemented ')

                    elif self.y_type == 'multi_event'and not self.align_y_seq:
                        raise NotImplementedError('for single_seq option, \
                            y_type=multi_event is not implemented ')

                    elif self.y_type == 'multi_event' and self.align_y_seq:
                        breakpoint = end_time - self.window_size_y

                    elif self.y_type == 'multi_step':

                        if self.random_breakpoint:
                            # choose breakpoint in the second half of the admission
                            bp_span_from_end = max(span / 2,
                                                   self.y_multistep_size\
                                                        .total_seconds())
                            random_bp_span_from_end = np.random.uniform(
                                low=self.y_multistep_size.total_seconds(),
                                high=bp_span_from_end)
                            breakpoint = end_time - timedelta(
                                seconds=random_bp_span_from_end)
                        else:
                            breakpoint = end_time - self.y_multistep_size

                        breakpoint_limit = end_time - self.y_multistep_size


                else:  # non-single-sequence
                       # (generate multiple sequences from an admission)
                    breakpoint = start_time + x_shift

                    if self.y_type == 'single_event':
                        one_prev_end_time = self.medTS[hadm_id][-2]['end_time']
                        breakpoint_limit = one_prev_end_time
                    elif self.y_type == 'multi_event':
                        breakpoint_limit = end_time - self.window_size_y
                    elif self.y_type == 'multi_step':
                        breakpoint_limit = end_time - self.y_multistep_size

                while self.single_seq or (not self.single_seq and (
                        breakpoint < breakpoint_limit)):

                    bin_x_start_t = bin_x_end_t = start_time
                    x_bins = []

                    # create x bins
                    while (bin_x_start_t + self.window_size_x) < breakpoint:
                        bin_x_start_t = bin_x_end_t
                        bin_x_end_t += self.window_size_x

                        x_bins.append(
                            dict(bin_start_t=bin_x_start_t,
                                 bin_end_t=bin_x_end_t,
                                 events={} if self.elapsed_time else []))

                    # create y bins
                    y_bins = []

                    if self.y_type in ['multi_event', 'single_event'] \
                            and not self.align_y_seq:
                        bin_y_start_t = breakpoint
                        bin_y_end_t = breakpoint + self.window_size_y

                        y_bins.append(
                            dict(bin_start_t=bin_y_start_t,
                                 bin_end_t=bin_y_end_t,
                                 events={} if self.elapsed_time else []))

                    if self.y_type == 'multi_event' and self.align_y_seq \
                            and self.single_seq:
                        for x_bin in x_bins:
                            y_bins.append(
                                dict(bin_start_t=x_bin['bin_end_t'],
                                     bin_end_t=x_bin['bin_end_t'] + self.window_size_y,
                                     events={} if self.elapsed_time else []
                                )
                            )

                    elif self.y_type == 'multi_step':

                        if self.y_start_from_end_of_x:
                            bin_y_start_t = bin_y_end_t = bin_x_end_t
                        else:
                            bin_y_start_t = bin_y_end_t = breakpoint

                        while bin_y_start_t < (
                                breakpoint + self.y_multistep_size):
                            bin_y_start_t = bin_y_end_t
                            bin_y_end_t += self.window_size_y

                            y_bins.append(
                                dict(bin_start_t=bin_y_start_t,
                                     bin_end_t=bin_y_end_t,
                                     events={} if self.elapsed_time else []))

                    if len(x_bins) and len(y_bins):
                        hadm_bin_x[hadm_id].append(x_bins)
                        hadm_bin_y[hadm_id].append(y_bins)

                    breakpoint += self.step_size

                    if self.single_seq:
                        break

            bar.finish()

            print(
                '[INFO] number of admissions remain: {}, before filter out:{}'.format(
                    len(_hadm_list), len(hadm_list)))
            hadm_list = _hadm_list

            assert len(hadm_bin_x[hadm_id]) == len(hadm_bin_y[hadm_id])

            # bar = IncrementalBar('2. Remove empty instances (find remove indices)',
            #     max=len(hadm_list))

            # # remove empty instances
            # remove_idxs = []
            # for i in range(len(hadm_bin_x.keys())):
            #     bar.next()
            #     for j in range(len(hadm_bin_x.values()[i])):
            #         if len(hadm_bin_x.values()[i][j]) == 0 \
            #               or len(hadm_bin_y.values()[i][j]) == 0:
            #             remove_idxs.append((i,j))
            # bar.finish()

            # bar = IncrementalBar('3. Remove empty instances (remove events)',
            #     max=len(remove_idxs))
            # for i,j in remove_idxs[::-1]:
            #     bar.next()
            #     del hadm_bin_x.values()[i][j]
            #     del hadm_bin_y.values()[i][j]
            # bar.finish()

            def is_overlap(start_a, end_a, start_b, end_b):
                return not(end_a < start_b or end_b < start_a)
                # latest_start = max(start_a, start_b)
                # earliest_end = min(end_a, end_b)
                # overlap = (earliest_end - latest_start).days + 1
                # return overlap > 0

            # 2. put events into bins

            n_items = reduce(lambda x, y: x + y,
                             [len(self.medTS[hadm_id]) for hadm_id in
                              hadm_list])

            bar = IncrementalBar('2. Put events into bins', max=n_items,
                                 suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

            for hadm_id in hadm_list:
                for item in self.medTS[hadm_id]:

                    # get item info
                    if self.use_mimicid:
                        item_idx = item['mimic_itemid']
                    else:
                        item_idx = item['vec_idx']
                        if item_idx in self.re_map.keys():
                            item_idx = self.re_map[item_idx]

                    category = item['category']
                    item_start_t = item['start_time']
                    item_end_t = item['end_time']

                    for bpoint in range(len(hadm_bin_x[hadm_id])):
                        # =====================
                        # generate X
                        # ---------------------

                        for i in range(len(hadm_bin_x[hadm_id][bpoint])):
                            bin_x = hadm_bin_x[hadm_id][bpoint][i]

                            bin_start_t, bin_end_t = bin_x['bin_start_t'], \
                                                     bin_x['bin_end_t']

                            if is_overlap(bin_start_t, bin_end_t, item_start_t,
                                          item_end_t):

                                if self.elapsed_time:
                                    t_delta = max((bin_end_t - item_end_t).total_seconds(), 0)
                                    if not bin_x['events'].has_key(item_idx):
                                        bin_x['events'][item_idx] = t_delta
                                    else:
                                        bin_x['events'][item_idx] = min(
                                            t_delta, bin_x['events'][item_idx])
                                else:
                                    bin_x['events'].append(item_idx)

                        # =====================
                        # generate Y
                        # ---------------------

                        for i in range(len(hadm_bin_y[hadm_id][bpoint])):
                            bin_y = hadm_bin_y[hadm_id][bpoint][i]
                            bin_start_t, bin_end_t = bin_y['bin_start_t'], \
                                                     bin_y['bin_end_t']

                            if is_overlap(bin_start_t, bin_end_t, item_start_t,
                                          item_end_t):

                                if (self.y_type in ['multi_step', 'multi_event']) \
                                    or (self.y_type == 'single_event' \
                                        and len(bin_y['events']) == 0
                                    ):


                                    if self.elapsed_time:
                                        t_delta = max((item_start_t - bin_start_t)\
                                                    .total_seconds(), 0)

                                        if not bin_y['events'].has_key(item_idx):
                                            bin_y['events'][item_idx] = t_delta
                                        else:
                                            bin_y['events'][item_idx] = min(
                                                t_delta, bin_y['events'][item_idx])
                                    else:
                                        bin_x['events'].append(item_idx)
                                else:
                                    raise NotImplementedError
                    bar.next()
            bar.finish()

            print('start to save hadm_bin_x and hadm_bin_y files')
            cp_save_obj(hadm_bin_x, direc_path + 'hadm_bin_x.npy')
            cp_save_obj(hadm_bin_y, direc_path + 'hadm_bin_y.npy')
            np.save('{}admission_lengths.npy'.format(self.direc_path),
                    span_lengths)
            print('done.')
        return hadm_bin_x, hadm_bin_y


def test_seq_instance_gen(args):
    batch_size = 8
    v2d = np.load('{}/data/vec_idx_2_label_info.npy'.format(args.base_path)).item()
    event_size = len(v2d)

    print('multi-event')
    va_data_loader = RNN_Instance_Generator(
        window_size_x_hr=12,
        window_size_y_hr=12,
        step_size_hr=6,
        medTSfile='data/medTS_MV_valid_instances.npy',
        path='data_seq',
        data_type='mimic',
        set_type='valid',
        y_type='multi_step',
        y_multistep_size_hr=48,
        testmode=True,
        elapsed_time=False
    )

    va_hadm_bin_x, va_hadm_bin_y = va_data_loader.get_data()

    valid_data = to_multihot_sorted_vectors(va_hadm_bin_x, va_hadm_bin_y,
                                            event_size)

    max_data_len = max(valid_data[2])
    valid = DatasetWithLength(valid_data)



def decorate_fname(fname, args):
    if args.lab_range:
        fname += '_labrange'
    if args.excl_lab_abnormal:
        fname += '_exclablab'
    if args.excl_chart_abnormal:
        fname += '_exclabchart'
    if args.testmode:
        fname += '_TEST'
    if args.split_id is not None:
        fname += '_split_{}'.format(args.split_id)
    fname += '.npy'
    return fname

    
def main():


    parser = ArgumentParser(description='To run Sequence Instance Generator')

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

    parser.add_argument('--window-x', dest='window_size_x_hr', type=int,
                        default=1,
                        help='window x size in hour')
    parser.add_argument('--window-y', dest='window_size_y_hr', type=int,
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
                        default='../../data/mimic_cs3750.events')
    parser.add_argument('--medts-file', dest='medTSfile', type=str,
                        default='medTS_MV_valid_instances_TEST.npy',
                        help='location of medTS file ')
    parser.add_argument('--data-type', dest='data_type', type=str,
                        default='mimic',
                        help='name of the dataset (will be used to set folder name)')
    parser.add_argument('--data-save-path', dest='data_save_path', type=str,
                        default='../../data/mimic_cs3750.sequence/',
                        help='name of the data path to files be saved ')
    parser.add_argument('--set-type', dest='set_type', type=str,
                        help='one of [train / test / valid]')
    parser.add_argument('--random-breakpoint', dest='random_breakpoint',
                        action='store_true', default=False,
                        help='randomly select breakpoint within '
                             'second half of the admission')
    parser.add_argument('--lab-range', dest='lab_range', action="store_true",
                        default=False)
    parser.add_argument('--split-id', dest='split_id', type=int, default=None)
    parser.add_argument('--use-mimicid', dest='use_mimicid', action="store_true",
                        default=False,
                        help='instead of mapped vec_index, use mimic itemid with'
                              'value string (abnormal/normal/etc.)')
    parser.add_argument('--opt-str', dest='opt_str', type=str,
                        help='optional string attached to output path name')
    parser.add_argument('--excl-lab-abnormal', dest='excl_lab_abnormal', action="store_true",
                        default=False)
    parser.add_argument('--excl-chart-abnormal', dest='excl_chart_abnormal', action="store_true",
                        default=False)

    parser.add_argument('--force-create', dest='force_create',
                        action="store_true", default=False)
    parser.add_argument('--elapsed-time', dest='elapsed_time',
                        action="store_true", default=False)


    args = parser.parse_args()

    
    if not args.use_mimicid:
        fname = '{}/data/vec_idx_2_label_info'.format(args.base_path)
        fname = decorate_fname(fname, args)

        v2d = np.load(fname).item()

        trg_fname = '{}/vec_idx_2_label_info'.format(args.data_save_path)
        trg_fname = decorate_fname(trg_fname, args)
        np.save(trg_fname, v2d)

        event_size = len(v2d)
        print('event size: {}'.format(event_size))

    print('-----------------------')
    for arg in sorted(vars(args)):  # print all args
        itm = str(getattr(args, arg))
        print('{0: <20}: {1}'.format(arg, itm))  #

    assert args.medTSfile is not None
    assert args.set_type is not None

    os.system('mkdir -p {}'.format(args.data_save_path))

    dataloader = RNN_Instance_Generator(
        window_size_x_hr=args.window_size_x_hr,
        window_size_y_hr=args.window_size_y_hr,
        step_size_hr=args.step_size_hr,
        medTSfile='{}/data/{}'.format(args.base_path, args.medTSfile),
        path=args.data_save_path,
        data_type=args.data_type,
        set_type=args.set_type,
        y_type=args.y_type,
        y_multistep_size_hr=args.y_multistep_size_hr,
        testmode=args.testmode,
        single_seq=args.single_seq,
        random_breakpoint=args.random_breakpoint,
        align_y_seq=args.align_y_seq,
        re_map_path='{}/dic/re_map.npy'.format(args.base_path),
        start_from_midnight=args.start_from_midnight,
        is_lab_range=args.lab_range,
        split_id=args.split_id,
        use_mimicid=args.use_mimicid,
        opt_str=args.opt_str,
        excl_lab_abnormal=args.excl_lab_abnormal,
        excl_chart_abnormal=args.excl_chart_abnormal,
        force_create=args.force_create,
        elapsed_time=args.elapsed_time
    )

    va_hadm_bin_x, va_hadm_bin_y = dataloader.get_data()
    print('done.')

if __name__ == "__main__":
    main()

