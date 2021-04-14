"""
This one generates split-based event_ts based on existing one.
Reason is to remedy event_dic name inconsistency problem observed on Aug 2019.
"""
import os
import sys
import csv
from time import strftime
import random
from copy import deepcopy
from collections import Counter

from progress.bar import IncrementalBar
import MySQLdb # conda install -c anaconda mysql-python

import numpy as np
from argparse import ArgumentParser

from DBConnectInfo import DBHOST, DBUSER, DBPASS, DBNAME

# # ---------------
# DBHOST = DBHOST
# DBUSER = DBUSER
# DBPASS = DBPASS
# DBNAME = DBNAME
# # ---------------


# Path setup
base_path = '/basepath/'
data_path = base_path + 'mimic.events/data/'
dic_path = base_path + './dic/'
log_path = base_path + './log/'

for path in [data_path, dic_path, log_path]:
    os.system('mkdir -p {}'.format(path))

# if you want 12 hour window overlap, then set 12.
# if you want no overlap, then set 24

# agefilter file
# - second column is admissions ids
age_filter_file = dic_path + 'dic_ADMISSION_age_filtered_between_18_90.csv'

# metavision admission list ; use first column
hadm_mv_whitelist_file = dic_path + 'dic_INPUTEVENTS_MV_hadm_list.csv'


#################
#
# Read item labels
#
#################

def apply_age_filter(medTS=None, metavision_hadm_ids=[]):
    # returning list is whitelist (hadm_ids that will be used)

    # add age filter

    db = MySQLdb.connect(host=DBHOST,
                         user=DBUSER,
                         passwd=DBPASS,
                         db=DBNAME)
    cur = db.cursor()
    query = """
    SELECT ICUSTAY_ID, TIMESTAMPDIFF(YEAR, P.DOB, I.INTIME) AS AGE
    FROM ICUSTAYS I
    LEFT OUTER JOIN PATIENTS P
    ON I.SUBJECT_ID = P.SUBJECT_ID
    WHERE I.dbsource='metavision' AND TIMESTAMPDIFF(YEAR, P.DOB, I.INTIME) >= 18
    """

    # fetch
    age_dic_hadm = {}
    cur.execute(query)
    for row in cur.fetchall():
        icustay_id = row[0]
        age = row[1]
        age_dic_hadm[icustay_id] = age
    
    db.close()

    age_list = age_dic_hadm.keys()

    # add metavision filter
    if metavision_hadm_ids == []:
        with open(hadm_mv_whitelist_file) as mvcsv:
            rd = csv.reader(mvcsv)
            for row in rd:
                if row[0] != 'HADM_ID':
                    metavision_hadm_ids.append(int(row[0]))

    # get overlap between three
    whitelist = list(set(age_list).intersection(metavision_hadm_ids))
    if medTS:
        whitelist = list(set(whitelist).intersection(medTS.keys()))
    print('admission count; age list:{}, mv list:{}, overlap:{}'.format(
        len(age_list), len(metavision_hadm_ids), len(whitelist)))
    return [x for x in whitelist]


def create_id_remap(remap_csv_file, SRC_COL_NAME, TRG_COl_NAME):
    re_map = {}
    with open(remap_csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            src_id = row[SRC_COL_NAME]
            trg_id = row[TRG_COl_NAME]
            if src_id != trg_id:
                if not re_map.has_key(src_id):
                    re_map[src_id] = []

                re_map[src_id].append(trg_id)
    return re_map


# dictionary generator
def create_dic(dic_csv_file, dic, ID_COL_NAME, LABEL_COL_NAME, re_map=None):

    _dic = {}

    with open(dic_csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mimic_itemid = str(row[ID_COL_NAME])
            label = row[LABEL_COL_NAME]

            if (re_map and mimic_itemid not in re_map.keys()) or not re_map:
                _dic[mimic_itemid] = {}
                _dic[mimic_itemid]['label'] = label

            elif re_map and mimic_itemid in re_map.keys():
                print('create_dic: {} found '
                      'in re_map.keys()'.format(mimic_itemid))

    return _dic


class LabEventDic(object):
    def __init__(self, read_from='csv', excl_lab_abnormal=False):
        self.range_dic = {}  # key: itemid, value: {'normal_h', 'normal_l'}
        self.rawid_dic = {}  # key:itemid, value: type_of_discrete_or_real
        self.read_from = read_from
        self.excl_lab_abnormal = excl_lab_abnormal
        self._create_dic_lab_events()

    def _create_dic_lab_events(self):
        """
        Logic:
            IF itemid is in normal_range_items:
                range: normal / abnormal_high / abnormal_low
            ELSE:
                range: normal / abnormal
        """
        # prev: dic_LAB_filter_100occur.csv
        lab_csv_file = '{}LABEVENT_MV_PER_ICUSTAY_COUNTS_larger_than_500_icustays.csv'.format(dic_path)
        normal_range_value_file = '{}dic_LABEVENT_NORMAL_AVG.csv'.format(dic_path)

        # Normal range file
        id_col_name = 'ITEMID'
        normal_avg_col_name= 'NORMAL_AVG'

        if self.read_from == 'db':
            raise NotImplementedError

        if self.read_from == 'csv' and not self.excl_lab_abnormal:
            with open(normal_range_value_file) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    mimic_itemid = str(row[id_col_name])
                    normal_avg = row[normal_avg_col_name]
                    self.range_dic[mimic_itemid] = {'normal_avg': normal_avg}
        else:
            self.range_dic = {}

        # Lab CSV file
        id_col_name = 'ITEMID'
        label_col_name = 'ITEM_LABEL'

        with open(lab_csv_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                mimic_itemid = str(row[id_col_name])
                label = row[label_col_name]

                self.rawid_dic[mimic_itemid] = {}
                self.rawid_dic[mimic_itemid]['label'] = label

        return None

    def generate_itemid_dic(self, dic):
        # executed only with lab_range option
        assert len(dic) > 0

        _dic = {}

        # process REAL types
        for itemid in self.rawid_dic.keys():

            if itemid in self.range_dic.keys():
                _values = ['NORMAL', 'ABNORMAL_LOW', 'ABNORMAL_HIGH']
            else:
                _values = ['NORMAL', 'ABNORMAL']
            
            for value in _values:
                _itemid = str(itemid) + '-' + value
                label = self.rawid_dic[itemid]['label']
                _dic[_itemid] = {}
                _dic[_itemid]['label'] = label + '-' + value

        return _dic

    def retrieve_itemid(self, mimic_itemidx, value, is_abnormal=False):

        if mimic_itemidx in self.rawid_dic.keys():

            if mimic_itemidx in self.range_dic.keys():
                if not is_abnormal:
                    _value = 'NORMAL'

                elif self.range_dic[mimic_itemidx]['normal_avg'] > value:
                    _value = 'ABNORMAL_LOW'

                elif self.range_dic[mimic_itemidx]['normal_avg'] < value:
                    _value = 'ABNORMAL_HIGH'
            else:
                if is_abnormal:
                    _value = 'ABNORMAL'
                else:
                    _value = 'NORMAL'
            
            if self.excl_lab_abnormal:
                # override other conditions
                _value = 'ORDER'

            itemid = mimic_itemidx + '-' + _value
        else:
            raise RuntimeError('Lab mimic_itemid not found. '
                               'mimic_itemid:{} value:{} isabnormal:{}'
                               ''.format(mimic_itemidx, value, is_abnormal))

        return itemid


class ChartEventDic(object):
    def __init__(self, excl_chart_abnormal=False):
        self.range_dic = {}  # key: itemid, value: {'normal_h', 'normal_l'}
        self.discrete_dic = {}  # key: itemid, value: list of discrete values
        self.chartevent_dic = {}  # key:itemid, value: type_of_discrete_or_real
        self.chart_raw_mimic_ids = []  # key:itemid
        self.excl_chart_abnormal = excl_chart_abnormal

        self._create_dic_chart_events()

    def _create_dic_chart_events(self):
        physio_csv_file \
            = '{}dic_PHYSIO_filter_manual_selection.csv'.format(
            dic_path)
        discrete_value_file \
            = '{}dic_PHYSIO_filter_manual_selection_9ITEM_discrete_values.csv'.format(dic_path)

        # Physio All Discrete Value file
        ID_COL_NAME = 'ITEMID'
        VAL_COL_NAME = 'VALUE'

        with open(discrete_value_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                raw_mimic_itemid = str(row[ID_COL_NAME])
                value = row[VAL_COL_NAME]
                
                if not self.discrete_dic.has_key(raw_mimic_itemid):
                    self.discrete_dic[raw_mimic_itemid] = []

                self.discrete_dic[raw_mimic_itemid].append(value)

        # Physio CSV file
        ID_COL_NAME = 'ITEMID'
        CAT_COL_NAME = 'CATEGORY'
        LABEL_COL_NAME = 'LABEL'
        PROCESS_COL_NAME = 'PROCESS'
        NORMAL_L_COL_NAME = 'NORMAL_L'
        NORMAL_H_COL_NAME = 'NORMAL_H'

        with open(physio_csv_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                raw_mimic_itemid = str(row[ID_COL_NAME])
                category = row[CAT_COL_NAME]
                label = row[LABEL_COL_NAME]
                process = row[PROCESS_COL_NAME]

                if process == 'DISCRETIZE_WITH_RANGE':
                    normal_l = row[NORMAL_L_COL_NAME]
                    normal_h = row[NORMAL_H_COL_NAME]
                    self.range_dic[raw_mimic_itemid] = \
                        {'normal_l': normal_l,'normal_h': normal_h}

                if process in ['DISCRETIZE_WITH_RANGE', 'DISCRETE_TYPE']:
                    self.chartevent_dic[raw_mimic_itemid] = {}
                    self.chartevent_dic[raw_mimic_itemid]['label'] \
                        = category + ': ' + label
                    self.chartevent_dic[raw_mimic_itemid]['value_type'] \
                        = process

        return None

    def generate_itemid_dic(self, dic):

        assert len(self.discrete_dic) > 0 and len(self.range_dic) > 0 and len(
            dic) > 0

        _dic = {}

        # process DISCRETE types
        for raw_mimic_itemid, values in self.discrete_dic.iteritems():

            self.chart_raw_mimic_ids.append(raw_mimic_itemid)

            for value in values:
                _itemid = str(raw_mimic_itemid)

                if not self.excl_chart_abnormal:
                     _itemid += '-' + value
                     
                raw_label = self.chartevent_dic[raw_mimic_itemid]['label']
                
                if not self.excl_chart_abnormal:
                     raw_label += '-' + value

                _dic[_itemid] = {}
                _dic[_itemid]['label'] = raw_label

        # process REAL types
        for raw_mimic_itemid in self.range_dic.keys():
            self.chart_raw_mimic_ids.append(raw_mimic_itemid)
            for value in ['NORMAL', 'ABNORMAL_LOW', 'ABNORMAL_HIGH']:
                _itemid = str(raw_mimic_itemid)

                if not self.excl_chart_abnormal:
                    _itemid +=  '-' + value
                
                raw_label = self.chartevent_dic[raw_mimic_itemid]['label']
                
                _dic[_itemid] = {}
                if not self.excl_chart_abnormal:
                    raw_label += '-' + value

                _dic[_itemid]['label'] = raw_label 

        return _dic

    def retrieve_itemid(self, mimic_itemidx, value):

        if mimic_itemidx in self.chart_raw_mimic_ids:

            if mimic_itemidx in self.range_dic.keys():
                if self.range_dic[mimic_itemidx]['normal_l'] > value:
                    _value = 'ABNORMAL_LOW'

                elif self.range_dic[mimic_itemidx]['normal_h'] < value:
                    _value = 'ABNORMAL_HIGH'
                else:
                    _value = 'NORMAL'

            else:  # case of discrete type
                _value = value

            itemid = mimic_itemidx + '-' + _value

        else:
            itemid = None

        return itemid


def create_dics(chart_dic, lab_dic=None, re_map=None, is_lab_by_range=False,
                excl_lab_abnormal=False):
    # medication
    dic = {}
    # mimic_medication_list.csv
    dic['drug'] = create_dic(
        dic_path + 'INPUTVENTS_MV_PER_ITEM_COUNTS_more_than_500_icustays.csv', dic, 'ITEMID',
        'new-label', re_map)

    # lab
    if is_lab_by_range:
        dic['lab'] = lab_dic.generate_itemid_dic(dic)

    else:
        dic['lab'] = create_dic(
            dic_path + 'LABEVENT_MV_PER_ICUSTAY_COUNTS_larger_than_500_icustays.csv', dic, 'ITEMID', 'ITEM_LABEL')
        # prev: dic_LAB_filter_100occur.csv

        # lab-abnormal
        if not excl_lab_abnormal:
            dic['abnormal_lab'] = create_dic(
                dic_path + 'dic_LAB_ABNORMAL_100occur.csv', dic, 'ITEMID',
                'ITEM_LABEL')
        else:
            dic['abnormal_lab'] = {}

    # procedure
    dic['proc'] = create_dic(
        dic_path + 'PROCEDUREEVENT_MV_PER_ITEM_COUNTS_more_than_500_icustays.csv', dic, 'ITEMID',
        'ITEM_LABEL')
    #prev: dic_PROCEDURE_filter_100occur.csv 
    
    # chart
    dic['chart'] = chart_dic.generate_itemid_dic(dic)

    np.save(dic_path + 'dic.npy', dic)

    return dic


def merge_itemdics(dic):
    item_dic = {}
    for dic_type, dic_entries in dic.iteritems():
        for item_id, item_info in dic_entries.iteritems():
            label = item_info['label']
            # item_idx = item_info['item_idx']
            item_dic[item_id] = {'label': label, 'category': dic_type}
    return item_dic


def fetch_admission(min_span_day, max_span_day):
    # connect and fetch from MIMIC-DB
    db = MySQLdb.connect(host=DBHOST,
                         user=DBUSER,
                         passwd=DBPASS,
                         db=DBNAME)
    cur = db.cursor()
    hadm_id_query = """
    SELECT ICUSTAY_ID, INTIME, OUTTIME, LOS
    FROM mimiciiiv14.ICUSTAYS
    WHERE DBSOURCE = 'metavision' 
        AND LOS BETWEEN {min_span_day} AND {max_span_day}
    ORDER BY ICUSTAY_ID, INTIME
    """.format(**locals())

    # fetch
    dic_hadm = {}
    cur.execute(hadm_id_query)
    for row in cur.fetchall():
        icustay_id = row[0]
        admittime = row[1]
        dischtime = row[2]
        los = row[3]
        
        if icustay_id not in dic_hadm:
            dic_hadm[icustay_id] = []

        dic_hadm[icustay_id].append(
            {'admittime': admittime, 'dischtime': dischtime, 'los': los})
    db.close()
    print(strftime("%Y-%m-%d %H:%M:%S") + ' fetch admission done.')
    return dic_hadm


# 1. Drug


def fetch_drug_data_inputevents_cv():
    raise NotImplementedError(
        'if needed, copy from other med_timeseries_extractor.py file')
    drug_data = None
    return drug_data


def create_drug_itemid_label():
    inputs_itemid_label_map = {}
    dic_csv_file = dic_path + 'inputevents_itemid_to_relabel.csv'
    with open(dic_csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            inputs_itemid_label_map[row['ITEMID']] = row['LABEL']
    return inputs_itemid_label_map


def fetch_drug_data_inputevents_mv(hadm_ids):
    if os.path.isfile(data_path + 'drug_data_mv_icustay.npy'):
        print(strftime("%Y-%m-%d %H:%M:%S") + ' load drug (mv) data')
        return np.load(data_path + 'drug_data_mv_icustay.npy').item()
    drug_data = {}
    # load itemid => label mapping
    # inputs_itemid_label_map = create_drug_itemid_label()
    db = MySQLdb.connect(host=DBHOST,
                         user=DBUSER,
                         passwd=DBPASS,
                         db=DBNAME)
    cur = db.cursor()

    drug_query = """
    SELECT ICUSTAY_ID, ITEMID, STARTTIME, ENDTIME
    FROM INPUTEVENTS_MV
    WHERE ICUSTAY_ID IN ({});
    """.format(hadm_ids)

    cur.execute(drug_query)
    for row in cur.fetchall():
        hadm_id = row[0]
        itemid = row[1]
        startdate = row[2]
        enddate = row[3]
        # change itemid to string label : when no key found,
        #   skip current drug (drug only)
        # try:
        # drug = inputs_itemid_label_map[str(itemid)]
        # except KeyError:
        # continue
        # add to the vectors
        if not drug_data.has_key(hadm_id):
            drug_data[hadm_id] = []
        drug_data[hadm_id].append(
            {'drug': itemid, 'startdate': startdate, 'enddate': enddate})
    db.close()
    np.save(data_path + 'drug_data_mv_icustay.npy', drug_data)
    return drug_data


def fetch_lab_data(hadm_ids):
    if os.path.isfile(data_path + 'lab_data_icustay.npy'):
        print(strftime("%Y-%m-%d %H:%M:%S") + ' load lab data')
        return np.load(data_path + 'lab_data_icustay.npy').item()
    lab_data = {}
    db = MySQLdb.connect(host=DBHOST,
                         user=DBUSER,
                         passwd=DBPASS,
                         db=DBNAME)
    cur = db.cursor()
    lab_query = """
    SELECT ICUSTAY_ID, ITEMID, CHARTTIME, FLAG
    FROM CS3750_Group6.LABEVENTS_MV
    WHERE ICUSTAY_ID IN ({});
    """.format(hadm_ids)

    cur.execute(lab_query)
    for row in cur.fetchall():
        hadm_id = row[0]
        itemid = row[1]
        charttime = row[2]
        flag = row[3]
        # add to the vectors
        try:
            if len(lab_data[hadm_id]) > 0:
                pass
        except KeyError:
            lab_data[hadm_id] = []
        lab_data[hadm_id].append(
            {'itemid': itemid, 'charttime': charttime, 'flag': flag})
    db.close()
    np.save(data_path + 'lab_data_icustay.npy', lab_data)
    return lab_data


# SELECT DISTINCT A.ITEMID, B.CATEGORY, B.FLUID, B.LABEL, COUNT(SUBJECT_ID)
# FROM LABEVENTS A
# LEFT JOIN D_LABITEMS B
# ON A.ITEMID = B.ITEMID
# GROUP BY B.CATEGORY, B.FLUID, B.LABEL;


def fetch_proc_data(hadm_ids):
    if os.path.isfile(data_path + 'proc_data_icustay.npy'):
        print(strftime("%Y-%m-%d %H:%M:%S") + ' load procedure data')
        return np.load(data_path + 'proc_data_icustay.npy').item()
    proc_data = {}
    db = MySQLdb.connect(host=DBHOST,
                         user=DBUSER,
                         passwd=DBPASS,
                         db=DBNAME)
    cur = db.cursor()
    proc_query = """
    SELECT ICUSTAY_ID, ITEMID, STARTTIME, ENDTIME
    FROM PROCEDUREEVENTS_MV
    WHERE ICUSTAY_ID IN ({});
    """.format(hadm_ids)


    cur.execute(proc_query)
    rows = cur.fetchall()
    bar = IncrementalBar('Processing data', max=len(rows))
    for row in rows:
        hadm_id = row[0]
        itemid = row[1]
        starttime = row[2]
        endtime = row[3]
        # add to the vectors
        try:
            if len(proc_data[hadm_id]) > 0:
                pass
        except KeyError:
            proc_data[hadm_id] = []
        proc_data[hadm_id].append(
            {'itemid': itemid, 'starttime': starttime, 'endtime': endtime})
        bar.next()
    bar.finish()
    db.close()
    np.save(data_path + 'proc_data_icustay.npy', proc_data)
    return proc_data


def fetch_chart_data(hadm_ids, chart_rawids, excl_chart_abnormal=False):
    f_name = 'chart_data_icustay.npy'
    
    if excl_chart_abnormal:
        f_name = 'chart_data_icustay_exclab.npy'

    if os.path.isfile(data_path + f_name):
        print(strftime("%Y-%m-%d %H:%M:%S") + ' load chart data')
        return np.load(data_path + f_name).item()
    
    
    chart_rawids = ','.join(map(lambda x: str(x), chart_rawids))
    chart_data = {}
    db = MySQLdb.connect(host=DBHOST,
                         user=DBUSER,
                         passwd=DBPASS,
                         db=DBNAME)
    cur = db.cursor()
    lab_query = """
    SELECT ICUSTAY_ID, ITEMID, CHARTTIME, VALUE
    FROM CHARTEVENTS 
    WHERE ERROR != 1
        AND ICUSTAY_ID IN ({})
        AND ITEMID IN ({});
    """.format(hadm_ids, chart_rawids)
    print('chart: fetch from db: start')
    cur.execute(lab_query)
    rows = cur.fetchall()
    print('chart: fetch from db: finished')
    bar = IncrementalBar('Processing data', max=len(rows))
    for row in rows:
        hadm_id = row[0]
        itemid = row[1]
        charttime = row[2]
        value = row[3]
        # add to the vectors
        try:
            if len(chart_data[hadm_id]) > 0:
                pass
        except KeyError:
            chart_data[hadm_id] = []
        chart_data[hadm_id].append(
            {'itemid': itemid, 'charttime': charttime, 'value': value})
        bar.next()
    bar.finish()
    db.close()
    np.save(data_path + f_name, chart_data)
    return chart_data


def get_med_timeseries(dic, chart_dic, lab_dic, hadm_ids, 
                       chart_rawids, lab_rawids,
                       testmode=False,
                       use_drug_data=True, use_lab_data=True,
                       use_proc_data=True, use_chart_data=True,
                       is_lab_by_range=False, 
                       excl_lab_abnormal=False,
                       excl_chart_abnormal=False,
                       allow_load=True):

    ts_file = 'medTS_MV_lab_med_proc_chart_icustay'
    if is_lab_by_range:
        ts_file += '_labrange'
    if excl_lab_abnormal:
        ts_file += '_exclablab'
    if excl_chart_abnormal:
        ts_file += '_exclabchart'
    if testmode:
        ts_file += '_test'
    ts_file += '_mimicid'
    ts_file += '.npy'

    if os.path.isfile(data_path + ts_file) and not testmode and allow_load:
        print(strftime("%Y-%m-%d %H:%M:%S") + ' load existing medTS object data')
        return np.load(data_path + ts_file).item()

    # drug_data_cv = fetch_drug_data_inputevents_cv()
    print('len hadm_ids before age filter: {}'.format(len(hadm_ids)))
    hadm_ids = apply_age_filter(metavision_hadm_ids=hadm_ids)
    print('len hadm_ids after age filter: {}'.format(len(hadm_ids)))
    if testmode:
        hadm_testcase_filter = hadm_ids[:40]
    else : 
        hadm_testcase_filter = []
        
    medTS = {}

    hadm_ids = ','.join(map(lambda x: str(x), hadm_ids))  # make string

    # fill drug :: inputevents_MV
    if use_drug_data:
        drug_data_mv = fetch_drug_data_inputevents_mv(hadm_ids)

        bar = IncrementalBar('Processing inputevents (MV) data',
                             max=len(drug_data_mv.keys()))
        for count, hadm_id in enumerate(drug_data_mv.keys()):
            bar.next()
            if testmode and (hadm_id not in hadm_testcase_filter):
                continue
            for drug in drug_data_mv[hadm_id]:
                mimic_itemid = str(drug['drug'])
                if dic['drug'].has_key(mimic_itemid):
                    start_time = drug['startdate']
                    end_time = drug['enddate']
                    value = None
                    if not medTS.has_key(hadm_id):
                        medTS[hadm_id] = []
                    medTS[hadm_id].append(
                        dict(mimic_itemid=mimic_itemid, start_time=start_time,
                             end_time=end_time, category='drug'))
        bar.finish()

    # fill lab
    if use_lab_data:
        lab_data = fetch_lab_data(hadm_ids)

        bar = IncrementalBar('Processing lab data', max=len(lab_data.keys()))
        for count, hadm_id in enumerate(lab_data.keys()):
            bar.next()
            if testmode and (hadm_id not in hadm_testcase_filter):
                continue
            for lab in lab_data[hadm_id]:
                # NOTE: both abnormal lab and binary lab order is contained to the TS

                # lab test
                if is_lab_by_range:

                    raw_mimic_itemid = str(lab['itemid'])

                    if raw_mimic_itemid in lab_rawids:
                        is_abnormal = (lab['flag'] == 'abnormal')

                        mimic_itemid_with_value = lab_dic.retrieve_itemid(
                            raw_mimic_itemid, value, is_abnormal)
                        charttime = lab['charttime']

                        if not medTS.has_key(hadm_id):
                            medTS[hadm_id] = []
                        
                        medTS[hadm_id].append(
                            dict(mimic_itemid=mimic_itemid_with_value, 
                                 start_time=charttime,
                                 end_time=charttime, category='lab'))

                else:
                    raw_mimic_itemid = str(lab['itemid'])
                    if dic['lab'].has_key(raw_mimic_itemid):
                        charttime = lab['charttime']
                        value = None  # lab['value']
                        if not medTS.has_key(hadm_id):
                            medTS[hadm_id] = []
                        medTS[hadm_id].append(
                            dict(mimic_itemid=raw_mimic_itemid, start_time=charttime,
                                 end_time=charttime, category='lab'))

                    # abnormal flag
                    if not excl_lab_abnormal and dic['abnormal_lab'].has_key(raw_mimic_itemid):
                        if lab['flag'] == 'abnormal':
                            if dic['abnormal_lab'].has_key(raw_mimic_itemid):
                                mimic_itemid_with_value = raw_mimic_itemid + '-ABNORMAL' 
                                medTS[hadm_id].append(
                                    dict(mimic_itemid=mimic_itemid_with_value, 
                                         start_time=charttime,
                                         end_time=charttime, category='abnormal_lab'))

        bar.finish()

    # fill procedure
    if use_proc_data:
        proc_data = fetch_proc_data(hadm_ids)

        bar = IncrementalBar('Processing procedure data',
                             max=len(proc_data.keys()))
        for hadm_id in proc_data.keys():
            bar.next()
            if testmode and (hadm_id not in hadm_testcase_filter):
                continue
            for proc in proc_data[hadm_id]:

                mimic_itemid = str(proc['itemid'])

                if dic['proc'].has_key(mimic_itemid):
                    start_time = proc['starttime']
                    end_time = proc['endtime']
                    value = None
                    if not medTS.has_key(hadm_id):
                        medTS[hadm_id] = []
                    medTS[hadm_id].append(
                        dict(mimic_itemid=mimic_itemid, start_time=start_time,
                             end_time=end_time, category='proc'))
        bar.finish()

    # fill chartevent
    if use_chart_data:
        chart_data = fetch_chart_data(hadm_ids, chart_rawids)

        bar = IncrementalBar('Processing chart data', max=len(chart_data.keys()))
        for hadm_id in chart_data.keys():
            bar.next()
            if testmode and (hadm_id not in hadm_testcase_filter):
                continue
            for chart in chart_data[hadm_id]:
                mimic_itemid = str(chart['itemid'])

                if mimic_itemid in chart_dic.chartevent_dic.keys():

                    value = chart['value']

                    charttime = chart['charttime']

                    if not excl_chart_abnormal:
                        mimic_itemid = chart_dic.retrieve_itemid(mimic_itemid, value)

                    if not medTS.has_key(hadm_id):
                        medTS[hadm_id] = []
                    medTS[hadm_id].append(
                        dict(mimic_itemid=mimic_itemid, start_time=charttime,
                             end_time=charttime, category='chart'))

        bar.finish()

    # sort within starttime
    bar = IncrementalBar(
        'Processing sorting by charttime (starttime) data',
        max=len(medTS.keys()))
    for hadm_id in medTS.keys():
        medTS[hadm_id] = sorted(medTS[hadm_id], key=lambda k: k['start_time'])
        bar.next()
    bar.finish()

    # remove key with None
    if medTS.has_key(None):
        del medTS[None]

    # check it
    # for x in sorted(medTS.values()[2], key=lambda k: k[1]):
    #     print x

    print('number of entries in medTS: {}'.format(len(medTS)))

    # save to disk
    print(strftime("%Y-%m-%d %H:%M:%S") +
          ' start to save medTS object on disk: {}'.format(ts_file))

    np.save(data_path + ts_file, medTS)

    print(strftime("%Y-%m-%d %H:%M:%S") + ' save done.')
    return medTS


def save_train_test_valid_subset(medTS, dic, ratio_train, ratio_test,
                                 ratio_valid, max_span_day=20, min_span_day=4,
                                 testmode=False, is_lab_by_range=False,
                                 excl_lab_abnormal=False, 
                                 excl_chart_abnormal=False, 
                                 skip_valid=False, seed=0):
    # get admission id list
    hadms = medTS.keys()

    # # filter out admissions based on MAX and MIN span
    # _hadm_list = deepcopy(hadms)
    # for hadm_id in hadms:
    #     _hadm_id = int(hadm_id)
    #     start_time = medTS[_hadm_id][0]['start_time']
    #     end_time = medTS[_hadm_id][-1]['end_time']
    #     span = (end_time - start_time).total_seconds()

    #     if (span > max_span_day * (3600 * 24) or 
    #             span < min_span_day * (3600 * 24)):
    #         _hadm_list.remove(hadm_id)
    #         continue

    # print(
    #     '[INFO] number of admissions remain: {}, '
    #     'before filter out based on MIN & MAX span:{}'.format(
    #         len(_hadm_list), len(hadms)))

    # hadms = _hadm_list

    if testmode:
        hadms = hadms[:40]

    # shuffle hadm list
    random.seed(seed)
    random.shuffle(hadms)

    # normalize the ratio
    if skip_valid:
        total = ratio_train + ratio_test
    else:
        total = ratio_train + ratio_test + ratio_valid
    ratio_train = ratio_train / total
    ratio_test = ratio_test / total

    hadm_train = hadms[:int(ratio_train * len(hadms))]
    hadm_test = hadms[int(ratio_train * len(hadms)): int(
        (ratio_train + ratio_test) * len(hadms))]
    hadm_valid = hadms[int((ratio_train + ratio_test) * len(hadms)):]

    medTS_train, medTS_test, medTS_valid = {}, {}, {}
    count_occur_events = {}

    # put into each and check occurrences in each.
    bar = IncrementalBar(
        'Processing train data (put and count occurrences)',
        max=len(hadm_train))
    for x in hadm_train:
        x = int(x)
        medTS_train[x] = medTS[x]
        for event in medTS_train[x]:
            mimic_itemid = event['mimic_itemid']
            if not count_occur_events.has_key(mimic_itemid):
                if skip_valid:
                    count_occur_events[mimic_itemid] = {'train': 0, 'test': 0}
                else:
                    count_occur_events[mimic_itemid] = {'train': 0, 'test': 0,
                                                   'valid': 0}
            count_occur_events[mimic_itemid]['train'] += 1
        bar.next()
    bar.finish()

    bar = IncrementalBar('Processing test data (put and count occurrences)',
                         max=len(hadm_test))
    for x in hadm_test:
        x = int(x)
        medTS_test[x] = medTS[x]
        for event in medTS_test[x]:
            mimic_itemid = event['mimic_itemid']
            if not count_occur_events.has_key(mimic_itemid):
                if skip_valid:
                    count_occur_events[mimic_itemid] = {'train': 0, 'test': 0}
                else:
                    count_occur_events[mimic_itemid] = {'train': 0, 'test': 0,
                                                   'valid': 0}
            count_occur_events[mimic_itemid]['test'] += 1
        bar.next()
    bar.finish()

    if not skip_valid:
        bar = IncrementalBar(
            'Processing valid data (put and count occurrences)',
            max=len(hadm_valid))
        for x in hadm_valid:
            x = int(x)
            medTS_valid[x] = medTS[int(x)]
            for event in medTS_valid[x]:
                mimic_itemid = event['mimic_itemid']
                if not count_occur_events.has_key(mimic_itemid):
                    count_occur_events[mimic_itemid] = {'train': 0, 'test': 0,
                                                   'valid': 0}
                count_occur_events[mimic_itemid]['valid'] += 1
            bar.next()
        bar.finish()

    cnt_tr = reduce(lambda x, y: x + y,
                    [len(eventlist) for eventlist in medTS_train.values()])
    cnt_te = reduce(lambda x, y: x + y,
                    [len(eventlist) for eventlist in medTS_test.values()])

    cnt_va = 0
    if not skip_valid:
        cnt_va = reduce(lambda x, y: x + y,
                        [len(eventlist) for eventlist in medTS_valid.values()])

    print('total number of datapoints : tr:{}, te:{}, va:{}, total:{}'.format(
        cnt_tr, cnt_te, cnt_va, cnt_tr + cnt_te + cnt_va))

    # find out items that not occur at least one of train, test, valid sets
    remove_items = []
    for vec_idx, countinfo in count_occur_events.iteritems():
        if not reduce(lambda x, y: x * y, countinfo.values()):
            remove_items.append(vec_idx)

    # remove those items from dic
    count_occur_events_ = deepcopy(count_occur_events)
    for k, v in count_occur_events_.iteritems():
        if k in remove_items:
            del count_occur_events[k]

    fname = 'MV_count_occur_events'

    def decorate_fname(fname):
        if is_lab_by_range:
            fname += '_labrange'
        if excl_lab_abnormal:
            fname += '_exclablab'
        if excl_chart_abnormal:
            fname += '_exclabchart'
        if testmode:
            fname += '_TEST'
        if seed > 0:
            fname += '_split_{}'.format(seed)
        if max_span_day != 20:
            fname += '_maxsd_{}'.format(max_span_day)
        if min_span_day != 4:
            fname += '_minsd_{}'.format(min_span_day)
        if skip_valid:
            fname += '_sv'
        fname += '_mimicid'
        fname += '.npy'
        return fname

    fname = decorate_fname(fname)
    
    np.save(data_path + fname, count_occur_events)

    print('number of items to be removed '
    '(as it is not seen in one of train/test/valid sets): {}'.format(
        len(remove_items)))
    cnt_remove_tr = cnt_remove_te = cnt_remove_va = 0

    print('count overlapping time points')
    names = ['train', 'test']
    medTS_list = [medTS_train, medTS_test]
    if not skip_valid:
        names.append('valid')
        medTS_list.append(medTS_valid)

    for i, medTS in enumerate(medTS_list):
        counters = []
        for hadmid, eventlist in medTS.iteritems():
            timestamps = [x['start_time'] for x in eventlist]
            counters.append(Counter(timestamps).values())
        counters = [item for sublist in counters for item in sublist]
        counters = Counter(counters)
        print('key: number of co-occuring events in a single time point. \
            \nvalue: number of times that happend in {} set\n{}'.format(
            names[i], counters))

        fname = '{}/count_overlapping_time_points_{}'.format(log_path, names[i])
        fname = decorate_fname(fname)
        np.save(fname, counters)

    print(strftime("%Y-%m-%d %H:%M:%S") + 'start to save train set')

    fname = 'medTS_MV_train_instances'
    fname = decorate_fname(fname)
    np.save(data_path + fname, medTS_train)
    print(strftime("%Y-%m-%d %H:%M:%S") + 'start to save test set')

    fname = 'medTS_MV_test_instances'
    fname = decorate_fname(fname)
    np.save(data_path + fname, medTS_test)

    if not skip_valid:
        print(strftime("%Y-%m-%d %H:%M:%S") + 'start to save valid set')
        fname = 'medTS_MV_valid_instances'
        fname = decorate_fname(fname)
        np.save(data_path + fname, medTS_valid)

    # medTS_valid = np.load('medTS_MV_valid_instances.npy').item()
    # medTS_test = np.load('medTS_MV_test_instances.npy').item()
    # medTS_train = np.load('medTS_MV_train_instances.npy').item()


def main():
    

    parser = ArgumentParser(description='To run Sequence Instance Generator')
    parser.add_argument('--min-span-day', dest='min_span_day', type=float,
                        default=4)
    parser.add_argument('--max-span-day', dest='max_span_day', type=float,
                        default=20)
    parser.add_argument('--testmode', dest='testmode', action="store_true",
                        default=False)
    parser.add_argument('--lab-range', dest='lab_range', action="store_true",
                        default=False)
    parser.add_argument('--dic-only', dest='dic_only', action="store_true",
                        default=False)
    parser.add_argument('--not-allow-load', dest='allow_load', action="store_false",
                        default=False)
    parser.add_argument('--skip-valid', dest='skip_valid', action="store_true",
                        default=False)
    parser.add_argument('--multi-split', dest='multi_split', type=int,
                        default=0)
    parser.add_argument('--excl-lab-abnormal', dest='excl_lab_abnormal', action="store_true",
                        default=False)
    parser.add_argument('--excl-chart-abnormal', dest='excl_chart_abnormal', action="store_true",
                        default=False)

    args = parser.parse_args()
    print('-----------------------')
    for arg in sorted(vars(args)):  # print all args
        itm = str(getattr(args, arg))
        print('{0: <20}: {1}'.format(arg, itm))  #
    print('-----------------------')

    dic_hadm = fetch_admission(args.min_span_day, args.max_span_day)
    hadm_ids = dic_hadm.keys()

    re_map = create_id_remap('{}/mimic_itemid_remap.csv'.format(dic_path), 'ORIGID', 'NEWID')
    # NOTE: remap is re-assign certain mimic items as a data preprocessing
    # this merges certain items that can be considered same but exist in different
    # item ids in mimic database. (e.g., items with different volume, units)
    np.save(dic_path + 're_map.npy', re_map)

    chart_dic = ChartEventDic(excl_chart_abnormal=args.excl_chart_abnormal)
    chart_event_mimic_ids = chart_dic.chart_raw_mimic_ids
    lab_dic = LabEventDic(excl_lab_abnormal=args.excl_lab_abnormal)
    dic = create_dics(chart_dic, lab_dic, re_map, is_lab_by_range=args.lab_range,
                      excl_lab_abnormal=args.excl_lab_abnormal)
    
    item_dic = merge_itemdics(dic)

    f_name = 'itemdic'
    if args.lab_range:
        f_name += '_labrange'
    if args.excl_lab_abnormal:
        f_name += '_exclablab'
    f_name += '_maxsd_{}'.format(args.max_span_day)
    f_name += '_minsd_{}'.format(args.min_span_day)
    if args.testmode:
        f_name += '_TEST'

    f_name += '.npy'

    np.save(data_path + f_name, item_dic)
    
    # med vector size
    vec_len = len(dic['drug']) + len(dic['lab']) + \
              len(dic['proc']) + len(dic['chart'])

    if not args.lab_range and not args.excl_lab_abnormal:
        vec_len += len(dic['abnormal_lab'])

    print('total |E| in dics : {}'.format(vec_len))
    print('drug: {}\nlab: {}\n\nprocedure: {}\nchart: {}'.format(
        len(dic['drug']), len(dic['lab']), len(dic['proc']), len(dic['chart'])))

    if not args.lab_range and not args.excl_lab_abnormal:
        print('lab abnormal: {}'.format(len(dic['abnormal_lab'])))

    if args.dic_only:
        exit()

    # medvec item index dic

    # get time series and save on disk
    medTS = get_med_timeseries(dic, chart_dic, lab_dic, hadm_ids,
                               chart_rawids=chart_event_mimic_ids,
                               lab_rawids=lab_dic.rawid_dic.keys(),
                               testmode=args.testmode,
                               use_chart_data=not args.testmode,
                               is_lab_by_range=args.lab_range,
                               excl_lab_abnormal=args.excl_lab_abnormal,
                               excl_chart_abnormal=args.excl_chart_abnormal,
                               allow_load=args.allow_load)

    if args.multi_split > 0:
        for seed in range(1, args.multi_split + 1):
            print('split : {}/{}'.format(seed, args.multi_split))
            save_train_test_valid_subset(
                medTS, dic, ratio_train=0.80, ratio_test=0.20, ratio_valid=0.00,
                testmode=args.testmode,
                max_span_day=args.max_span_day, min_span_day=args.min_span_day,
                is_lab_by_range=args.lab_range, 
                excl_chart_abnormal=args.excl_chart_abnormal,
                excl_lab_abnormal=args.excl_lab_abnormal,
                skip_valid=args.skip_valid,
                seed=seed
            )

    else:
        # split train/test/valid sets and remove non-intersecting (non-seen) events
        # and save to disk
        save_train_test_valid_subset(
            medTS, dic, ratio_train=0.70, ratio_test=0.20, ratio_valid=0.10,
            testmode=args.testmode,
            max_span_day=args.max_span_day, min_span_day=args.min_span_day,
            is_lab_by_range=args.lab_range, 
            excl_chart_abnormal=args.excl_chart_abnormal,
            excl_lab_abnormal=args.excl_lab_abnormal,
            skip_valid=args.skip_valid)


if __name__ == "__main__":
    main()

    """
    # create emprty vectors with 12hour window
    medvecs_12hr = create_empty_12hour_vector(dic_hadm, vec_len)
    # fill 12hour window with data from MIMIC Database
    medvecs_12hr = get_medvector_12hour(dic_hadm, dic['drug'], 
    dic['lab'], dic['abnormal_lab'], dic['proc'], medvecs_12hr)
    medvecs_24hr = get_medvector_24hour()
    """
