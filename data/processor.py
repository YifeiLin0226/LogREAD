import os
import re
import sys
import string
import time 

from logparser.Drain import LogParser as Drain
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

_LOG_FORMAT = {
               'BGL': '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Content>',
               'Thunderbird': '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>',
               'Spirit': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Content>'
               }

class Processor:
    def __init__(self, input_dir, output_dir, dataset, config):
        self.input_dir = input_dir
        assert os.path.exists(self.input_dir), 'Input dir not exist!'
        self.dataset = dataset
        self.output_dir = output_dir
        assert os.path.isdir(self.output_dir), 'Output dir not exist!'

        self.log_file = None
        if dataset == 'Spirit':
            self.log_file = 'spirit_small.log'
        elif dataset == 'BGL':
            self.log_file = 'BGL.log'
        elif dataset == 'Thunderbird':
            self.log_file = 'Thunderbird_20M.log'
        else:
            raise NotImplementedError('Only support Spirit, BGL, and Thunderbird dataset')
    
        self.parser = Drain(log_format = _LOG_FORMAT[dataset], indir = input_dir, outdir = output_dir, **config)
    




    def process(self, parse = False):
        if parse:
            start = time.time()
            self.parser.parse(self.log_file)
            # print('Parsing done in', time.time() - start, 'seconds')
            # print('Average time per log:', (time.time() - start) / len(open(os.path.join(self.input_dir, self.log_file)).readlines()), 'seconds')
            # return
        print('Processing log file')
        # if self.dataset == 'HDFS':
        #     df = pd.read_csv(os.path.join(self.output_dir, 'HDFS.log_structured.csv'), dtype = {'Date': str, 'Time': str})
        #     df['BlockId'] = df['ParameterList'].str.extract(r'(blk_-?\d+)', expand=False)
        #     label_df = pd.read_csv(os.path.join(self.input_dir, "preprocessed", "anomaly_label.csv"))
        #     label_map = {'Normal': 0, 'Anomaly': 1}
        #     label_df['Label'] = label_df['Label'].map(label_map)
        #     label_dict = dict(zip(label_df['BlockId'], label_df['Label']))
        #     df['Label'] = df['BlockId'].map(label_dict)
            

        if self.dataset == 'BGL':
            df = pd.read_csv(os.path.join(self.output_dir, "BGL.log_structured.csv"))
            df['Datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
            df['Timestamp'] = df['Datetime'].values.astype(np.int64) // 10 ** 9
            df['DeltaT'] = df['Datetime'].diff() / np.timedelta64(1, 's')
            df['DeltaT'] = df['DeltaT'].fillna(0)
            df['Label'] = np.where(df['Label'] == '-', 0, 1)
        
        elif self.dataset == 'Thunderbird':
            df = pd.read_csv(os.path.join(self.output_dir, "Thunderbird_20M.log_structured.csv"))
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
            df['Timestamp'] = df['Datetime'].values.astype(np.int64) // 10 ** 9
            df['DeltaT'] = df['Datetime'].diff() / np.timedelta64(1, 's')
            df['DeltaT'] = df['DeltaT'].fillna(0)
            df['Label'] = np.where(df['Label'] == '-', 0, 1)
        
        elif self.dataset == 'Spirit':
            df = pd.read_csv(os.path.join(self.output_dir, "spirit_small.log_structured.csv"))
            df['Label'] = np.where(df['Label'] == '-', 0, 1)

        df['Index'] = df.index

        with open(os.path.join(self.input_dir, self.log_file), 'r') as f:
            loglines = f.readlines()

        # if self.dataset == 'HDFS':
        #     series = pd.Series(loglines)
        #     df['BlockId'] = series.str.extract(r'(blk_-?\d+)', expand=False)
        #     df['Label'] = df['BlockId'].map(label_dict)
        #     del series
        
        df['Label'] = df['Label'].astype(int)
        
        
        # stop_words = set(stopwords.words('english'))
        # tokenizer = RegexpTokenizer(r'\w+')
        for i, s in enumerate(loglines):
            start = time.time()
            s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
            s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
            s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
            s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
            trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
            content = s.translate(trantab)
            s = " ".join([word.lower().strip() for word in content.strip().split()])
            end = time.time()
            print('Time:', end - start)
            loglines[i] = s

        if self.dataset == 'BGL' or self.dataset == 'Thunderbird' or self.dataset == 'Spirit':
            new_df = pd.DataFrame({'Log': loglines, 'Label': df['Label'], 'Index': df['Index']})

            
        # else:
        #     new_df = pd.DataFrame({'Log': loglines, 'BlockId': df['BlockId'], 'Label': df['Label'], 'Index': df['Index']})
        #     group_df = new_df.groupby('BlockId')
        #     # take grouped logs in a list and the max label
        #     new_df = group_df.agg({'Log': lambda x: list(x), 'Label': 'max', 'Index': 'min'}).reset_index()

        #     new_df = new_df.sort_values(by='Index').drop('Index', axis=1)

        new_df.to_csv(os.path.join(self.output_dir, 'processed_log.csv'), index = False)
        print('Processing done!')

















    # def process(self, parse = False):
    #     if parse:
    #         self.parser.parse(self.log_file)

    #     print('Processing log file')
    #     if self.dataset == 'HDFS':
    #         df = pd.read_csv(os.path.join(self.output_dir, 'HDFS.log_structured.csv'), dtype = {'Date': str, 'Time': str})
    #         df['BlockId'] = df['ParameterList'].str.extract(r'(blk_-?\d+)', expand=False)
    #         label_df = pd.read_csv(os.path.join(self.input_dir, "preprocessed", "anomaly_label.csv"))
    #         label_map = {'Normal': 0, 'Anomaly': 1}
    #         label_df['Label'] = label_df['Label'].map(label_map)
    #         label_dict = dict(zip(label_df['BlockId'], label_df['Label']))
    #         df['Label'] = df['BlockId'].map(label_dict)
            

    #     elif self.dataset == 'BGL':
    #         df = pd.read_csv(os.path.join(self.output_dir, "BGL.log_structured.csv"))
    #         df['Datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    #         df['Timestamp'] = df['Datetime'].values.astype(np.int64) // 10 ** 9
    #         df['DeltaT'] = df['Datetime'].diff() / np.timedelta64(1, 's')
    #         df['DeltaT'] = df['DeltaT'].fillna(0)
    #         df['Label'] = np.where(df['Label'] == '-', 0, 1)
        
    #     elif self.dataset == 'Thunderbird':
    #         df = pd.read_csv(os.path.join(self.output_dir, "Thunderbird_20M.log_structured.csv"))
    #         df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
    #         df['Timestamp'] = df['Datetime'].values.astype(np.int64) // 10 ** 9
    #         df['DeltaT'] = df['Datetime'].diff() / np.timedelta64(1, 's')
    #         df['DeltaT'] = df['DeltaT'].fillna(0)
    #         df['Label'] = np.where(df['Label'] == '-', 0, 1)

    #     df['Index'] = df.index

    #     with open(os.path.join(self.input_dir, self.log_file), 'r') as f:
    #         loglines = f.readlines()

    #     if self.dataset == 'HDFS':
    #         series = pd.Series(loglines)
    #         df['BlockId'] = series.str.extract(r'(blk_-?\d+)', expand=False)
    #         df['Label'] = df['BlockId'].map(label_dict)
    #         del series
        
    #     df['Label'] = df['Label'].astype(int)
        
        
    #     stop_words = set(stopwords.words('english'))
    #     tokenizer = RegexpTokenizer(r'\w+')
    #     for i, line in enumerate(loglines):
    #         words = tokenizer.tokenize(line)
    #         words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    #         loglines[i] = ' '.join(words)

    #     if self.dataset == 'BGL' or self.dataset == 'Thunderbird':
    #         new_df = pd.DataFrame(columns = ['Log', 'Label'])
    #         window_size = 5 * 60 if self.dataset == 'BGL' else 60
    #         step_size = 60 if self.dataset == 'BGL' else 30

    #         time_data = df['Timestamp']
    #         start_time = time_data[0]
    #         end_time = start_time + window_size
    #         start_index = 0
    #         end_index = 0

    #         start_end_index_pair = set()

    #         for cur_time in time_data:
    #             if cur_time < end_time:
    #                 end_index += 1
    #             else:
    #                 break
            
    #         start_end_index_pair.add((start_index, end_index))

    #         while end_index < len(time_data):
    #             start_time = start_time + step_size
    #             end_time = start_time + window_size
    #             if start_index == end_index:
    #                 start_time = time_data[start_index]
    #                 end_time = start_time + window_size
    #             i = start_index
    #             j = end_index
    #             while i < len(time_data):
    #                 if time_data[i] < start_time:
    #                     i += 1
    #                 else:
    #                     break
    #             while j < len(time_data):
    #                 if time_data[j] < end_time:
    #                     j += 1
    #                 else:
    #                     break
                
    #             start_index = i
    #             end_index = j

    #             if start_index != end_index:
    #                 start_end_index_pair.add((start_index, end_index))

    #         length_set = set()
    #         for start_index, end_index in tqdm(start_end_index_pair):
    #             logs = loglines[start_index:end_index]
    #             label = df['Label'].iloc[start_index:end_index].max()
    #             # keep logs in a list
    #             new_df.loc[len(new_df)] = [logs, label]
    #             length_set.add(len(logs))
            
    #         print('Length set:', length_set)

            
    #     else:
    #         new_df = pd.DataFrame({'Log': loglines, 'BlockId': df['BlockId'], 'Label': df['Label'], 'Index': df['Index']})
    #         group_df = new_df.groupby('BlockId')
    #         # take grouped logs in a list and the max label
    #         new_df = group_df.agg({'Log': lambda x: list(x), 'Label': 'max', 'Index': 'min'}).reset_index()

    #         new_df = new_df.sort_values(by='Index').drop('Index', axis=1)

    #     new_df.to_csv(os.path.join(self.output_dir, 'processed_log.csv'), index = False)
    #     print('Processing done!')