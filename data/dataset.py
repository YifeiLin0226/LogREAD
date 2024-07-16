import os
import copy
import sys

from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class LogDataset(Dataset):
    def __init__(self, dataset_name, dir, mode = 'train', train_ratio = 0.5):
        self.dataset_name = dataset_name
        self.dir = dir
        data_path = os.path.join(dir, 'processed_log.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found")
        
        dup_data = pd.read_csv(data_path)
        print('Length of the whole dataset:', len(dup_data))
        # print('length before drop duplicates:', len(self.data))
        # self.data = self.data.drop_duplicates(subset = 'Log', keep = 'first')
        # print('length after drop duplicates:', len(self.data))
        dup_data['order_column'] = range(len(dup_data))

        normal = dup_data[dup_data['Label'] == 0]
        abnormal = dup_data[dup_data['Label'] == 1]

        if mode == 'train':
            self.data = normal[:int(len(normal) * train_ratio)].drop_duplicates(subset = 'Log', keep = 'first').reset_index(drop = True)
            
            # synonym_path = os.path.join(dir, f'train_log_synonym_{train_ratio}.csv')
            # if not os.path.exists(synonym_path):
            #     print(f"Creating synonym file {synonym_path}")
            #     # Enable tqdm progress bar for pandas
            #     tqdm.pandas(desc="Processing logs")
    
            #     # Apply the replace_log function with a progress bar
            #     self.data['synonym'] = self.data['Log'].progress_apply(replace_log)
            #     self.data.to_csv(synonym_path, index = False)
            #     sys.exit()
            # else:
            #     self.data = pd.read_csv(synonym_path)

        else:
            rest_norm = normal[int(len(normal) * train_ratio):]
            rest_data = pd.concat([rest_norm, abnormal]).sort_values(by = 'order_column').reset_index(drop = True)
            valid_data = rest_data.sample(frac = 0.1)
            if mode == 'valid':
                self.data = valid_data.reset_index(drop = True)
            elif mode == 'test':
                self.data = rest_data.drop(valid_data.index).reset_index(drop = True)
            else:
                raise ValueError(f"Invalid mode {mode}")
        
        self.data = self.data.drop('order_column', axis = 1)
        self.mode = mode
        print("Length of data:", len(self.data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError
        log = self.data.loc[idx, 'Log']
        label = self.data.loc[idx, 'Label']
        return log, label
    

    def collate_fn(self, batch, tokenizer, augmentation):
        logs, labels = zip(*batch)
        source_tokens = tokenizer(logs, padding = True, truncation = True, max_length = 512)

        input_ids = []
        attention_mask = []
        token_type_ids = []

        if self.mode == 'train':
            if augmentation:
                target_tokens = copy.deepcopy(source_tokens)
                for i in range(len(source_tokens['input_ids'])):
                    input_ids.append(source_tokens['input_ids'][i])
                    input_ids.append(target_tokens['input_ids'][i])
                    attention_mask.append(source_tokens['attention_mask'][i])
                    attention_mask.append(target_tokens['attention_mask'][i])
                    token_type_ids.append(source_tokens['token_type_ids'][i])
                    token_type_ids.append(target_tokens['token_type_ids'][i])
                
    
                return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids)
        
        for i in range(len(source_tokens['input_ids'])):
            input_ids.append(source_tokens['input_ids'][i])
            attention_mask.append(source_tokens['attention_mask'][i])
            token_type_ids.append(source_tokens['token_type_ids'][i])
                

        if self.mode == 'train':
            return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids)
        
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), torch.tensor(labels)
        


        
            


