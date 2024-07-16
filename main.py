import random
import argparse 
import os
import gc

import numpy as np
import torch
import faiss
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from trainer import Trainer

def seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

class NameSpace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type = str, default = 'bgl', choices = ['bgl', 'thunderbird', 'spirit'])
parser.add_argument('--train_ratio', type = float, default = 0.3)
parser.add_argument('--model_name', type = str, choices = ['bert', 'roberta'], default = 'bert')
parser.add_argument('--device', type = str, default = 'cuda')
parser.add_argument('--pretrain_epochs', type = int, default = 10)
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--batch_size', type = int, default = 4)
parser.add_argument('--lr', type = float, default = 3e-5)
parser.add_argument('--linear', type = str2bool, default = False)
parser.add_argument('--adam_epsilon', type = float, default = 1e-8)
parser.add_argument('--warmup_proportion', type = float, default = 0.1)
parser.add_argument('--weight_decay', type = float, default = 0.01)
parser.add_argument('--gradient_accumulation_steps', type = int, default = 1)
parser.add_argument('--num_nn', type = int, default = 2)
parser.add_argument('--dim_coreset_feat', type = int, default = 256)
parser.add_argument('--percentage', type = float, default = 0.5)
# model config
parser.add_argument('--num_hidden_layers', type = int, default = 1)
parser.add_argument('--layers_extract', nargs='+', type = int, default = [0])

parser.add_argument('--retrain_epochs', type = int, default = 10)
parser.add_argument('--retrain_lr', type = float, default = 3e-6)
parser.add_argument('--retrain_lambda', type = float, default = 0.5)

# parser.add_argument('--window_size', type = int, default = 128)

args = parser.parse_args()
dir_map = {'bgl': '/home/datasets/log_data/BGL/output', 'thunderbird': '/home/datasets/log_data/Thunderbird/output', 'spirit': '/home/datasets/log_data/Spirit/'}
# add custom dataset directory here
# e.g. dir_map['custom'] = 'path/to/custom/dataset'

args.dir = dir_map[args.dataset_name]

def main():
    with open('result.txt', 'a+') as f:
        f.write(f'{args.dataset_name} {args.train_ratio}\n')
        for s in [42]:
            f.write(str(s) + ':\t')
            seed(s)
            trainer = Trainer(args)
            trainer.pretrain()
            trainer.fill_memory()
            trainer.validate()
            result, anomaly_scores, gt = trainer.predict()
            threshold = result['threshold']
            f.write(f'{result}\n')
            result, cur_prec, cur_rec, cur_f1 = trainer.online()
            f.write(f'online: {result}\n')
            del trainer 
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Explicitly deallocate Faiss resources
            res = faiss.StandardGpuResources()
            res.syncDefaultStreamCurrentDevice()
            gc.collect()
            # Resetting the CUDA context by creating a new context
            torch.cuda.init()

                


if __name__ == '__main__':
    main()