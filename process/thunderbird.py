import os
import ast

import pandas as pd

from data.processor import Processor


indir  = '/home/datasets/log_data/Thunderbird' # The input directory of log file
outdir = '/home/datasets/log_data/Thunderbird/output'  # The output directory of parsing results

st         = 0.3  # Similarity threshold
depth      = 3  # Depth of all leaf nodes

processor = Processor(indir, outdir, 'Thunderbird',  {'depth':depth, 'st':st})
processor.process(parse = True)

# df = pd.read_csv(os.path.join(outdir, 'processed_log.csv'))
# df['Log'] = df['Log'].apply(ast.literal_eval)
# df = df[df['Log'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
# df.to_csv(os.path.join(outdir, 'processed_log.csv'), index=False)
