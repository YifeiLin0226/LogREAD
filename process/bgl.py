import os
import ast

import pandas as pd

from data.processor import Processor

indir = '/home/datasets/log_data/BGL'
outdir = '/home/datasets/log_data/BGL/output/'

st         = 0.5  # Similarity threshold
depth      = 5  # Depth of all leaf nodes
regex = [
        r'(0x)[0-9a-fA-F]+', #hexadecimal
        r'\d+.\d+.\d+.\d+',
        # r'/\w+( )$'
        r'\d+'
    ]

processor = Processor(indir, outdir, 'BGL', {'st': st, 'depth': depth, 'rex': regex})
processor.process(parse = False)

df = pd.read_csv(os.path.join(outdir, 'processed_log.csv'))
df['Log'] = df['Log'].apply(ast.literal_eval)
df = df[df['Log'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
df.to_csv(os.path.join(outdir, 'processed_log.csv'), index=False)
