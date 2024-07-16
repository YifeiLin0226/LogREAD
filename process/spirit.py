from data.processor import Processor

input_dir = '/home/datasets/log_data/Spirit'
output_dir = '/home/datasets/log_data/Spirit'
dataset = 'Spirit'
st         = 0.5  # Similarity threshold
depth      = 5  # Depth of all leaf nodes
regex = [ r'(\d+\.){ 3 }\d+', r'(\/.*?\.[ \S: ]+)' ]

config = {'st': st, 'depth': depth, 'rex': regex}

processor = Processor(input_dir, output_dir, dataset, config)
processor.process(parse = True)