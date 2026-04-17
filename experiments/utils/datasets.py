
import os
import pandas as pd
from utils.generate_toy_datasets import load_or_generate_toy_dataset
from utils.load_ml_datasets import load_ml100k, load_ml1m, load_ml10m

from utils.constants import TOY_DATASETS_SAVE_PATH

DATASETS_TABLE = pd.DataFrame(
    [
     [1,     'toy_dataset_1k_100k',    lambda: load_or_generate_toy_dataset(1_000,   100_000,    os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_1k_100k'))],
     [2,     'toy_dataset_1k_500k',    lambda: load_or_generate_toy_dataset(1_000,   500_000,    os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_1k_500k'))],
     [3,     'toy_dataset_1k_1M',      lambda: load_or_generate_toy_dataset(1_000,   1_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_1k_1M'))],
     [4,     'toy_dataset_1k_5M',      lambda: load_or_generate_toy_dataset(1_000,   5_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_1k_5M'))],
     [5,     'toy_dataset_1k_15M',     lambda: load_or_generate_toy_dataset(1_000,   15_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_1k_15M'))],
     [6,     'toy_dataset_1k_30M',     lambda: load_or_generate_toy_dataset(1_000,   30_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_1k_30M'))],
     
     [7,     'toy_dataset_5k_500k',    lambda: load_or_generate_toy_dataset(5_000,   500_000,    os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_5k_500k'))],
     
     [8,     'toy_dataset_10k_100k',   lambda: load_or_generate_toy_dataset(10_000,  100_000,    os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_10k_100k'))],
     [9,     'toy_dataset_10k_500k',   lambda: load_or_generate_toy_dataset(10_000,  500_000,    os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_10k_500k'))],
     [10,    'toy_dataset_10k_1M',     lambda: load_or_generate_toy_dataset(10_000,  1_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_10k_1M'))],
     [11,    'toy_dataset_10k_5M',     lambda: load_or_generate_toy_dataset(10_000,  5_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_10k_5M'))],
     [12,    'toy_dataset_10k_15M',    lambda: load_or_generate_toy_dataset(10_000,  15_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_10k_15M'))],
     [13,    'toy_dataset_10k_30M',    lambda: load_or_generate_toy_dataset(10_000,  30_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_10k_30M'))],
     
     [14,    'toy_dataset_50k_500k',   lambda: load_or_generate_toy_dataset(50_000,  500_000,    os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_50k_500k'))],
     
     [15,    'toy_dataset_100k_500k',  lambda: load_or_generate_toy_dataset(100_000, 500_000,    os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_100k_500k'))],
     [16,    'toy_dataset_100k_1M',    lambda: load_or_generate_toy_dataset(100_000, 1_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_100k_1M'))],
     [17,    'toy_dataset_100k_5M',    lambda: load_or_generate_toy_dataset(100_000, 5_000_000,  os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_100k_5M'))],
     [18,    'toy_dataset_100k_15M',   lambda: load_or_generate_toy_dataset(100_000, 15_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_100k_15M'))],
     [19,    'toy_dataset_100k_30M',   lambda: load_or_generate_toy_dataset(100_000, 30_000_000, os.path.join(TOY_DATASETS_SAVE_PATH, 'toy_dataset_100k_30M'))],
     
     [20,    'ml-100k',                load_ml100k],
     [21,    'ml-1m',                  load_ml1m],
     [22,    'ml-10m',                 load_ml10m]],
    columns=['id', 'name', 'get_dataset']
).set_index('id')