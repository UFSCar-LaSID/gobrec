import os
import gc
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Queue

from utils.datasets import DATASETS_TABLE
from utils.wrappers_table import WRAPPERS_TABLE
from utils.constants import (
    RESULTS_SAVE_PATH,
    NUM_EXECUTIONS_PER_EXPERIMENT,
    CONTEXTS_PER_BATCH,
    ITEM_ID_COLUMN
)
from utils.parameters_handle import get_input
from utils.BaseWrapper import BaseWrapper


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def memory_experiment_worker(wrapper_class: BaseWrapper, context_size: int, interactions_df: pd.DataFrame, contexts: np.ndarray, queue):
    gc.collect()

    wrapper = wrapper_class(context_size=context_size)

    num_items = interactions_df[ITEM_ID_COLUMN].nunique()
    split_index = int(len(interactions_df) * 0.5)

    train_df = interactions_df[:split_index]
    train_contexts = contexts[:split_index]

    for start in range(0, len(train_contexts), CONTEXTS_PER_BATCH):
        if start == 0:
            wrapper.fit(
                train_df[start:start + CONTEXTS_PER_BATCH],
                train_contexts[start:start + CONTEXTS_PER_BATCH],
                num_items=num_items
            )
        else:
            wrapper.partial_fit(
                train_df[start:start + CONTEXTS_PER_BATCH],
                train_contexts[start:start + CONTEXTS_PER_BATCH]
            )

    gc.collect()

    memory_used = get_memory_usage_mb()

    queue.put(memory_used)


def execute_memory_experiment(wrapper_class, interactions_df, contexts, save_path):
    df_save_path = os.path.join(save_path, 'memory_usage.csv')

    existing_runs = 0
    if os.path.exists(df_save_path):
        existing_runs = pd.read_csv(df_save_path).shape[0]

    num_runs = NUM_EXECUTIONS_PER_EXPERIMENT - existing_runs

    for _ in tqdm(range(num_runs), desc='Executing memory experiment'):
        queue = Queue()

        p = Process(
            target=memory_experiment_worker,
            args=(wrapper_class, contexts.shape[1], interactions_df, contexts, queue)
        )
        p.start()
        p.join()

        memory_used = queue.get()

        results_df = pd.DataFrame([{"memory_mb": memory_used}])
        results_df.to_csv(
            df_save_path,
            mode='a',
            header=not os.path.exists(df_save_path),
            index=False
        )


def get_experiment_save_path(dataset_name, wrapper_name):
    save_path = os.path.join(RESULTS_SAVE_PATH, dataset_name, wrapper_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path


datasets_options, wrappers_options = get_input(
    'Select datasets and algorithms',
    [
        {
            'name': 'datasets',
            'description': 'Datasets to be used',
            'name_column': 'name',
            'options': DATASETS_TABLE
        },
        {
            'name': 'algorithms',
            'description': 'Algorithms to be used',
            'name_column': 'name',
            'options': WRAPPERS_TABLE
        }
    ]
)


for dataset_option in datasets_options:
    dataset_name = DATASETS_TABLE.loc[dataset_option, 'name']
    dataset_getter = DATASETS_TABLE.loc[dataset_option, 'get_dataset']

    print(f'Loading dataset {dataset_name}...')
    interactions_df, contexts = dataset_getter()
    print(f'Dataset {dataset_name} loaded.')

    for wrapper_option in wrappers_options:
        wrapper_name = WRAPPERS_TABLE.loc[wrapper_option, 'name']
        WrapperClass = WRAPPERS_TABLE.loc[wrapper_option, 'AlgoWrapper']

        print(f'Running memory experiment for {wrapper_name} on {dataset_name}...')
        execute_memory_experiment(
            WrapperClass,
            interactions_df,
            contexts,
            get_experiment_save_path(dataset_name, wrapper_name)
        )
        print('Done.')