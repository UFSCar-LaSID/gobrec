
from utils.datasets import DATASETS_TABLE
from utils.wrappers_table import WRAPPERS_TABLE
from utils.constants import RESULTS_SAVE_PATH, NUM_EXECUTIONS_PER_EXPERIMENT, TRAIN_TIME_COLUMN, RECS_TIME_COLUMN, TOTAL_TIME_COLUMN, CONTEXTS_PER_BATCH, ITEM_ID_COLUMN
from utils.parameters_handle import get_input
from utils.BaseWrapper import BaseWrapper

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time



def execute_not_incremental_experiment(wrapper: BaseWrapper, interactions_df: pd.DataFrame, contexts: np.ndarray, save_path: str):
    df_save_path = os.path.join(save_path, 'not_incremental.csv')
    num_necessary_executions = get_num_necessary_executions(df_save_path)

    num_items = interactions_df[ITEM_ID_COLUMN].nunique()

    split_index = int(len(interactions_df) * 0.5)

    train_df = interactions_df.copy()[:split_index]

    train_contexts = contexts[:split_index]
    test_contexts = contexts[split_index:]
    
    for _ in tqdm(range(num_necessary_executions), desc='Executing not incremental experiment'):
        start_time = time()
        for start in range(0, len(train_contexts), CONTEXTS_PER_BATCH):
            if start == 0:
                wrapper.fit(train_df[start:start+CONTEXTS_PER_BATCH], train_contexts[start:start+CONTEXTS_PER_BATCH], num_items=num_items)
            else:
                wrapper.partial_fit(train_df[start:start+CONTEXTS_PER_BATCH], train_contexts[start:start+CONTEXTS_PER_BATCH])
        fit_time = time() - start_time

        start_time = time()
        for start in range(0, len(test_contexts), CONTEXTS_PER_BATCH):
            wrapper.recommend(test_contexts[start:start+CONTEXTS_PER_BATCH])
        recommend_time = time() - start_time

        full_time = fit_time + recommend_time

        results = {
            TRAIN_TIME_COLUMN: fit_time,
            RECS_TIME_COLUMN: recommend_time,
            TOTAL_TIME_COLUMN: full_time
        }
        results_df = pd.DataFrame([results])
        results_df.to_csv(df_save_path, mode='a', header=not os.path.exists(df_save_path), index=False)

        wrapper.reset()
        

def execute_incremental_experiment(wrapper: BaseWrapper, interactions_df: pd.DataFrame, contexts: np.ndarray, save_path: str):
    NUM_WINDOWS = 10
    df_save_path = os.path.join(save_path, 'incremental.csv')
    num_necessary_executions = get_num_necessary_executions(df_save_path)

    num_items = interactions_df[ITEM_ID_COLUMN].nunique()

    split_index = int(len(interactions_df) * 0.5)

    train_df = interactions_df.copy()[:split_index]
    test_df = interactions_df.copy()[split_index:]

    train_contexts = contexts[:split_index]
    test_contexts = contexts[split_index:]

    results = {}

    for _ in tqdm(range(num_necessary_executions), desc='Executing incremental experiment'):

        results = {}

        start_time = time()
        
        for start in range(0, len(train_contexts), CONTEXTS_PER_BATCH):
            if start == 0:
                wrapper.fit(train_df[start:start+CONTEXTS_PER_BATCH], train_contexts[start:start+CONTEXTS_PER_BATCH], num_items=num_items)
            else:
                wrapper.partial_fit(train_df[start:start+CONTEXTS_PER_BATCH], train_contexts[start:start+CONTEXTS_PER_BATCH])
        fit_time = time() - start_time
        results[TRAIN_TIME_COLUMN] = fit_time

        for window_number in range(NUM_WINDOWS):

            current_window_start_index = int(len(test_df) * (window_number / NUM_WINDOWS))
            current_window_end_index = int(len(test_df) * ((window_number + 1) / NUM_WINDOWS))

            current_window_df = test_df.iloc[current_window_start_index:current_window_end_index]
            current_window_contexts = test_contexts[current_window_start_index:current_window_end_index]

            start_time = time()
            for start in range(0, len(current_window_contexts), CONTEXTS_PER_BATCH):
                wrapper.recommend(current_window_contexts[start:start+CONTEXTS_PER_BATCH])
            recommend_time = time() - start_time
            results[RECS_TIME_COLUMN + f'_{window_number+1}'] = recommend_time

            if window_number != NUM_WINDOWS - 1:
                start_time = time()
                for start in range(0, len(current_window_df), CONTEXTS_PER_BATCH):
                    wrapper.partial_fit(current_window_df[start:start+CONTEXTS_PER_BATCH], current_window_contexts[start:start+CONTEXTS_PER_BATCH])
                partial_fit_time = time() - start_time
                results[TRAIN_TIME_COLUMN + f'_{window_number+1}'] = partial_fit_time

        full_time = sum(results.values())
        results[TOTAL_TIME_COLUMN] = full_time
        
        results_df = pd.DataFrame([results])
        results_df.to_csv(df_save_path, mode='a', header=not os.path.exists(df_save_path), index=False)

        wrapper.reset()



def get_num_necessary_executions(file_path: str):
    if os.path.exists(file_path):
        return NUM_EXECUTIONS_PER_EXPERIMENT - pd.read_csv(file_path).shape[0]
    return NUM_EXECUTIONS_PER_EXPERIMENT

def get_experiment_save_path(dataset_name, wrapper_name):
    save_path = os.path.join(RESULTS_SAVE_PATH, dataset_name, wrapper_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path




EXPERIMENTS_TABLE = pd.DataFrame(
    [[1, 'not_incremental', execute_not_incremental_experiment],
     [2, 'incremental', execute_incremental_experiment]],
    columns=['id', 'name', 'experiment_function']
).set_index('id')


datasets_options, wrappers_options, experiments_options = get_input(
    'Select the options to be used in the experiments',
    [
        {
            'name': 'datasets',
            'description': 'Datasets to be used in the experiments',
            'name_column': 'name',
            'options': DATASETS_TABLE
        },
        {
            'name': 'algorithms',
            'description': 'Algorithms to be used in the experiments',
            'name_column': 'name',
            'options': WRAPPERS_TABLE
        },
        {
            'name': 'experiments',
            'description': 'Experiments to be executed',
            'name_column': 'name',
            'options': EXPERIMENTS_TABLE
        }
    ]
)

for dataset_option in datasets_options:
    dataset_name = DATASETS_TABLE.loc[dataset_option, 'name']
    dataset_getter = DATASETS_TABLE.loc[dataset_option, 'get_dataset']
    print(f'Loading dataset {dataset_name}...')
    interactions_df, contexts = dataset_getter()
    print(f'Dataset {dataset_name} loaded...')

    for wrapper_option in wrappers_options:
        wrapper_name = WRAPPERS_TABLE.loc[wrapper_option, 'name']
        WrapperClass = WRAPPERS_TABLE.loc[wrapper_option, 'AlgoWrapper']
        print(f'Using wrapper {wrapper_name}...')
        wrapper = WrapperClass(context_size=contexts.shape[1])

        for experiment_option in experiments_options:
            experiment_name = EXPERIMENTS_TABLE.loc[experiment_option, 'name']
            experiment_function = EXPERIMENTS_TABLE.loc[experiment_option, 'experiment_function']
            print(f'Executing experiment {experiment_name} with wrapper {wrapper_name} on dataset {dataset_name}...')
            experiment_function(wrapper, interactions_df, contexts, get_experiment_save_path(dataset_name, wrapper_name))
            print(f'Experiment {experiment_name} completed.')