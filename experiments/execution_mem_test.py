import os
import gc
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Pipe
import time

from utils.datasets import DATASETS_TABLE
from utils.wrappers_table import WRAPPERS_TABLE
from utils.batch_size_table import BATCH_SIZE_TABLE
from utils.constants import (
    RESULTS_SAVE_PATH,
    NUM_EXECUTIONS_PER_EXPERIMENT,
    ITEM_ID_COLUMN
)
from utils.parameters_handle import get_input
from utils.BaseWrapper import BaseWrapper
import resource


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def memory_experiment_worker(
    wrapper_class,
    context_size,
    interactions_df,
    contexts,
    conn,
    batch_size
):
    try:
        gc.collect()

        wrapper = wrapper_class(context_size=context_size)

        num_items = interactions_df[ITEM_ID_COLUMN].nunique()
        split_index = int(len(interactions_df) * 0.5)

        train_df = interactions_df[:split_index]
        train_contexts = contexts[:split_index]
        test_contexts = contexts[split_index:]

        start_time = time.time()

        for start in range(0, len(train_contexts), batch_size):
            if start == 0:
                wrapper.fit(
                    train_df[start:start + batch_size],
                    train_contexts[start:start + batch_size],
                    num_items=num_items
                )
            else:
                wrapper.partial_fit(
                    train_df[start:start + batch_size],
                    train_contexts[start:start + batch_size]
                )

        for start in range(0, len(test_contexts), batch_size):
            wrapper.recommend(
                test_contexts[start:start + batch_size]
            )

        gc.collect()

        memory_used = get_memory_usage_mb()
        peak_memory = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss / 1024
        time_spent = time.time() - start_time

        conn.send(("success", memory_used, peak_memory, time_spent))

    except Exception as e:
        conn.send(("error", repr(e)))

    finally:
        conn.close()


def execute_memory_experiment(wrapper_class, interactions_df, contexts, save_path):
    df_save_path = os.path.join(save_path, 'memory_usage.csv')

    existing_runs = 0
    if os.path.exists(df_save_path):
        existing_runs = pd.read_csv(df_save_path).shape[0]

    num_runs = NUM_EXECUTIONS_PER_EXPERIMENT - existing_runs

    for _ in tqdm(range(num_runs), desc='Executing memory experiment'):
        parent_conn, child_conn = Pipe()

        p = Process(
            target=memory_experiment_worker,
            args=(
                wrapper_class,
                contexts.shape[1],
                interactions_df,
                contexts,
                child_conn,
                batch_size
            )
        )

        p.start()

        while True:
            if parent_conn.poll(1):
                status, *result = parent_conn.recv()
                break

            if not p.is_alive():
                print(
                    f"Worker terminated unexpectedly. "
                    f"Exit code: {p.exitcode}"
                )

                raise RuntimeError(
                    f"Worker died with exit code {p.exitcode}"
                )

        p.join()

        if status == "error":
            raise RuntimeError(result[0])

        memory_used, peak_memory, time_spent = result
        
        results_df = pd.DataFrame([{"memory_mb": memory_used, "peak_memory_mb": peak_memory, "time": time_spent}])
        results_df.to_csv(
            df_save_path,
            mode='a',
            header=not os.path.exists(df_save_path),
            index=False
        )


def get_experiment_save_path(dataset_name, wrapper_name, batch_size_name):
    save_path = os.path.join(RESULTS_SAVE_PATH, batch_size_name, dataset_name, wrapper_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path


batch_size_options, datasets_options, wrappers_options = get_input(
    'Select datasets and algorithms',
    [
        {
            'name': 'batch_size',
            'description': 'Batch size',
            'name_column': 'name',
            'options': BATCH_SIZE_TABLE
        },
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

    for batch_size_option in batch_size_options:
        batch_size = BATCH_SIZE_TABLE.loc[batch_size_option, 'value']
        batch_size_name = BATCH_SIZE_TABLE.loc[batch_size_option, 'name']

        for wrapper_option in wrappers_options:
            wrapper_name = WRAPPERS_TABLE.loc[wrapper_option, 'name']
            WrapperClass = WRAPPERS_TABLE.loc[wrapper_option, 'AlgoWrapper']

            print(f'Running memory experiment for {wrapper_name} on {dataset_name} with batch size {batch_size_name}...')
            execute_memory_experiment(
                WrapperClass,
                interactions_df,
                contexts,
                get_experiment_save_path(dataset_name, wrapper_name, batch_size_name)
            )
            print('Done.')