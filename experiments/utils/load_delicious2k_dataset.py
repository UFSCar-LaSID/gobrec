
# Original dataset download: https://grouplens.org/datasets/hetrec-2011/

import os
import pandas as pd
import numpy as np

def preprocess_delicious2k(input_path: str, output_path: str):

    df_interactions = pd.read_csv(os.path.join(input_path, 'user_taggedbookmarks.dat'), sep='\t', header=0, index_col=False, encoding='iso-8859-1')
    df_interactions[['year', 'month', 'day', 'hour', 'minute', 'second']] = df_interactions[['year', 'month', 'day', 'hour', 'minute', 'second']].astype(str)
    df_interactions['datetime'] =  df_interactions['year'].str.zfill(4) + '-' + df_interactions['month'].str.zfill(2) + '-' + df_interactions['day'].str.zfill(2) + ' ' + df_interactions['hour'].str.zfill(2) + ':' + df_interactions['minute'].str.zfill(2) + ':' + df_interactions['second'].str.zfill(2)
    df_interactions = df_interactions.sort_values('datetime')
    df_interactions = df_interactions.drop_duplicates(subset=['userID', 'bookmarkID'], keep='first')
    df_interactions = df_interactions[['bookmarkID']]
    df_interactions['rating'] = 1
    df_interactions = df_interactions.rename(columns={'bookmarkID': 'item_id'})

    contexts = np.random.uniform(-5, 5, size=(df_interactions.shape[0], 18))

    os.makedirs(output_path, exist_ok=True)

    df_interactions.to_csv(os.path.join(output_path, 'interactions.csv'), index=False)
    np.save(os.path.join(output_path, 'contexts.npy'), contexts)


def load_delicious2k():
    if not os.path.exists('./datasets/delicious2k/preprocessed/interactions.csv') and not os.path.exists('./datasets/delicious2k/preprocessed/contexts.npy'):
        preprocess_delicious2k('./datasets/delicious2k', './datasets/delicious2k/preprocessed')
    
    interactions = pd.read_csv('./datasets/delicious2k/preprocessed/interactions.csv')
    contexts = np.load('./datasets/delicious2k/preprocessed/contexts.npy')

    return interactions, contexts
