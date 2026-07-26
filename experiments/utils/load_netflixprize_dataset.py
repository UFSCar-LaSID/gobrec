
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def preprocess_netflixprize(input_path: str, output_path: str):
    records = []

    filenames = [
        os.path.join(input_path, "combined_data_1.txt"),
        os.path.join(input_path, "combined_data_2.txt"),
        os.path.join(input_path, "combined_data_3.txt"),
        os.path.join(input_path, "combined_data_4.txt")
    ]

    for filename in filenames:
        current_movie = None

        with open(filename, "r") as f:
            for line in tqdm(f, desc=f"Processing {filename}"):
                line = line.strip()

                if line.endswith(":"):
                    current_movie = int(line[:-1])
                    continue

                _, rating, date = line.split(",")

                records.append((date, current_movie, int(rating)))
    
    interactions_df = pd.DataFrame(records, columns=["date", "item_id", "rating"])
    interactions_df = interactions_df.sort_values(by="date").reset_index(drop=True)
    interactions_df = interactions_df.drop(columns=["date"])

    contexts = np.random.uniform(-5, 5, size=(interactions_df.shape[0], 18))

    os.makedirs(output_path, exist_ok=True)

    interactions_df.to_csv(os.path.join(output_path, 'interactions.csv'), index=False)
    np.save(os.path.join(output_path, 'contexts.npy'), contexts)

def load_netflixprize():
    if not os.path.exists('./datasets/netflixprize/preprocessed/interactions.csv') and not os.path.exists('./datasets/netflixprize/preprocessed/contexts.npy'):
        preprocess_netflixprize('./datasets/netflixprize', './datasets/netflixprize/preprocessed')
    
    interactions = pd.read_csv('./datasets/netflixprize/preprocessed/interactions.csv')
    contexts = np.load('./datasets/netflixprize/preprocessed/contexts.npy')

    return interactions, contexts

print(load_netflixprize())