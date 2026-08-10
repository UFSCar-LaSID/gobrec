
import pandas as pd

BATCH_SIZE_TABLE = pd.DataFrame(
    [
     [1,   '100',  100],
     [2,   '500',  500],
     [3,   '1k',   1_000],
     [4,   '5k',   5_000],
     [5,   '10k',  10_000],
     [6,   '50k',  50_000],
     [7,   '100k', 100_000]],
    columns=['id', 'name', 'value']
).set_index('id')
