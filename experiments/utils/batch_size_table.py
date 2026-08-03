
import pandas as pd

BATCH_SIZE_TABLE = pd.DataFrame(
    [[1,   '10k',  10_000],
     [2,   '50k',  50_000],
     [3,   '100k', 100_000]],
    columns=['id', 'name', 'value']
).set_index('id')
