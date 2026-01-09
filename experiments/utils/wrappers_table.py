
import pandas as pd
from utils.mab2rec_wrappers import LinUCBMab2RecWrapper, LinGreedyMab2RecWrapper, LinTSMab2RecWrapper
from utils.gobrec_wrappers import LinUCBGobrecWrapperCPU, LinGreedyGobrecWrapperCPU, LinTSGobrecWrapperCPU, \
    LinUCBGobrecWrapperGPU, LinGreedyGobrecWrapperGPU, LinTSGobrecWrapperGPU
from utils.irec_wrappers import LinUCBIrecWrapper, LinGreedyIrecWrapper, LinTSIrecWrapper


WRAPPERS_TABLE = pd.DataFrame(
    [[1,     'mab2rec_LinUCB',        LinUCBMab2RecWrapper],
     [2,     'mab2rec_LinGreedy',     LinGreedyMab2RecWrapper],
     [3,     'mab2rec_LinTS',         LinTSMab2RecWrapper],
     [4,     'gobrec_LinUCB_CPU',     LinUCBGobrecWrapperCPU],
     [5,     'gobrec_LinGreedy_CPU',  LinGreedyGobrecWrapperCPU],
     [6,     'gobrec_LinTS_CPU',      LinTSGobrecWrapperCPU],
     [7,     'gobrec_LinUCB_GPU',     LinUCBGobrecWrapperGPU],
     [8,     'gobrec_LinGreedy_GPU',  LinGreedyGobrecWrapperGPU],
     [9,     'gobrec_LinTS_GPU',      LinTSGobrecWrapperGPU],
     [10,    'irec_LinUCB',           LinUCBIrecWrapper],
     [11,    'irec_LinGreedy',        LinGreedyIrecWrapper],
     [12,    'irec_LinTS',            LinTSIrecWrapper]],
    columns=['id', 'name', 'AlgoWrapper']
).set_index('id')