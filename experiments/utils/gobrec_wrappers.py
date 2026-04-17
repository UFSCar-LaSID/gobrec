
import gobrec
from utils.constants import TOP_K, SEED, L2_LAMBDA, LINUCB_ALPHA, LINGREEDY_EPSILON, LINTS_ALPHA, ITEM_ID_COLUMN, RATING_COLUMN
from utils.BaseWrapper import BaseWrapper
import pandas as pd
import numpy as np


class BaseGobrecWrapper(BaseWrapper):

    def __init__(self):
        self.gobrec_recommender = gobrec.Recommender(
            mab_algo=self.mab_algo,  # This should be set in the subclass
            top_k=TOP_K
        )
    
    def fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray):
        """
        Fit the GOBRec model to the interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing ITEM_ID_COLUMN and RATING_COLUMN.
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        """
        self.gobrec_recommender.fit(
            decisions=interactions_df[ITEM_ID_COLUMN].values,
            rewards=interactions_df[RATING_COLUMN].values,
            contexts=contexts
        )
    
    def partial_fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray):
        """
        Incrementally fit the GOBRec model with new interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing ITEM_ID_COLUMN and RATING_COLUMN.
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        """
        self.gobrec_recommender.fit(
            decisions=interactions_df[ITEM_ID_COLUMN].values,
            rewards=interactions_df[RATING_COLUMN].values,
            contexts=contexts
        )
    
    def recommend(self, contexts: np.ndarray):
        """
        Recommend items based on the current model state and contexts.
        
        Parameters:
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        
        Returns:
            list: List of recommended item IDs.
        """
        return self.gobrec_recommender.recommend(contexts=contexts)
    
    def reset(self):
        """
        Reset the GOBRec model to its initial state.
        """
        self.gobrec_recommender.reset()


class LinGobrecWrapper(BaseGobrecWrapper):
    """
    Wrapper for the LinUCB algorithm in GOBRec.
    """
    def __init__(self):
        self.mab_algo = gobrec.mabs.lin_mabs.Lin(l2_lambda=L2_LAMBDA, use_gpu=False)
        super().__init__()

class LinUCBGobrecWrapperCPU(BaseGobrecWrapper):
    """
    Wrapper for the LinUCB algorithm in GOBRec.
    """
    def __init__(self):
        self.mab_algo = gobrec.mabs.lin_mabs.LinUCB(l2_lambda=L2_LAMBDA, alpha=LINUCB_ALPHA, use_gpu=False)
        super().__init__()

class LinGreedyGobrecWrapperCPU(BaseGobrecWrapper):
    """
    Wrapper for the LinGreedy algorithm in GOBRec.
    """
    def __init__(self):
        self.mab_algo = gobrec.mabs.lin_mabs.LinGreedy(l2_lambda=L2_LAMBDA, epsilon=LINGREEDY_EPSILON, use_gpu=False)
        super().__init__()

class LinTSGobrecWrapperCPU(BaseGobrecWrapper):
    """
    Wrapper for the LinGreedy algorithm in GOBRec.
    """
    def __init__(self):
        self.mab_algo = gobrec.mabs.lin_mabs.LinTS(l2_lambda=L2_LAMBDA, alpha=LINTS_ALPHA, use_gpu=False)
        super().__init__()


class LinUCBGobrecWrapperGPU(BaseGobrecWrapper):
    """
    Wrapper for the LinUCB algorithm in GOBRec.
    """
    def __init__(self):
        self.mab_algo = gobrec.mabs.lin_mabs.LinUCB(l2_lambda=L2_LAMBDA, alpha=LINUCB_ALPHA, use_gpu=True, items_per_batch=1_000)
        super().__init__()

class LinGreedyGobrecWrapperGPU(BaseGobrecWrapper):
    """
    Wrapper for the LinGreedy algorithm in GOBRec.
    """
    def __init__(self):
        self.mab_algo = gobrec.mabs.lin_mabs.LinGreedy(l2_lambda=L2_LAMBDA, epsilon=LINGREEDY_EPSILON, use_gpu=True, items_per_batch=1_000)
        super().__init__()

class LinTSGobrecWrapperGPU(BaseGobrecWrapper):
    """
    Wrapper for the LinGreedy algorithm in GOBRec.
    """
    def __init__(self):
        self.mab_algo = gobrec.mabs.lin_mabs.LinTS(l2_lambda=L2_LAMBDA, alpha=LINTS_ALPHA, use_gpu=True, items_per_batch=1_000)
        super().__init__()
