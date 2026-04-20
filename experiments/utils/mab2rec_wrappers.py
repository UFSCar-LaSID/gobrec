
from mab2rec import BanditRecommender, LearningPolicy
from utils.constants import TOP_K, SEED, L2_LAMBDA, LINUCB_ALPHA, LINGREEDY_EPSILON, LINTS_ALPHA, ITEM_ID_COLUMN, RATING_COLUMN
from utils.BaseWrapper import BaseWrapper
import pandas as pd
import numpy as np


class BaseMab2recWrapper(BaseWrapper):

    def __init__(self, context_size: int = None):
        self._init()
    
    def _init(self):
        self.mab2rec_recommender = BanditRecommender(
            learning_policy=self.mab2rec_learning_policy,  # This should be set in the subclass
            top_k=TOP_K,
            seed=SEED,
            n_jobs=-1
        )
    
    def fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray, num_items: int = None):
        """
        Fit the MAB2Rec model to the interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing ITEM_ID_COLUMN and RATING_COLUMN.
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        """
        self.mab2rec_recommender.fit(
            decisions=interactions_df[ITEM_ID_COLUMN],
            rewards=interactions_df[RATING_COLUMN],
            contexts=contexts
        )
    
    def partial_fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray):
        """
        Incrementally fit the MAB2Rec model with new interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing ITEM_ID_COLUMN and RATING_COLUMN.
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        """
        new_arms = np.setdiff1d(interactions_df[ITEM_ID_COLUMN].unique(), self.mab2rec_recommender.mab._imp.arms)
        for new_arm in new_arms:
            self.mab2rec_recommender.add_arm(new_arm)
        
        self.mab2rec_recommender.partial_fit(
            decisions=interactions_df[ITEM_ID_COLUMN],
            rewards=interactions_df[RATING_COLUMN],
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
        return self.mab2rec_recommender.recommend(contexts=contexts, return_scores=True, apply_sigmoid=False)

    def reset(self):
        """
        Reset the MAB2Rec model to its initial state.
        """
        self._init()


class LinMab2RecWrapper(BaseMab2recWrapper):
    """
    Wrapper for the LinUCB algorithm in MAB2Rec.
    """
    def __init__(self, context_size: int = None):
        self.mab2rec_learning_policy = LearningPolicy.LinGreedy(l2_lambda=L2_LAMBDA, epsilon=0)
        super().__init__()

class LinUCBMab2RecWrapper(BaseMab2recWrapper):
    """
    Wrapper for the LinUCB algorithm in MAB2Rec.
    """
    def __init__(self, context_size: int = None):
        self.mab2rec_learning_policy = LearningPolicy.LinUCB(l2_lambda=L2_LAMBDA, alpha=LINUCB_ALPHA)
        super().__init__()

class LinGreedyMab2RecWrapper(BaseMab2recWrapper):
    """
    Wrapper for the LinGreedy algorithm in MAB2Rec.
    """
    def __init__(self, context_size: int = None):
        self.mab2rec_learning_policy = LearningPolicy.LinGreedy(l2_lambda=L2_LAMBDA, epsilon=LINGREEDY_EPSILON)
        super().__init__()

class LinTSMab2RecWrapper(BaseMab2recWrapper):
    """
    Wrapper for the LinTS algorithm in MAB2Rec.
    """
    def __init__(self, context_size: int = None):
        self.mab2rec_learning_policy = LearningPolicy.LinTS(l2_lambda=L2_LAMBDA, alpha=LINTS_ALPHA)
        super().__init__()