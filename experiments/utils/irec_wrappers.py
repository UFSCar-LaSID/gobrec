from utils.irec_modified.environment.dataset import Dataset
from utils.irec_modified.recommendation.agents.simple_agent import SimpleAgent
from utils.irec_modified.recommendation.agents.value_functions.linear_ucb import LinearUCB
from utils.irec_modified.recommendation.agents.value_functions.linear_egreedy import LinearEGreedy
from utils.irec_modified.recommendation.agents.value_functions.linear_ts import LinearThompsonSampling
from utils.irec_modified.recommendation.agents.action_selection_policies.greedy import ASPGreedy
from utils.irec_modified.recommendation.agents.action_selection_policies.egreedy import ASPEGreedy

from utils.constants import (
    TOP_K, LINUCB_ALPHA, LINGREEDY_EPSILON, LINTS_ALPHA,
    ITEM_ID_COLUMN, RATING_COLUMN
)
from utils.BaseWrapper import BaseWrapper

# Fixed number of dimensions for context representation
CONTEXT_DIMENSION = 18
# iRec requires the number of items to be known beforehand
NUM_ITEMS = {
    'ml-100k': 1682,
    'ml-1m': 3706,
    'ml-10m': 10677,
}
CURRENT_DATASET = 'ml-100k'

import pandas as pd
import numpy as np


class BaseIrecWrapper(BaseWrapper):
    def __init__(self):
        self._init()
    
    def _init(self):
        self.agent = SimpleAgent(
            value_function=self.value_function,
            action_selection_policy=self.action_selection_policy,
            name="iRecAgent"
        )
        self.PLACEHOLDER_USER = 0
    
    def fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray):
        """Fit the iRec model to the interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing ITEM_ID_COLUMN and RATING_COLUMN.
            contexts (np.ndarray): Numpy array containing contexts for each interaction (not used).
        """
        # Incorporate placeholder user
        array_data = interactions_df[[ITEM_ID_COLUMN, RATING_COLUMN]].to_numpy()
        array_data = np.hstack((np.zeros((array_data.shape[0], 1)), array_data)).astype(int)
    
        # Create dataset
        train_dataset = Dataset(data=array_data)
        train_dataset.set_parameters()
        train_dataset.update_num_total_users_items(num_total_items=NUM_ITEMS[CURRENT_DATASET]+1)
        self.candidate_items = np.arange(train_dataset.num_total_items)
        self.num_total_items = train_dataset.num_total_items

        # Reset agent with dataset
        self.agent.reset(train_dataset)

        # Set info, if needed
        self.info = self.get_info()

    def partial_fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray):
        """
        Incrementally fit the iRec model with new interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing ITEM_ID_COLUMN and RATING_COLUMN.
            contexts (np.ndarray): Numpy array containing contexts for each interaction (not used).
        """
        # Update the agent by observing all interactions
        for _, row in interactions_df.iterrows():
            item_id = row[ITEM_ID_COLUMN]
            action = (self.PLACEHOLDER_USER, item_id)
            reward = float(row[RATING_COLUMN])
            self.agent.observe(observation=None, action=action, reward=reward, info=self.info)
    
    def recommend(self, contexts: np.ndarray):
        """Recommend items based on the current model state and contexts.
        
        Parameters:
            contexts (np.ndarray): Numpy array containing contexts for each interaction.
        
        Returns:
            list: List of recommended item IDs.
        """
        recommended_items = list()
        recommended_scores = list()
        for _ in range(len(contexts)):
            # Get item scores
            candidate_actions = (self.PLACEHOLDER_USER, self.candidate_items) 
            actions_estimate, self.info = self.agent.value_function.actions_estimate(candidate_actions)
            
            # Get recommendations from agent
            actions, self.info = self.agent.act(candidate_actions, TOP_K)
            
            # Extract recommended items
            recommended_items.append(actions[1])
            recommended_scores.append(actions_estimate[:TOP_K])

        recommendations = [np.array(recommended_items), np.array(recommended_scores)]
        return recommendations
    
    def reset(self):
        """Reset the iRec model to its initial state.
        """
        self.agent = SimpleAgent(
            value_function=self.value_function,
            action_selection_policy=self.action_selection_policy,
            name="iRecAgent"
        )


class LinUCBIrecWrapper(BaseIrecWrapper):
    """Wrapper for LinearUCB algorithm in iRec."""
    
    def __init__(self):
        # Value Function
        self.value_function = LinearUCB(
            alpha=LINUCB_ALPHA,
            num_lat=CONTEXT_DIMENSION,
            iterations=20,
            var=0.05,
            user_var=0.01,
            item_var=0.01,
            stop_criteria=0.0009
        )
        # Action Selection Policy
        self.action_selection_policy = ASPGreedy()
        super().__init__()

    def get_info(self):
        return dict()


class LinGreedyIrecWrapper(BaseIrecWrapper):
    """Wrapper for LinearEGreedy algorithm in iRec."""
    
    def __init__(self):
        # Value Function
        self.value_function = LinearEGreedy(
            num_lat=CONTEXT_DIMENSION,
            iterations=20,
            var=0.05,
            user_var=0.01,
            item_var=0.01,
            stop_criteria=0.0009
        )
        # Action Selection Policy
        self.action_selection_policy = ASPEGreedy(epsilon=LINGREEDY_EPSILON)
        super().__init__()
    
    def get_info(self):
        return dict()


class LinTSIrecWrapper(BaseIrecWrapper):
    """Wrapper for LinearTS algorithm in iRec."""
    
    def __init__(self):
        # Value Function
        self.value_function = LinearThompsonSampling(
            num_lat=CONTEXT_DIMENSION,
            iterations=20,
            var=0.05,
            user_var=0.01,
            item_var=0.01,
            stop_criteria=0.0009
        )
        # Action Selection Policy
        self.action_selection_policy = ASPGreedy()
        super().__init__()

    def get_info(self):
        # Generates first recommendation to get info
        candidate_actions = (self.PLACEHOLDER_USER, self.candidate_items) 
        _, info = self.agent.act(candidate_actions, TOP_K)
        return info
