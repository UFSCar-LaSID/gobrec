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

import pandas as pd
import numpy as np


class _LabelEncoder:
    """A simple label encoder to convert item IDs to integer indices and vice versa.

    Different from sklearn's LabelEncoder, this implementation can be updated with new
    classes after fitting. This is useful for MAB algorithms where new items may appear
    over time.

    Attributes
    ----------
    class_to_index : dict[Union[str, int], int]
        A mapping from item IDs, which can be strings or integers, to integer indices.
    index_to_class : list[Union[str, int]]
        A list mapping integer indices back to item IDs.
    """

    def __init__(self):
        """Initialize the label encoder."""
        self.class_to_index: dict[Union[str, int], int] = {}
        self.index_to_class: list[Union[str, int]] = []

    def fit(self, decisions: np.ndarray):
        """Fit the label encoder with the provided item IDs.

        New item IDs will be added to the existing mapping.
        Different from sklearn.label_encoder, this method can
        be called multiple times to update the mapping.

        Parameters
        ----------
        decisions : np.ndarray
            A 1D array of item IDs, which can be strings or integers.
        """
        for cls in decisions:
            if cls not in self.class_to_index:
                idx = len(self.index_to_class)
                self.class_to_index[cls] = idx
                self.index_to_class.append(cls)

    def transform(self, decisions: np.ndarray) -> np.ndarray:
        """Transform item IDs to encoded integer indices.

        Parameters
        ----------
        decisions : np.ndarray
            A 1D array of item IDs, which can be strings or integers.
        
        Returns
        -------
        indices : np.ndarray
            A 1D array of integer indices corresponding to the item IDs.
        """
        return np.array([self.class_to_index[cls] for cls in decisions])

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        """Transform encoded integer indices back to item IDs.

        Parameters
        ----------
        indices : np.ndarray
            A 1D array of integer indices.
        
        Returns
        -------
        item_ids : np.ndarray
            A 1D array of item IDs corresponding to the integer indices.
        """
        return np.array([self.index_to_class[idx] for idx in indices])

    @property
    def classes_(self):
        """Array of item IDs known to the encoder."""
        return np.array(self.index_to_class)



class BaseIrecWrapper(BaseWrapper):
    def __init__(self, context_size: int = None):
        self._init()
    
    def _init(self):
        self.label_encoder = _LabelEncoder()
        self.agent = SimpleAgent(
            value_function=self.value_function,
            action_selection_policy=self.action_selection_policy,
            name="iRecAgent"
        )
        self.PLACEHOLDER_USER = 0
    
    def fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray, num_items: int = None):
        """Fit the iRec model to the interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing ITEM_ID_COLUMN and RATING_COLUMN.
            contexts (np.ndarray): Numpy array containing contexts for each interaction (not used).
        """
        # Incorporate placeholder user
        array_data = interactions_df[[ITEM_ID_COLUMN, RATING_COLUMN]].to_numpy()
        self.label_encoder.fit(array_data[:, 0])
        array_data = np.hstack((np.zeros((array_data.shape[0], 1)), self.label_encoder.transform(array_data[:, 0]).reshape(-1, 1), array_data[:, 1].reshape(-1, 1))).astype(int)
    
        # Create dataset
        train_dataset = Dataset(data=array_data)
        train_dataset.set_parameters()
        train_dataset.update_num_total_users_items(num_total_items=num_items+1)
        self.candidate_items = np.arange(len(self.label_encoder.classes_))
        self.num_total_items = train_dataset.num_total_items

        # Reset agent with dataset
        self.agent.reset(train_dataset)

        # Set info, if needed
        self.info = self.get_info()
        
        self.partial_fit(interactions_df, contexts)

    def partial_fit(self, interactions_df: pd.DataFrame, contexts: np.ndarray):
        """
        Incrementally fit the iRec model with new interactions data.
        
        Parameters:
            interactions_df (pd.DataFrame): DataFrame containing ITEM_ID_COLUMN and RATING_COLUMN.
            contexts (np.ndarray): Numpy array containing contexts for each interaction (not used).
        """
        # Update the agent by observing all interactions
        for _, row in interactions_df.iterrows():
            item_id = [row[ITEM_ID_COLUMN]]
            self.label_encoder.fit(item_id)
            item_id = self.label_encoder.transform(item_id)[0]
            action = (item_id, item_id)
            reward = float(row[RATING_COLUMN])
            self.agent.observe(observation=None, action=action, reward=reward, info=self.info)
        self.candidate_items = np.arange(len(self.label_encoder.classes_))
    
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
            recommended_items.append(self.label_encoder.inverse_transform(actions[1]))
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
    
    def __init__(self, context_size: int = None):
        # Value Function
        self.value_function = LinearUCB(
            alpha=LINUCB_ALPHA,
            num_lat=context_size,
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
    
    def __init__(self, context_size: int = None):
        # Value Function
        self.value_function = LinearEGreedy(
            num_lat=context_size,
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
    
    def __init__(self, context_size: int = None):
        # Value Function
        self.value_function = LinearThompsonSampling(
            num_lat=context_size,
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
