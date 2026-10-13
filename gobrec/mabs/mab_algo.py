"""Abstract class for Multi-Armed Bandit algorithms.

This class defines the interface for MAB algorithms to be used in the GOBRec
recommender system.
"""



from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import torch
from gobrec.mabs import LabelEncoder



class MABAlgo(ABC):
    """Abstract class for Multi-Armed Bandit algorithms.

    This class defines the interface for MAB algorithms to be used in the GOBRec
    recommender system. It also includes commom label encoding functionality to handle
    item IDs.

    Attributes
    ----------
    seed : int
        Random seed for reproducibility.
    rng : np.random.Generator
        Random number generator, initialized with the provided seed.
    label_encoder : LabelEncoder
        A label encoder to convert item IDs to integer indices and vice versa.
    num_arms : int
        The number of unique arms (items) known to the algorithm. It starts as None and
        is updated when fitting with new item IDs.
    num_features : int
        The number of features in the context vectors. It starts as None and is set
        when fitting for the first time. It is asserted to be consistent in subsequent
        fits.
    """

    def __init__(self, seed: int = None):
        """Initialize the MAB algorithm.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Default is None.
        """
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.label_encoder = None
        self.num_arms = None
        self.num_features = None
    
    def _update_label_encoder(self, decisions: np.ndarray, num_features: int):
        """Update the label encoder with new item IDs if any.

        It also sets the new number of arms and checks for consistency in the
        number of features. It is recommended to call this method at the beginning
        of the `fit` method of subclasses.

        Parameters
        ----------
        decisions : np.ndarray
            A 1D array of item IDs, which can be strings or integers.
        num_features : int
            The number of features in the context vectors.
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        self.label_encoder.fit(decisions)
        self.num_arms = len(self.label_encoder.classes_)
        
        if self.num_features is None:
            self.num_features = num_features
        
        assert num_features == self.num_features, "Number of features has changed!"
        

    @abstractmethod
    def fit(self, contexts: np.ndarray, decisions: np.ndarray, rewards: np.ndarray):
        """Fit the MAB algorithm with the provided contexts, item IDs, and rewards.

        Parameters
        ----------
        contexts : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the context arrays.
        decisions : np.ndarray
            A 1D array of item IDs (arms or decisions) of shape (n_samples,)
            where each element can be strings or integers.
        rewards : np.ndarray
            A 1D array of rewards (ratings) of shape (n_samples,). It can be integers
            or floats.
        """
        pass

    @abstractmethod
    def predict(self, contexts: np.ndarray) -> torch.Tensor:
        """Predict the expected rewards for the given contexts.

        Parameters
        ----------
        contexts : np.ndarray
            A 2D array of shape (n_samples, n_features) representing the context arrays
        
        Returns
        -------
        expected_rewards : torch.Tensor
            A 2D tensor of shape (n_samples, n_arms) representing the expected
            rewards for each arm (item) given the contexts. The encoded items
            ids are used here. To get the original item IDs, it is possible to
            use the `label_encoder.inverse_transform` method.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the MAB algorithm to its initial state.

        This method should clear any learned parameters and reinitialize the
        random number generator with the original seed.

        This will make the algorithm behave as if it was just initialized.
        """
        self.label_encoder = None
        self.num_arms = None
        self.num_features = None
        self.rng = np.random.default_rng(self.seed)