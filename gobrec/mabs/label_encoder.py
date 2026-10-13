"""Label encoder for Multi-Armed Bandit (MAB) algorithms.

This module provides a simple label encoder that can convert item IDs (which can be strings or integers) to integer indices and vice versa. It is designed to be used in MAB algorithms where new items may appear over time, allowing the encoder to be updated with new classes after fitting.
"""


import numpy as np
from typing import Union



class LabelEncoder:
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