"""LinTS: Linear Thompson Sampling implementation."""


from gobrec.mabs.lin_mabs import Lin
import numpy as np
import torch


class LinTS(Lin):
    r"""LinTS: Linear Thompson Sampling algorithm [1]_.

    This class implements a linear MAB algorithm that uses ridge regression to
    estimate the expected rewards for each arm. Then, it samples the coefficients
    from a multivariate normal distribution with mean equal to the estimated coefficients
    and covariance proportional to the inverse of the design matrix.

    Attributes
    ----------
    alpha : float
        Controls the variance of the sampling distribution. Higher values lead to more exploration.
    l2_lambda : float
        Regularization parameter for ridge regression.
    device : str
        Device to use for computations ('cpu' or 'cuda').
    items_per_batch : int
        Number of items to process in each batch when updating the model.
        More items per batch means more memory usage but faster computation.
    
    References
    ----------
    .. [1] Shipra Agrawal and Navin Goyal. Thompson sampling for contextual bandits with 
       linear payoffs. In Proceedings of the 30th International Conference on Machine 
       Learning, ICML'13, pages 1220-1228, New York, NY, USA, 2013. JMLR.org. doi: 
       10.48550/arXiv.1209.3352.
    
    Examples
    --------
    An example using LinTS with :math:`\alpha = 0.5`.

    >>> import numpy as np
    >>> from gobrec.mabs.lin_mabs import LinTS
    >>> contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...                      [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> decisions = np.array(['a', 'a', 'a', 
    ...                       'b', 'b', 'b',
    ...                       'c', 'c', 'c'])
    >>> rewards = np.array([10, 0 , 1 , 
    ...                     1 , 10, 0 ,
    ...                     0 , 1 , 10])
    >>> lints_mab = LinTS(seed=42, alpha=0.5)
    >>> lints_mab.fit(contexts, decisions, rewards)
    >>> lints_mab.predict(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    tensor([[ 5.1077,  0.6077,  0.1077],
            [-0.6898,  4.3102, -0.1898],
            [ 0.4941, -0.0059,  4.9941]], dtype=torch.float64)
    """

    def __init__(self, seed: int = None, alpha: float = 1.0, l2_lambda: float = 1.0, use_gpu: bool = False, items_per_batch: int = 10_000):
        """Initialize LinTS algorithm.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Default is None.
        alpha : float, optional
            Controls the variance of the sampling distribution. Higher values lead to more exploration.
            Default is 1.0.
        l2_lambda : float, optional
            Regularization parameter for ridge regression. Default is 1.0.
        use_gpu : bool, optional
            Whether to use GPU for computations if available. Default is False.
        items_per_batch : int, optional
            Number of items to process in each batch when updating the model.
            More items per batch means more memory usage but faster computation.
            Default is 10,000.
        """
        super().__init__(seed, l2_lambda, use_gpu, items_per_batch)
        self.alpha = alpha

        if self.alpha == 0:
            self.predict = super().predict
    
    def predict(self, contexts: np.ndarray):
        """Predict the expected rewards for each arm given the contexts.

        Parameters
        ----------
        contexts : np.ndarray
            A 2D array where each row represents the context features for which
            predictions are to be made.
        
        Returns
        -------
        expected_rewards : torch.Tensor
            A 2D tensor of shape (n_samples, n_arms) where each element
            is the expected reward for the corresponding context-arm pair.
            The encoded items ids are used here. To get the original item IDs, 
            it is possible to use the `label_encoder.inverse_transform` method.
        """
        x = torch.tensor(contexts, device=self.device, dtype=torch.double)

        num_arms, num_features = self.beta.shape
        num_contexts = contexts.shape[0]

        scores = torch.empty((num_contexts, num_arms), device=self.device, dtype=torch.double)

        eps = torch.from_numpy(self.rng.standard_normal(size=(num_contexts, num_features))).to(device=self.device, dtype=torch.double)

        for start in range(0, num_arms, self.items_per_batch):
            end = min(start + self.items_per_batch, num_arms)  

            beta_chunk = self.beta[start:end]
            A_chunk = self.A[start:end]
            A_inv_chunk = torch.linalg.inv(A_chunk)

            L_chunk = torch.linalg.cholesky((self.alpha ** 2) * A_inv_chunk)
            beta_sampled = torch.einsum('bd,add->bad', eps, L_chunk) + beta_chunk

            scores[:, start:end] = torch.einsum('bd,bad->ba', x, beta_sampled)

        return scores