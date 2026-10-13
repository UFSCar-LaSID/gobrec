
import os
import sys

local_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, local_path)

import gobrec
import numpy as np
import pytest
import torch


class MockRNG:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def random(self, size=None):
        self.calls.append(size)
        value = np.asarray(next(self.values))

        assert value.shape == tuple(size)

        return value


class MockRandomMAB(gobrec.mabs.random_mab.RandomMAB):
    def __init__(self, seed=None, rng_values=None):
        super().__init__(seed=seed)
        self._mock_rng = MockRNG(
            [] if rng_values is None else rng_values
        )

    @property
    def rng(self):
        return self._mock_rng

    @rng.setter
    def rng(self, value):
        # MABAlgo.__init__ and reset() assign to this property.
        self._mock_rng = value


def test_init():
    mab = MockRandomMAB(seed=42)

    assert mab.seed == 42
    assert isinstance(mab.rng, MockRNG)
    assert mab.label_encoder is None
    assert mab.num_arms is None
    assert mab.num_features is None


def test_fit_initializes_label_encoder_and_dimensions():
    contexts = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    decisions = np.array(["a", "b", "a"])
    rewards = np.array([1.0, 2.0, 3.0])

    mab = MockRandomMAB(seed=42)
    mab.fit(contexts, decisions, rewards)

    assert mab.label_encoder is not None
    assert mab.num_arms == 2
    assert mab.num_features == 2


def test_predict_returns_expected_random_scores():
    contexts = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    expected_scores = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ])

    mab = MockRandomMAB(
        seed=42,
        rng_values=[expected_scores],
    )
    mab.num_arms = 2

    scores = mab.predict(contexts)

    assert scores.shape == (3, 2)

    assert np.allclose(scores.numpy(), expected_scores)
    assert mab.rng.calls == [(3, 2)]


def test_predict_requests_correct_shape_from_rng():
    contexts = np.zeros((4, 3))
    expected_scores = np.arange(8).reshape(4, 2) / 10.0

    mab = MockRandomMAB(
        seed=42,
        rng_values=[expected_scores],
    )
    mab.num_arms = 2

    mab.predict(contexts)

    assert mab.rng.calls == [(4, 2)]


def test_predict_calls_rng_once_per_prediction():
    contexts = np.zeros((2, 1))

    values1 = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])
    values2 = np.array([
        [0.5, 0.6],
        [0.7, 0.8],
    ])

    mab = MockRandomMAB(
        seed=42,
        rng_values=[values1, values2],
    )
    mab.num_arms = 2

    first = mab.predict(contexts)
    second = mab.predict(contexts)

    assert mab.rng.calls == [(2, 2), (2, 2)]

    assert np.allclose(first.numpy(), values1)
    assert np.allclose(second.numpy(), values2)



def test_reset_clears_internal_state():
    mab = MockRandomMAB(seed=42)
    
    mab.reset()

    assert mab.label_encoder is None
    assert mab.num_arms is None
    assert mab.num_features is None
    assert mab.seed == 42
    assert mab.rng is not None
