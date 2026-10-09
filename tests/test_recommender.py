import os
import sys

local_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, local_path)

import gobrec
import numpy as np
import torch


class MockLabelEncoder:
    """Minimal label encoder for testing the recommender."""

    def __init__(self, classes):
        self.classes_ = np.asarray(classes)

    def transform(self, labels):
        labels = np.asarray(labels)
        mapping = {
            label: index
            for index, label in enumerate(self.classes_)
        }
        return np.array([mapping[label] for label in labels])

    def inverse_transform(self, indices):
        return self.classes_[indices]


class MockMABAlgo:
    """Mock MAB algorithm with deterministic predictions."""

    def __init__(self, predictions, classes):
        self.predictions = torch.tensor(
            predictions, dtype=torch.float64
        )
        self.label_encoder = MockLabelEncoder(classes)
        self.fit_args = None
        self.reset_called = False

    def fit(self, contexts, decisions, rewards):
        self.fit_args = (contexts, decisions, rewards)

    def predict(self, contexts):
        assert len(contexts) == len(self.predictions)
        return self.predictions.clone()

    def reset(self):
        self.reset_called = True






def test_fit_delegates_to_mab_algo():
    contexts = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
    ])
    decisions = np.array(["a", "b", "c"])
    rewards = np.array([1.0, 2.0, 3.0])

    mab = MockMABAlgo(
        predictions=np.zeros((3, 3)),
        classes=["a", "b", "c"],
    )
    recommender = gobrec.Recommender(mab_algo=mab, top_k=2)

    recommender.fit(contexts, decisions, rewards)

    fitted_contexts, fitted_decisions, fitted_rewards = mab.fit_args

    np.testing.assert_array_equal(fitted_contexts, contexts)
    np.testing.assert_array_equal(fitted_decisions, decisions)
    np.testing.assert_array_equal(fitted_rewards, rewards)


def test_reset_delegates_to_mab_algo():
    mab = MockMABAlgo(
        predictions=np.zeros((2, 3)),
        classes=["a", "b", "c"],
    )
    recommender = gobrec.Recommender(mab_algo=mab, top_k=2)

    recommender.reset()

    assert mab.reset_called

def test_recommend_returns_top_k_items():
    contexts = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
    ])

    predictions = [
        [0.1, 0.9, 0.5],
        [0.8, 0.2, 0.6],
        [0.3, 0.4, 0.7],
    ]

    mab = MockMABAlgo(
        predictions=predictions,
        classes=["a", "b", "c"],
    )
    recommender = gobrec.Recommender(mab_algo=mab, top_k=2)

    recommendations, scores = recommender.recommend(contexts)

    expected_recommendations = np.array([
        ["b", "c"],
        ["a", "c"],
        ["c", "b"],
    ])
    expected_scores = np.array([
        [0.9, 0.5],
        [0.8, 0.6],
        [0.7, 0.4],
    ])

    np.testing.assert_array_equal(
        recommendations, expected_recommendations
    )
    np.testing.assert_allclose(scores, expected_scores)


def test_recommend_returns_scores_in_descending_order():
    contexts = np.array([
        [1, 0],
        [0, 1],
    ])

    predictions = [
        [0.2, 0.8, 0.5],
        [0.9, 0.1, 0.6],
    ]

    mab = MockMABAlgo(
        predictions=predictions,
        classes=["a", "b", "c"],
    )
    recommender = gobrec.Recommender(mab_algo=mab, top_k=3)

    recommendations, scores = recommender.recommend(contexts)

    assert recommendations.shape == (2, 3)
    assert scores.shape == (2, 3)
    assert np.all(scores == np.sort(np.array(predictions), axis=1)[:, ::-1])
    assert np.all(recommendations == np.array([
        ["b", "c", "a"],
        ["a", "c", "b"],
    ]))


def test_recommend_with_item_filter():
    contexts = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
    ])

    predictions = [
        [0.9, 0.8, 0.7, 0.6],
        [0.9, 0.8, 0.7, 0.6],
        [0.9, 0.8, 0.7, 0.6],
    ]

    mab = MockMABAlgo(
        predictions=predictions,
        classes=["a", "b", "c", "d"],
    )
    recommender = gobrec.Recommender(mab_algo=mab, top_k=2)

    decisions_filter = [
        np.array([0, 0, 1]),
        np.array(["a", "c", "b"]),
    ]

    recommendations, scores = recommender.recommend(
        contexts,
        decisions_filter=decisions_filter,
    )

    assert recommendations.shape == (3, 2)
    assert recommendations[0].tolist() == ["b", "d"]  # "a" and "c" are filtered out from context 0
    assert recommendations[1].tolist() == ["a", "c"]  # "b" is filtered out from context 1
    assert recommendations[2].tolist() == ["a", "b"]  # no filtering for context 2
    assert scores.shape == (3, 2)


