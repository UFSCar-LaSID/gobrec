
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


@pytest.fixture(params=[False, True], ids=["cpu", "gpu"])
def lin_greedy_factory(request):
    use_gpu = request.param

    if use_gpu and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    def create_lingreedy(**kwargs):
        return gobrec.mabs.lin_mabs.LinGreedy(
            use_gpu=use_gpu,
            **kwargs,
        )

    return create_lingreedy

class MockRNG:
    def __init__(self, values):
        self.values = iter(values)

    def random(self, *args):
        return next(self.values)


# alpha = 0, so it should be equal to Lin
# so, the below codes are the same from lin

def test_predict_one_arm(lin_greedy_factory):
    contexts = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2]])
    decisions = np.array([0, 0, 0])
    rewards = np.array([3, 3, 1])

    lin = lin_greedy_factory(epsilon=0.0, l2_lambda=1.0)

    lin.fit(contexts, decisions, rewards)
    pred = lin.predict(np.array([[0, 1, 2, 3, 5]])).cpu().numpy()

    assert np.allclose(pred, np.array([2.8167701863354]), atol=1e-8)

def test_predict_multiple_arms(lin_greedy_factory):
    decisions = np.array(["a", "a", "a",
                          "b", "b", "b",
                          "c", "c", "c"])
    contexts = np.vstack([np.eye(3), np.eye(3), np.eye(3)])
    rewards = np.array([
        10, 0, 1,
        1, 10, 0,
        0, 1, 10
    ])

    lin = lin_greedy_factory(epsilon=0.0, l2_lambda=1.0)

    lin.fit(contexts, decisions, rewards)

    pred = lin.predict(np.eye(3)).cpu().numpy()

    expected = np.array([
        [5.0, 0.5, 0.0],
        [0.0, 5.0, 0.5],
        [0.5, 0.0, 5.0]
    ])

    assert np.allclose(pred, expected)

def test_fit(lin_greedy_factory):
    contexts = np.array([
        [1, 0, 2, 1, 1],
        [3, 1, 2, 3, 4],
        [2, -1, 1, 0, 2]
    ])

    rewards = np.array([3, 3, 1])
    decisions = np.array([1, 1, 1])

    lin = lin_greedy_factory(epsilon=0.0, l2_lambda=1.0)

    lin.fit(contexts, decisions, rewards)

    assert lin.num_features == 5
    assert lin.num_arms == 1

    expected = np.array([
        0.09161491,
        0.00310559,
        0.97515528,
        0.32142857,
        -0.02018634,
    ])

    assert np.allclose(lin.beta[0].cpu(), expected)

    contexts = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2],  [-1, 4, 2, 0, 1],
                         [2, 2, 2, 2, 2], [3, 2, 1, 2, 3], [0, 0, 0, 0, 0],   [2, 1, 1, 1, 2],
                         [3, 2, 3, 2, 3], [8, 2, 3, 1, 0], [1, 2, -9, -7, 1], [0, 1, 1, 1, 1]])
    rewards = np.array([3, 3, 1, 0, -1, 2, 1, 2, 1, 1, 0, 3])
    decisions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    lin = lin_greedy_factory(epsilon=0.0, l2_lambda=1.0)

    lin.fit(contexts, decisions, rewards)

    expected = np.array([
        0.09927202,
        -0.17141953,
        0.09091367,
        -0.03705452,
        0.59027579,
    ])

    assert np.allclose(lin.beta[0].cpu(), expected)

def test_incremental_fit(lin_greedy_factory):
    # First batch
    context1 = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2],])
    rewards1 = np.array([3, 2, 1])
    decisions1 = np.array([1, 1, 1])

    lin = lin_greedy_factory(epsilon=0.0, l2_lambda=1.0)

    lin.fit(context1, decisions1, rewards1)

    assert lin.num_features == 5
    assert lin.num_arms == 1

    expected_beta_arm1 = np.array([
        0.47619048,
        0.04761905,
        -0.59523810,
        0.14285714,
        0.66666667,
    ])

    assert np.allclose(
        lin.beta[0].cpu().numpy(),
        expected_beta_arm1,
        atol=1e-8,
    )

    # Second batch
    context2 = np.array([[2, 1, 2, 1, 2], [3, 3, 3, 2, 1], [1, 1, 1, 1, 1]])
    rewards2 = np.array([1, 1, 1])
    decisions2 = np.array([0, 0, 1])

    lin.fit(context2, decisions2, rewards2)

    assert lin.num_features == 5
    assert lin.num_arms == 2

    # Arm 0 is new and becomes encoded index 1
    expected_beta_arm0 = np.array([
        0.11940299,
        0.01492537,
        0.11940299,
        0.04477612,
        0.17910448,
    ])

    # Arm 1 was updated
    expected_beta_arm1 = np.array([
        0.53019146,
        0.13402062,
        -0.56553756,
        0.17525773,
        0.61266568,
    ])

    # label_encoder order is [1, 0] because arm 1 was seen first and arm 0 was seen second.
    assert np.array_equal(
        lin.label_encoder.classes_,
        np.array([1, 0])
    )

    assert np.allclose(
        lin.beta[0].cpu().numpy(),
        expected_beta_arm1,
        atol=1e-8,
    )

    assert np.allclose(
        lin.beta[1].cpu().numpy(),
        expected_beta_arm0,
        atol=1e-8,
    )

def test_incremental_vs_full_fit(lin_greedy_factory):
    # Batch fit - all data at once
    contexts_batch = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2],
                               [2, 1, 2, 1, 2], [3, 3, 3, 2, 1], [1, 1, 1, 1, 1]])
    rewards_batch = np.array([0, 1, 1, 0, 1, 0])
    decisions_batch = np.array([1, 1, 1, 0, 0, 1])

    lin_batch = lin_greedy_factory(epsilon=0.0)

    lin_batch.fit(
        contexts_batch,
        decisions_batch,
        rewards_batch,
    )

    # Incremental fit - split data into two batches
    contexts1 = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2]])
    rewards1 = np.array([0, 1, 1])
    decisions1 = np.array([1, 1, 1])

    contexts2 = np.array([[2, 1, 2, 1, 2],[3, 3, 3, 2, 1],[1, 1, 1, 1, 1],])
    rewards2 = np.array([0, 1, 0])
    decisions2 = np.array([0, 0, 1])

    lin_inc = lin_greedy_factory(epsilon=0.0)

    lin_inc.fit(contexts1, decisions1, rewards1)
    lin_inc.fit(contexts2, decisions2, rewards2)

    assert np.array_equal(
        lin_batch.label_encoder.classes_,
        lin_inc.label_encoder.classes_,
    )

    assert np.allclose(
        lin_batch.beta.cpu().numpy(),
        lin_inc.beta.cpu().numpy(),
        atol=1e-8,
    )

    assert np.allclose(
        lin_batch.Xty.cpu().numpy(),
        lin_inc.Xty.cpu().numpy(),
        atol=1e-8,
    )

    assert np.allclose(
        lin_batch.A.cpu().numpy(),
        lin_inc.A.cpu().numpy(),
        atol=1e-8,
    )


def test_epsilon_1(lin_greedy_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]
    ])

    decisions = np.array([
        1, 1, 1, 2, 2,
        3, 3, 3, 3, 3
    ])

    rewards = np.array([
        0, 0, 1, 0, 0,
        0, 1, 1, 1, 1
    ])

    lin_greedy = lin_greedy_factory(
        epsilon=1.0,
        seed=123456,
    )

    lin_greedy.fit(contexts, decisions, rewards)

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1]
    ])

    pred = lin_greedy.predict(test_contexts).cpu().numpy()

    assert pred.shape == (2, 3)
    assert np.all(pred >= 0)
    assert np.all(pred < 1)

def test_epsilon_mixed_with_mocked_rng(lin_greedy_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5]
    ])

    decisions = np.array([1, 2, 3, 1])

    rewards = np.array([1, 2, 3, 4])

    lin_greedy = lin_greedy_factory(
        epsilon=0.5,
    )

    lin_greedy.fit(contexts, decisions, rewards)

    expected_exploitation = gobrec.mabs.lin_mabs.Lin.predict(  # Uses the base Lin class to compute the expected exploitation scores
        lin_greedy,
        contexts
    )

    random_mask = np.array([0.1, 0.9, 0.2, 0.8])

    random_scores = np.array([
        [0.11, 0.22, 0.33],
        [0.44, 0.55, 0.66],
    ])

    lin_greedy.rng = MockRNG([
        random_mask,
        random_scores,
    ])

    pred = lin_greedy.predict(contexts).cpu().numpy()

    expected = expected_exploitation.cpu().numpy()

    expected[0] = random_scores[0]
    expected[2] = random_scores[1]

    assert np.allclose(pred, expected)


def test_seed_reproducibility(lin_greedy_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1]
    ])

    decisions = np.array([1, 1, 2, 2, 3])

    rewards = np.array([0, 1, 0, 1, 1])

    mab1 = lin_greedy_factory(
        seed=123456,
        epsilon=0.5,
    )

    mab2 = lin_greedy_factory(
        seed=123456,
        epsilon=0.5,
    )

    mab1.fit(contexts, decisions, rewards)
    mab2.fit(contexts, decisions, rewards)

    pred1 = mab1.predict(contexts).cpu().numpy()
    pred2 = mab2.predict(contexts).cpu().numpy()

    assert np.allclose(pred1, pred2)

def test_different_seed_changes_exploration(lin_greedy_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1]
    ])

    decisions = np.array([1, 1, 2, 2, 3])

    rewards = np.array([0, 1, 0, 1, 1])

    mab1 = lin_greedy_factory(
        seed=123,
        epsilon=1.0,
    )

    mab2 = lin_greedy_factory(
        seed=456,
        epsilon=1.0,
    )

    mab1.fit(contexts, decisions, rewards)
    mab2.fit(contexts, decisions, rewards)

    pred1 = mab1.predict(contexts).cpu().numpy()
    pred2 = mab2.predict(contexts).cpu().numpy()

    assert not np.allclose(pred1, pred2)

