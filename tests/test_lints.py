
import os
import sys

local_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, local_path)
import gobrec

import numpy as np
import torch
import pytest


class MockRNG:
    def __init__(self, values):
        self.values = iter(values)

    def standard_normal(self, size=None):
        value = np.asarray(next(self.values))

        assert value.shape == tuple(size)

        return value

@pytest.fixture(params=[False, True], ids=["cpu", "gpu"])
def lin_ts_factory(request):
    use_gpu = request.param

    if use_gpu and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    def create_lints(**kwargs):
        return gobrec.mabs.lin_mabs.LinTS(
            use_gpu=use_gpu,
            **kwargs,
        )

    return create_lints




def test_alpha_zero_equals_lin(lin_ts_factory):
    contexts = np.array([
        [1, 0, 0, 0, 1],
        [0, 1, 2, 3, 4],
        [2, 0, 1, 0, 2],
        [2, 1, 2, 1, 2],
        [3, 3, 3, 2, 1],
        [1, 1, 1, 1, 1],
    ])

    decisions = np.array([1, 1, 1, 0, 0, 1])
    rewards = np.array([0, 1, 1, 0, 1, 0])

    lints = lin_ts_factory(
        alpha=0.0,
        l2_lambda=1.0,
        seed=123,
    )

    lin = gobrec.mabs.lin_mabs.LinGreedy(
        epsilon=0.0,
        l2_lambda=1.0,
        seed=123,
    )

    lints.fit(contexts, decisions, rewards)
    lin.fit(contexts, decisions, rewards)

    test_contexts = np.array([
        [1, 0, 0, 0, 1],
        [0, 1, 2, 3, 4],
        [2, 0, 1, 0, 2],
    ])

    pred_lints = lints.predict(test_contexts).cpu().numpy()
    pred_lin = lin.predict(test_contexts).cpu().numpy()

    assert np.allclose(pred_lints, pred_lin, atol=1e-8)




def test_mocked_random_sampling(lin_ts_factory):
    contexts = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    decisions = np.array([
        "a", "a", "a",
        "b", "b", "b",
    ])

    rewards = np.array([
        10, 0, 1,
        1, 10, 0,
    ])

    lints = lin_ts_factory(
        alpha=1.0,
        l2_lambda=1.0,
    )

    lints.fit(contexts, decisions, rewards)

    test_contexts = np.array([
        [1, 0, 0],
        [0, 1, 0],
    ])

    # One deterministic Gaussian sample per context.
    mocked_eps = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    lints.rng = MockRNG([mocked_eps])

    pred = lints.predict(test_contexts).cpu().numpy()

    # Reproduce the exact calculation performed by LinTS.
    beta = lints.beta.cpu()
    A = lints.A.cpu()

    A_inv = torch.linalg.inv(A)
    L = torch.linalg.cholesky(A_inv)

    beta_sampled = (
        torch.einsum(
            "bd,add->bad",
            torch.tensor(mocked_eps, dtype=torch.double),
            L,
        )
        + beta
    )

    expected = torch.einsum(
        "bd,bad->ba",
        torch.tensor(test_contexts, dtype=torch.double),
        beta_sampled,
    ).numpy()

    assert np.allclose(pred, expected, atol=1e-8)




def test_alpha_scales_random_component(lin_ts_factory):
    contexts = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    decisions = np.array([
        "a", "a", "a",
        "b", "b", "b",
    ])

    rewards = np.array([
        10, 0, 1,
        1, 10, 0,
    ])

    test_contexts = np.array([
        [1, 0, 0],
        [0, 1, 0],
    ])

    mocked_eps = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    mab_alpha_1 = lin_ts_factory(
        alpha=1.0,
        l2_lambda=1.0,
    )

    mab_alpha_2 = lin_ts_factory(
        alpha=2.0,
        l2_lambda=1.0,
    )

    mab_alpha_1.fit(contexts, decisions, rewards)
    mab_alpha_2.fit(contexts, decisions, rewards)

    mab_alpha_1.rng = MockRNG([mocked_eps])
    mab_alpha_2.rng = MockRNG([mocked_eps])

    pred1 = mab_alpha_1.predict(test_contexts).cpu().numpy()
    pred2 = mab_alpha_2.predict(test_contexts).cpu().numpy()

    beta = mab_alpha_1.beta.cpu()
    A = mab_alpha_1.A.cpu()

    L = torch.linalg.cholesky(torch.linalg.inv(A))

    eps = torch.tensor(mocked_eps, dtype=torch.double)

    exploration = torch.einsum(
        "bd,add->bad",
        eps,
        L,
    )

    expected1 = torch.einsum(
        "bd,bad->ba",
        torch.tensor(test_contexts, dtype=torch.double),
        beta + exploration,
    ).numpy()

    expected2 = torch.einsum(
        "bd,bad->ba",
        torch.tensor(test_contexts, dtype=torch.double),
        beta + 2.0 * exploration,
    ).numpy()

    assert np.allclose(pred1, expected1, atol=1e-8)
    assert np.allclose(pred2, expected2, atol=1e-8)


def test_mocked_rng_changes_prediction(lin_ts_factory):
    contexts = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    decisions = np.array([
        "a", "a", "a",
        "b", "b", "b",
    ])

    rewards = np.array([
        10, 0, 1,
        1, 10, 0,
    ])

    test_contexts = np.eye(3)

    mab = lin_ts_factory(
        alpha=1.0,
        l2_lambda=1.0,
    )

    mab.fit(contexts, decisions, rewards)

    eps1 = np.ones((3, 3))
    eps2 = -np.ones((3, 3))

    mab.rng = MockRNG([eps1])
    pred1 = mab.predict(test_contexts).cpu().numpy()

    mab.rng = MockRNG([eps2])
    pred2 = mab.predict(test_contexts).cpu().numpy()

    assert not np.allclose(pred1, pred2)


def test_seed_reproducibility(lin_ts_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 2, 2, 3, 5],
        [1, 3, 1, 1, 1],
    ])

    decisions = np.array([1, 1, 2, 2, 3])
    rewards = np.array([0, 1, 0, 1, 1])

    mab1 = lin_ts_factory(
        seed=123456,
        alpha=0.5,
    )

    mab2 = lin_ts_factory(
        seed=123456,
        alpha=0.5,
    )

    mab1.fit(contexts, decisions, rewards)
    mab2.fit(contexts, decisions, rewards)

    pred1 = mab1.predict(contexts).cpu().numpy()
    pred2 = mab2.predict(contexts).cpu().numpy()

    assert np.allclose(pred1, pred2)

def test_different_seed_changes_sampling(lin_ts_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 2, 2, 3, 5],
        [1, 3, 1, 1, 1],
    ])

    decisions = np.array([1, 1, 2, 2, 3])
    rewards = np.array([0, 1, 0, 1, 1])

    mab1 = lin_ts_factory(
        seed=123,
        alpha=1.0,
    )

    mab2 = lin_ts_factory(
        seed=456,
        alpha=1.0,
    )

    mab1.fit(contexts, decisions, rewards)
    mab2.fit(contexts, decisions, rewards)

    pred1 = mab1.predict(contexts).cpu().numpy()
    pred2 = mab2.predict(contexts).cpu().numpy()

    assert not np.allclose(pred1, pred2)