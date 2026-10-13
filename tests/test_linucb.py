

import os
import sys

local_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, local_path)
import gobrec

import numpy as np
import datetime
import torch
import pytest




@pytest.fixture(params=[False, True], ids=["cpu", "gpu"])
def lin_ucb_factory(request):
    use_gpu = request.param

    if use_gpu and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    def create_linucb(**kwargs):
        return gobrec.mabs.lin_mabs.LinUCB(
            use_gpu=use_gpu,
            **kwargs,
        )

    return create_linucb

def get_best_arms(linucb, contexts):
    pred = linucb.predict(contexts).cpu().numpy()
    best_arms = linucb.label_encoder.inverse_transform(pred.argmax(axis=1))
    return pred, best_arms




def test_alpha0(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], 
        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]
    ])
    decisions=np.array([
        1, 1, 1, 2, 2,
        2, 3, 3, 3, 1
    ])
    rewards=np.array([
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 1
    ])

    linucb = lin_ucb_factory(alpha=0.0)

    linucb.fit(contexts, decisions, rewards)
    
    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
    pred = linucb.predict(test_contexts).cpu().numpy()
    expected = np.array([
        [-0.018378378378378413, 0.0, 0.9966292134831471],
        [0.14054054054054055, 0.0, 0.43258426966292074]
    ])

    assert pred.shape == (2, 3)
    assert np.allclose(pred, expected)


def test_alpha1(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5],[1, 1, 1, 1, 1],[0, 0, 1, 0, 0],[0, 2, 2, 3, 5],[1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0],[0, 1, 4, 3, 5],[0, 1, 2, 4, 5],[1, 2, 1, 1, 3],[0, 2, 1, 0, 0],
    ])

    decisions = np.array([
        1, 1, 1, 2, 2,
        3, 3, 3, 3, 3,
    ])

    rewards = np.array([
        0, 0, 1, 0, 0,
        0, 0, 1, 1, 1,
    ])

    linucb = lin_ucb_factory(alpha=1.0)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1],
    ])

    pred = linucb.predict(test_contexts).cpu().numpy()

    expected = np.array([
        [0.9790312458954391, 1.1617706239438832, 1.4247056229871702],
        [0.8896475809353053, 0.923364043088837, 1.457085577251709],
    ])

    assert pred.shape == (2, 3)
    assert np.allclose(pred, expected)

def test_linucb_t1(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5],[1, 1, 1, 1, 1],[0, 0, 1, 0, 0],[0, 2, 2, 3, 5],[1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0],[0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],[0, 2, 1, 0, 0]
    ])

    decisions = np.array([
        1, 1, 1, 3, 2,
        2, 3, 1, 3, 1
    ])

    rewards = np.array([
        0, 1, 1, 0, 1,
        0, 1, 1, 1, 1
    ])

    linucb = lin_ucb_factory(alpha=0.24)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1]
    ])
    pred, best_arms = get_best_arms(linucb, test_contexts)

    assert pred.shape == (2, 3)
    assert (best_arms == np.array([2, 1])).all()

def test_linucb_t2(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]
    ])

    decisions = np.array([
        1, 1, 1, 3, 2,
        2, 3, 1, 3, 1
    ])

    rewards = np.array([
        0, 1, 1, 0, 1,
        0, 1, 1, 1, 1
    ])

    linucb = lin_ucb_factory(alpha=1.5)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1]
    ])

    pred, best_arms = get_best_arms(linucb, test_contexts)

    assert pred.shape == (2, 3)
    assert (best_arms == np.array([2, 2])).all()

def test_linucb_t3(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0],
        [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0], [0, 2, 2, 3, 5],
        [1, 3, 1, 1, 1]
    ])

    decisions = np.array([
        1, 1, 4, 4, 2,
        2, 1, 1, 4, 2,
        1, 4, 1, 2, 4,
        1
    ])

    rewards = np.array([
        7, 9, 10, 20, 2,
        5, 8, 15, 17, 11,
        0, 5, 2, 9, 3,
        1
    ])

    linucb = lin_ucb_factory(alpha=1.25)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1]
    ])

    pred, best_arms = get_best_arms(linucb, test_contexts)

    assert pred.shape == (2, 3)
    assert (best_arms == np.array([4, 4])).all()

def test_linucb_t4(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0],
        [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0], [0, 2, 2, 3, 5],
        [1, 3, 1, 1, 1]
    ])

    decisions = np.array([
        1, 1, 4, 4, 2,
        2, 1, 1, 4, 2,
        1, 4, 1, 2, 4,
        1
    ])

    rewards = np.array([
        7, 9, 10, 20, 2,
        5, 8, 15, 17, 11,
        0, 5, 2, 9, 3,
        1
    ])

    linucb = lin_ucb_factory(alpha=2.0)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1]
    ])

    pred, best_arms = get_best_arms(linucb, test_contexts)

    assert pred.shape == (2, 3)
    assert (best_arms == np.array([4, 4])).all()

def test_linucb_t5(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]
    ])

    decisions = np.array([
        "one", "one", "one", "three", "two",
        "two", "three", "one", "three", "two"
    ])

    rewards = np.array([
        1, 0, 1, 0, 1,
        0, 1, 1, 1, 0
    ])

    linucb = lin_ucb_factory(alpha=1.0)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1]
    ])

    pred, best_arms = get_best_arms(linucb, test_contexts)

    assert pred.shape == (2, 3)
    assert (best_arms == np.array(["two", "two"])).all()

def test_linucb_t6(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0],
        [0, 1, 4, 3, 5]
    ])

    decisions = np.array([
        "one", "one", "one", "three", "two",
        "two", "three", "one", "three", "two",
        "one"
    ])

    rewards = np.array([
        2, 7, 7, 9, 1,
        3, 1, 2, 6, 4,
        1
    ])

    linucb = lin_ucb_factory(alpha=1.25)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1]
    ])

    pred, best_arms = get_best_arms(linucb, test_contexts)

    assert pred.shape == (2, 3)
    assert (best_arms == np.array(["three", "one"])).all()

def test_linucb_t8(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]
    ])

    decisions = np.array([
        "a", "b", "c", "a", "b",
        "c", "a", "b", "c", "a"
    ])

    rewards = np.array([
        -1.25, 0.7, 12, 10, 12,
        9.2, -1, -10, 4, 0
    ])

    linucb = lin_ucb_factory(alpha=0.5)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1]
    ])

    pred, best_arms = get_best_arms(linucb, test_contexts)

    assert pred.shape == (2, 3)
    assert (best_arms == np.array(["c", "c"])).all()

def test_linucb_t9(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]
    ])

    a = datetime.datetime(2018, 1, 1)
    b = datetime.datetime(2017, 7, 31)
    c = datetime.datetime(2018, 9, 15)

    decisions = np.array([a, b, c, a, b, c, a, b, c, a], dtype=object)

    rewards = np.array([
        1.25, 0.7, 12, 10, 1.43,
        0.2, -1, -10, 4, 0
    ])

    linucb = lin_ucb_factory(alpha=0.25)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1]
    ])

    pred, best_arms = get_best_arms(linucb, test_contexts)

    assert pred.shape == (2, 3)
    assert (best_arms == np.array([c, c], dtype=object)).all()

def test_linucb_t10(lin_ucb_factory):
    contexts = np.array([
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1],
        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0],
        [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]
    ])

    a = datetime.datetime(2018, 1, 1)
    b = datetime.datetime(2017, 7, 31)
    c = datetime.datetime(2018, 9, 15)

    decisions = np.array([
        a, b, c, a, b,
        c, a, b, c, a,
        b, b, a
    ], dtype=object)

    rewards = np.array([
        7, 12, 1, -10, 5,
        1, 2, 9, 3, 3,
        6, 7, 1
    ])

    linucb = lin_ucb_factory(alpha=1.0)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    test_contexts = np.array([
        [0, 1, 2, 3, 5],
        [1, 1, 1, 1, 1]
    ])

    pred, best_arms = get_best_arms(linucb, test_contexts)

    assert pred.shape == (2, 3)
    assert (best_arms == np.array([b, b], dtype=object)).all()

def test_fit(lin_ucb_factory):
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
        0, 0, 1, 1, 1
    ])

    linucb = lin_ucb_factory(alpha=1.0)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    assert np.allclose(-0.0825688, linucb.beta[0][0], atol=0.00001)
    assert np.allclose(0.023696, linucb.beta[2][0], atol=0.00001)



def test_partial_fit(lin_ucb_factory):
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
        0, 0, 1, 1, 1
    ])

    linucb = lin_ucb_factory(alpha=1.0)

    linucb.fit(contexts, decisions, rewards)

    assert linucb.num_features == 5
    assert linucb.num_arms == 3

    assert np.isclose(-0.0825688, linucb.beta[0][0], atol=0.00001)
    assert np.isclose(0.023696, linucb.beta[2][0], atol=0.00001)

    contexts2 = np.array([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]])
    decisions2 = np.array([1, 3, 4])
    rewards2 = np.array([0, 1, 1])

    linucb.fit(contexts2, decisions2, rewards2)

    assert np.isclose(-0.05142857, linucb.beta[0][0], atol=0.00001)
    assert np.isclose(0.22099152, linucb.beta[2][0], atol=0.00001)
