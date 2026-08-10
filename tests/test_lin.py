
import gobrec
import numpy as np


def test_lin_prediction_one_arm():
    contexts = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2]])
    decisions = np.array([0, 0, 0])
    rewards = np.array([3, 3, 1])

    lin = gobrec.mabs.lin_mabs.Lin(l2_lambda=1.0, use_gpu=False)

    lin.fit(contexts, decisions, rewards)
    pred = lin.predict(np.array([[0, 1, 2, 3, 5]])).cpu().numpy()

    assert np.allclose(pred, np.array([2.8167701863354]), atol=1e-8)

def test_predict_multiple_arms():
    decisions = np.array(["a", "a", "a",
                          "b", "b", "b",
                          "c", "c", "c"])
    contexts = np.vstack([np.eye(3), np.eye(3), np.eye(3)])
    rewards = np.array([
        10, 0, 1,
        1, 10, 0,
        0, 1, 10
    ])

    lin = gobrec.mabs.lin_mabs.Lin(l2_lambda=1.0)

    lin.fit(contexts, decisions, rewards)

    pred = lin.predict(np.eye(3)).cpu().numpy()

    expected = np.array([
        [5.0, 0.5, 0.0],
        [0.0, 5.0, 0.5],
        [0.5, 0.0, 5.0]
    ])

    assert np.allclose(pred, expected)

def test_fit():
    contexts = np.array([
        [1, 0, 2, 1, 1],
        [3, 1, 2, 3, 4],
        [2, -1, 1, 0, 2]
    ])

    rewards = np.array([3, 3, 1])
    decisions = np.array([1, 1, 1])

    lin = gobrec.mabs.lin_mabs.Lin(l2_lambda=1.0)

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

    lin = gobrec.mabs.lin_mabs.Lin(l2_lambda=1.0)

    lin.fit(contexts, decisions, rewards)

    expected = np.array([
        0.09927202,
        -0.17141953,
        0.09091367,
        -0.03705452,
        0.59027579,
    ])

    assert np.allclose(lin.beta[0].cpu(), expected)

def test_incremental_fit():
    # First batch
    context1 = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2],])
    rewards1 = np.array([3, 2, 1])
    decisions1 = np.array([1, 1, 1])

    lin = gobrec.mabs.lin_mabs.Lin(l2_lambda=1.0, use_gpu=False)

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

def test_incremental_vs_full_fit():
    # Batch fit - all data at once
    contexts_batch = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2],
                               [2, 1, 2, 1, 2], [3, 3, 3, 2, 1], [1, 1, 1, 1, 1]])
    rewards_batch = np.array([0, 1, 1, 0, 1, 0])
    decisions_batch = np.array([1, 1, 1, 0, 0, 1])

    lin_batch = gobrec.mabs.lin_mabs.Lin(use_gpu=False)

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

    lin_inc = gobrec.mabs.lin_mabs.Lin(use_gpu=False)

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

def test_incremental_different_order():
    contexts1 = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2]])
    rewards1 = np.array([0, 1, 1])
    decisions1 = np.array([1, 1, 1])

    contexts2 = np.array([[2, 1, 2, 1, 2],[3, 3, 3, 2, 1],[1, 1, 1, 1, 1]])
    rewards2 = np.array([0, 1, 0])
    decisions2 = np.array([0, 0, 1])

    lin12 = gobrec.mabs.lin_mabs.Lin(
        l2_lambda=1.0,
        use_gpu=False,
    )
    lin12.fit(
        contexts1,
        decisions1,
        rewards1,
    )
    lin12.fit(
        contexts2,
        decisions2,
        rewards2,
    )

    lin21 = gobrec.mabs.lin_mabs.Lin(
        l2_lambda=1.0,
        use_gpu=False,
    )
    lin21.fit(
        contexts2,
        decisions2,
        rewards2,
    )
    lin21.fit(
        contexts1,
        decisions1,
        rewards1,
    )

    # The label encoding depends on the order in which arms are first seen.
    # Therefore, compare the models using the original arm IDs rather than
    # directly comparing the encoded rows.

    for arm in [0,1]:
        idx12 = lin12.label_encoder.transform(np.array([arm]))[0]
        idx21 = lin21.label_encoder.transform(np.array([arm]))[0]

        assert np.allclose(
            lin12.beta[idx12].cpu().numpy(),
            lin21.beta[idx21].cpu().numpy(),
            atol=1e-8,
        )

        assert np.allclose(
            lin12.A[idx12].cpu().numpy(),
            lin21.A[idx21].cpu().numpy(),
            atol=1e-8,
        )

        assert np.allclose(
            lin12.Xty[idx12].cpu().numpy(),
            lin21.Xty[idx21].cpu().numpy(),
            atol=1e-8,
        )
    
def test_batch_vs_3_incremental_fit():
    contexts = np.array([
        [1, 0, 0, 0, 1],[0, 1, 2, 3, 4],[2, 0, 1, 0, 2],
        [2, 1, 2, 1, 2],[3, 3, 3, 2, 1],[1, 1, 1, 1, 1],
        [1, 2, 0, 1, 3],[2, 2, 1, 0, 1],[3, 0, 2, 1, 2],
    ])
    rewards = np.array([
        0, 1, 1,
        0, 1, 0,
        1, 0, 1,
    ])
    decisions = np.array([
        1, 1, 1,
        0, 0, 1,
        0, 1, 0,
    ])

    lin_batch = gobrec.mabs.lin_mabs.Lin(
        l2_lambda=1.0,
        use_gpu=False,
    )

    lin_batch.fit(
        contexts,
        decisions,
        rewards,
    )



    lin_incremental = gobrec.mabs.lin_mabs.Lin(
        l2_lambda=1.0,
        use_gpu=False,
    )

    lin_incremental.fit(
        contexts[:3],
        decisions[:3],
        rewards[:3],
    )
    lin_incremental.fit(
        contexts[3:6],
        decisions[3:6],
        rewards[3:6],
    )
    lin_incremental.fit(
        contexts[6:],
        decisions[6:],
        rewards[6:],
    )


    assert np.array_equal(
        lin_batch.label_encoder.classes_,
        lin_incremental.label_encoder.classes_,
    )


    assert np.allclose(
        lin_batch.A.cpu().numpy(),
        lin_incremental.A.cpu().numpy(),
        atol=1e-8,
    )

    assert np.allclose(
        lin_batch.Xty.cpu().numpy(),
        lin_incremental.Xty.cpu().numpy(),
        atol=1e-8,
    )

    assert np.allclose(
        lin_batch.beta.cpu().numpy(),
        lin_incremental.beta.cpu().numpy(),
        atol=1e-8,
    )

    test_contexts = np.array([[1, 2, 3, 4, 5],[5, 4, 3, 2, 1],[0, 1, 0, 1, 0]])

    pred_batch = lin_batch.predict(test_contexts)
    pred_incremental = lin_incremental.predict(test_contexts)

    assert np.allclose(
        pred_batch.cpu().numpy(),
        pred_incremental.cpu().numpy(),
        atol=1e-8,
    )

def test_l2_low():
    context = np.array([[1, 1, 0, 0, 1],[0, 1, 2, 9, 4],[2, 3, 1, 0, 2]])
    rewards = np.array([3, 2, 1])
    decisions = np.array([1, 1, 1])

    lin = gobrec.mabs.lin_mabs.Lin(
        l2_lambda=0.1,
        use_gpu=False,
    )

    lin.fit(context, decisions, rewards)

    assert lin.num_features == 5
    assert lin.num_arms == 1

    expected_beta = np.array([
        1.59499705,
        -0.91856183,
        -2.49775977,
        0.14219195,
        1.65819347,
    ])

    assert np.allclose(
        lin.beta[0].cpu().numpy(),
        expected_beta,
        atol=1e-8,
    )

def test_l2_high():
    context = np.array([[1, 1, 0, 0, 1],[0, 1, 2, 9, 4],[2, 3, 1, 0, 2]])
    rewards = np.array([3, 2, 1])
    decisions = np.array([1, 1, 1])

    lin = gobrec.mabs.lin_mabs.Lin(
        l2_lambda=10,
        use_gpu=False,
    )

    lin.fit(context, decisions, rewards)

    assert lin.num_features == 5
    assert lin.num_arms == 1

    expected_beta = np.array([
        0.18310155,
        0.16372811,
        -0.00889076,
        0.09434416,
        0.22503229,
    ])

    assert np.allclose(
        lin.beta[0].cpu().numpy(),
        expected_beta,
        atol=1e-8,
    )

def test_l2_0():
    context = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2], [-1, 4, 2, 0, 1],
                        [2, 2, 2, 2, 2], [3, 2, 1, 2, 3], [0, 0, 0, 0, 0], [2, 1, 1, 1, 2],
                        [3, 2, 3, 2, 3], [8, 2, 3, 1, 0], [1, 2, -9, -7, 1], [0, 1, 1, 1, 1]])
    rewards = np.array([
        3, 3, 1, 0, 
        -1, 2, 1, 2, 
        1, 1, 0, 3,
    ])
    decisions = np.array([
        1, 1, 1, 1, 
        1, 1, 1, 1, 
        1, 1, 1, 1,
    ])

    lin = gobrec.mabs.lin_mabs.Lin(
        l2_lambda=0,
        use_gpu=False,
    )

    lin.fit(context, decisions, rewards)

    assert lin.num_features == 5
    assert lin.num_arms == 1

    expected_beta = np.array([
        0.09224215,
        -0.20569848,
        0.13434242,
        -0.10000450,
        0.63726682,
    ])

    assert np.allclose(
        lin.beta[0].cpu().numpy(),
        expected_beta,
        atol=1e-8,
    )