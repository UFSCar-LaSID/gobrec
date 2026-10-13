
import os
import sys
from unittest.mock import MagicMock, patch

local_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, local_path)

import numpy as np
import pytest
import torch

from gobrec.mabs.mab_algo import MABAlgo

class DummyMABAlgo(MABAlgo):
    def fit(self, contexts, decisions, rewards):
        pass

    def predict(self, contexts):
        pass

    def reset(self):
        super().reset()



def test_mab_algo_is_abstract():
    with pytest.raises(TypeError):
        MABAlgo()

def test_init():
    mab = DummyMABAlgo(seed=42)

    assert mab.seed == 42
    assert mab.rng is not None
    assert mab.label_encoder is None
    assert mab.num_arms is None
    assert mab.num_features is None


def test_update_label_encoder_initializes_and_fits_encoder():
    decisions = np.array(["a", "b", "c", "a"])
    num_features = 3

    mock_encoder = MagicMock()
    mock_encoder.classes_ = np.array(["a", "b", "c"])

    with patch(
        "gobrec.mabs.mab_algo.LabelEncoder",
        return_value=mock_encoder,
    ) as encoder_class:
        mab = DummyMABAlgo(seed=42)

        mab._update_label_encoder(decisions, num_features)

        encoder_class.assert_called_once_with()
        mock_encoder.fit.assert_called_once_with(decisions)

    assert mab.label_encoder is mock_encoder
    assert mab.num_arms == 3
    assert mab.num_features == num_features


def test_update_label_encoder_reuses_existing_encoder():
    decisions = np.array(["a", "b"])
    num_features = 2

    mock_encoder = MagicMock()
    mock_encoder.classes_ = np.array(["a", "b"])

    mab = DummyMABAlgo(seed=42)
    mab.label_encoder = mock_encoder

    with patch("gobrec.mabs.mab_algo.LabelEncoder") as encoder_class:
        mab._update_label_encoder(decisions, num_features)

    encoder_class.assert_not_called()
    mock_encoder.fit.assert_called_once_with(decisions)

    assert mab.label_encoder is mock_encoder
    assert mab.num_arms == 2
    assert mab.num_features == 2


def test_update_label_encoder_rejects_changed_feature_count():
    decisions = np.array(["a", "b"])

    mock_encoder = MagicMock()
    mock_encoder.classes_ = np.array(["a", "b"])

    mab = DummyMABAlgo(seed=42)
    mab.label_encoder = mock_encoder
    mab.num_features = 3

    with pytest.raises(
        AssertionError,
        match="Number of features has changed!",
    ):
        mab._update_label_encoder(decisions, num_features=4)


def test_reset_clears_shared_attributes():
    mab = DummyMABAlgo(seed=42)
    mock_encoder = MagicMock()
    mab.label_encoder = mock_encoder
    mab.num_arms = 2
    mab.num_features = 3
    mab.rng.standard_normal(size=10)

    mab.reset()

    assert mab.label_encoder is None
    assert mab.num_arms is None
    assert mab.num_features is None
    assert mab.seed == 42
    assert mab.rng is not None


def test_reset_reinitializes_rng():
    mab = DummyMABAlgo(seed=42)

    expected_values = mab.rng.standard_normal(size=5)

    mab.reset()

    actual_values = mab.rng.standard_normal(size=5)

    assert np.all(actual_values == expected_values)