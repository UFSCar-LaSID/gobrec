
import os
import sys

local_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, local_path)

import numpy as np
import pytest

from gobrec.mabs import LabelEncoder


def test_init():
    encoder = LabelEncoder()

    assert encoder.class_to_index == {}
    assert encoder.index_to_class == []
    assert encoder.classes_.size == 0


def test_fit_string_decisions():
    encoder = LabelEncoder()
    decisions = np.array(["a", "b", "c"])

    encoder.fit(decisions)

    assert encoder.class_to_index == {
        "a": 0,
        "b": 1,
        "c": 2,
    }
    assert encoder.index_to_class == ["a", "b", "c"]
    assert np.all(encoder.classes_ == decisions)


def test_fit_integer_decisions():
    encoder = LabelEncoder()
    decisions = np.array([10, 20, 30])

    encoder.fit(decisions)

    assert encoder.class_to_index == {
        10: 0,
        20: 1,
        30: 2,
    }
    assert encoder.index_to_class == [10, 20, 30]
    assert np.all(encoder.classes_ == decisions)


def test_fit_duplicate_decisions():
    encoder = LabelEncoder()
    decisions = np.array(["a", "b", "a", "c", "b"])

    encoder.fit(decisions)

    assert encoder.class_to_index == {
        "a": 0,
        "b": 1,
        "c": 2,
    }
    assert encoder.index_to_class == ["a", "b", "c"]
    assert np.all(encoder.classes_ == np.array(["a", "b", "c"]))

def test_fit_incrementally_adds_new_classes():
    encoder = LabelEncoder()

    encoder.fit(np.array(["a", "b"]))
    encoder.fit(np.array(["b", "c"]))

    assert encoder.class_to_index == {
        "a": 0,
        "b": 1,
        "c": 2,
    }
    assert encoder.index_to_class == ["a", "b", "c"]


def test_fit_incrementally_preserves_existing_indices():
    encoder = LabelEncoder()

    encoder.fit(np.array(["a", "b", "c"]))
    original_mapping = encoder.class_to_index.copy()

    encoder.fit(np.array(["c", "d", "a", "e"]))

    for item, index in original_mapping.items():
        assert encoder.class_to_index[item] == index

    assert encoder.class_to_index["d"] == 3
    assert encoder.class_to_index["e"] == 4


def test_fit_empty_decisions():
    encoder = LabelEncoder()

    encoder.fit(np.array([], dtype=str))

    assert encoder.class_to_index == {}
    assert encoder.index_to_class == []


def test_transform_string_decisions():
    encoder = LabelEncoder()
    encoder.fit(np.array(["a", "b", "c"]))

    result = encoder.transform(np.array(["c", "a", "b"]))

    assert np.all(result == np.array([2, 0, 1]))

def test_transform_integer_decisions():
    encoder = LabelEncoder()
    encoder.fit(np.array([10, 20, 30]))

    result = encoder.transform(np.array([30, 10, 20]))

    assert np.all(result == np.array([2, 0, 1]))

def test_transform_repeated_decisions():
    encoder = LabelEncoder()
    encoder.fit(np.array(["a", "b", "c"]))

    result = encoder.transform(np.array(["b", "b", "a", "b"]))

    assert np.all(result == np.array([1, 1, 0, 1]))


def test_transform_unknown_class_raises_key_error():
    encoder = LabelEncoder()
    encoder.fit(np.array(["a", "b"]))

    with pytest.raises(KeyError, match="c"):
        encoder.transform(np.array(["c"]))


def test_transform_empty_decisions():
    encoder = LabelEncoder()
    encoder.fit(np.array(["a", "b"]))

    result = encoder.transform(np.array([], dtype=str))

    assert result.shape == (0,)


def test_inverse_transform_string_decisions():
    encoder = LabelEncoder()
    encoder.fit(np.array(["a", "b", "c"]))

    result = encoder.inverse_transform(np.array([2, 0, 1]))

    assert np.all(result == np.array(["c", "a", "b"]))


def test_inverse_transform_integer_decisions():
    encoder = LabelEncoder()
    encoder.fit(np.array([10, 20, 30]))

    result = encoder.inverse_transform(np.array([2, 0, 1]))

    assert np.all(result == np.array([30, 10, 20]))


def test_inverse_transform_repeated_indices():
    encoder = LabelEncoder()
    encoder.fit(np.array(["a", "b", "c"]))

    result = encoder.inverse_transform(np.array([1, 1, 0, 1]))

    assert np.all(result == np.array(["b", "b", "a", "b"]))


def test_inverse_transform_invalid_index_raises_index_error():
    encoder = LabelEncoder()
    encoder.fit(np.array(["a", "b"]))

    with pytest.raises(IndexError):
        encoder.inverse_transform(np.array([2]))


def test_inverse_transform_empty_indices():
    encoder = LabelEncoder()
    encoder.fit(np.array(["a", "b"]))

    result = encoder.inverse_transform(np.array([], dtype=int))

    assert result.shape == (0,)


def test_classes_returns_numpy_array():
    encoder = LabelEncoder()
    encoder.fit(np.array(["a", "b", "c"]))

    classes = encoder.classes_

    assert isinstance(classes, np.ndarray)
    assert np.all(classes == np.array(["a", "b", "c"]))


def test_classes_reflects_incremental_updates():
    encoder = LabelEncoder()
    encoder.fit(np.array(["a", "b"]))

    assert np.all(encoder.classes_ == np.array(["a", "b"]))

    encoder.fit(np.array(["c"]))

    assert np.all(encoder.classes_ == np.array(["a", "b", "c"]))


def test_transform_inverse_transform_round_trip():
    encoder = LabelEncoder()
    decisions = np.array(["item_c", "item_a", "item_b", "item_c"])

    encoder.fit(decisions)

    encoded = encoder.transform(decisions)
    decoded = encoder.inverse_transform(encoded)
    
    assert np.all(decoded == decisions)


def test_incremental_fit_round_trip():
    encoder = LabelEncoder()

    encoder.fit(np.array(["a", "b"]))
    encoder.fit(np.array(["c", "d"]))

    decisions = np.array(["d", "a", "c", "b"])
    encoded = encoder.transform(decisions)
    decoded = encoder.inverse_transform(encoded)

    assert np.all(decoded == decisions)