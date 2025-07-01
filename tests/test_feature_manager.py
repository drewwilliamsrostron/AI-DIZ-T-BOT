import numpy as np
from artibot.feature_manager import enforce_feature_dim


def test_enforce_feature_dim_padding():
    arr = np.ones((3, 10), dtype=float)
    fixed = enforce_feature_dim(arr.copy(), 16)
    assert fixed.shape == (3, 16)
    assert np.allclose(fixed[:, :10], arr)
    assert np.allclose(fixed[:, 10:], 0.0)


def test_enforce_feature_dim_no_change_equal():
    arr = np.random.rand(5, 16).astype(float)
    fixed = enforce_feature_dim(arr.copy(), 16)
    assert fixed.shape == (5, 16)
    assert np.allclose(fixed, arr)


def test_enforce_feature_dim_no_change_greater():
    arr = np.random.rand(4, 20).astype(float)
    fixed = enforce_feature_dim(arr.copy(), 16)
    assert fixed.shape == (4, 20)
    assert np.allclose(fixed, arr)
