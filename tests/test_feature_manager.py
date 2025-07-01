import numpy as np
from artibot.feature_manager import enforce_feature_dim
from artibot.hyperparams import IndicatorHyperparams
from artibot.utils import feature_mask_for, zero_disabled
from artibot.constants import FEATURE_DIMENSION


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


def test_indicator_toggle_zeroed_columns(monkeypatch):
    """Disabled indicators should produce zeroed feature columns."""
    arr = np.ones((3, FEATURE_DIMENSION), dtype=float)
    hp = IndicatorHyperparams(use_sma=False, use_atr=False, use_vortex=False)
    mask = feature_mask_for(hp)
    import torch

    monkeypatch.setattr(torch, "is_tensor", lambda x: False)
    result = zero_disabled(arr.copy(), mask)

    assert result.shape == arr.shape
    disabled_idx = np.where(~mask)[0]
    assert disabled_idx.size > 0
    assert np.all(result[:, disabled_idx] == 0.0)
    assert np.all(result[:, mask] == 1.0)
