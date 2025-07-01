import pytest
from artibot.utils import (
    validate_feature_dimension,
    validate_features,
    feature_mask_for,
    DimensionError,
)
from artibot.hyperparams import IndicatorHyperparams
import numpy as np
import logging


def test_validate_feature_dimension_no_change():
    arr = np.ones((2, 10), dtype=float)
    logger = logging.getLogger("test")
    fixed = validate_feature_dimension(arr.copy(), 16, logger)
    assert fixed.shape == arr.shape
    assert np.allclose(fixed, arr)


def test_validate_feature_dimension_no_trim():
    arr = np.ones((3, 18), dtype=float)
    logger = logging.getLogger("test")
    fixed = validate_feature_dimension(arr.copy(), 16, logger)
    assert fixed.shape == arr.shape


def test_validate_features_respects_mask():
    arr = np.random.randn(4, 16).astype(float)
    arr[:, 5] = 0.0
    hp = IndicatorHyperparams(use_sma=False)
    mask = feature_mask_for(hp)
    validate_features(arr, enabled_mask=mask)


def test_validate_features_zero_enabled():
    arr = np.ones((4, 16), dtype=float)
    arr[:, 6] = 0.0
    hp = IndicatorHyperparams()
    mask = feature_mask_for(hp)
    with pytest.raises(DimensionError):
        validate_features(arr, enabled_mask=mask)


def test_zero_variance_ignored_on_disabled_cols():
    feats = np.random.randn(100, 16).astype(np.float32)
    mask = np.ones(16, dtype=np.uint8)
    mask[8:] = 0
    feats[:, 8:] = 0.0
    validate_features(feats, enabled_mask=mask)


def test_validate_features_numpy2_dummy_mask():
    arr = np.random.randn(10, 16).astype(float)
    mask = np.ones(16, dtype=bool)
    validate_features(arr, enabled_mask=mask)


def test_zero_variance_partial_columns():
    arr = np.ones((5, 2), dtype=float)
    arr[:, 1] = np.arange(5)
    validate_features(arr, enabled_mask=np.array([True, True]))


def test_zero_variance_all_columns():
    arr = np.ones((5, 2), dtype=float)
    with pytest.raises(DimensionError):
        validate_features(arr, enabled_mask=np.array([True, True]))
