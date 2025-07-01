from artibot.utils import validate_feature_dimension
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
