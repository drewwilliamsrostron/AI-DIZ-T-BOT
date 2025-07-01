from artibot.utils import validate_feature_dimension
import numpy as np
import logging


def test_validate_feature_dimension_pad():
    arr = np.ones((2, 10), dtype=float)
    logger = logging.getLogger("test")
    fixed = validate_feature_dimension(arr, 16, logger)
    assert fixed.shape == (2, 16)
    assert np.allclose(fixed[:, :10], 1)
    assert np.allclose(fixed[:, 10:], 0)


def test_validate_feature_dimension_trim():
    arr = np.ones((3, 18), dtype=float)
    logger = logging.getLogger("test")
    fixed = validate_feature_dimension(arr, 16, logger)
    assert fixed.shape == (3, 16)
