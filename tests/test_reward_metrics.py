import torch
import pytest
from artibot.utils.reward_utils import sortino_ratio, omega_ratio, calmar_ratio


def test_sortino_ratio_basic():
    r = torch.tensor([0.1, -0.05, 0.2])
    val = sortino_ratio(r)
    assert val > 0


def test_omega_ratio_basic():
    r = torch.tensor([0.1, -0.1, 0.2])
    val = omega_ratio(r)
    assert val > 1


def test_calmar_ratio_basic():
    val = calmar_ratio(0.2, -0.1, 365)
    assert val == pytest.approx(2.0, rel=1e-3)
