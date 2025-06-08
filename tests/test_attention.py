import torch
from artibot.utils import attention_entropy


def test_attention_entropy_range():
    sample = torch.randn(4, 8)
    val = attention_entropy(sample)
    assert 0.0 < val < 5.0
