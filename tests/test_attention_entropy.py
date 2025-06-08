import torch
from artibot.utils import attention_entropy


def test_attention_entropy_zero():
    assert attention_entropy(torch.zeros(2, 4, 4)) == 0.0


def test_attention_entropy_random():
    val = attention_entropy(torch.randn(2, 4, 4))
    assert val > 0.5
