import torch
from artibot.model import PositionalEncoding


def test_positional_encoding_adapts():
    pe = PositionalEncoding()
    x1 = torch.zeros(1, 4, 13)
    out1 = pe(x1)
    assert out1.shape == x1.shape

    x2 = torch.zeros(1, 4, 16)
    out2 = pe(x2)
    assert out2.shape == x2.shape

    x3 = torch.zeros(1, 2, 13)
    out3 = pe(x3)
    assert out3.shape == x3.shape
