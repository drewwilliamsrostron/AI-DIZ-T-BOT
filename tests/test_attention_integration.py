import math
import torch
from artibot import model, utils


def test_attention_integration():
    torch.manual_seed(0)
    m = model.TradingTransformer()
    x = torch.randn(2, 8, m.input_dim)
    _ = m(x)
    block = m.transformer_encoder.layers[0].self_attn
    _, attn_weights = block(x, x, x, need_weights=True)
    p = attn_weights.mean(dim=(0, 1))
    ent = utils.attention_entropy(p)
    assert 1.0 <= ent <= math.log(p.size(-1)) + 0.1
    assert p.max().item() <= 1.0
