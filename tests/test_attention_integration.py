import torch
import math
import logging
from artibot import model, utils

logging.getLogger().setLevel(logging.ERROR)


def test_model_attention_entropy():
    m = model.TradingModel(input_size=16)
    x = torch.randn(2, 8, 16)
    _ = m(x)
    attn_block = m.transformer_encoder.layers[0].self_attn
    _, attn_weights = attn_block(
        x,
        x,
        x,
        need_weights=True,
        average_attn_weights=False,
    )
    p = attn_weights.mean(dim=(0, 1))
    ent = utils.attention_entropy(p)
    assert 0.5 < ent < math.log(p.size(-1))
    assert p.max().item() <= 1.0
