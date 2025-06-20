from artibot.model import TradingModel
from artibot.hyperparams import TRANSFORMER_HEADS


def test_embed_dim_multiple_of_heads():
    model = TradingModel(input_size=19)
    assert model.d_model % TRANSFORMER_HEADS == 0
    assert model.d_model >= 19
