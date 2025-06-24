from artibot.model import TradingModel


def test_embed_dim_multiple_of_heads():
    model = TradingModel(input_size=16)
    assert model.d_model == 16
