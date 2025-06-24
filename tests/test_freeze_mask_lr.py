def test_feature_pad_and_mask():
    from artibot.feature_store import freeze_feature_dim, get_frozen_dim

    freeze_feature_dim(17)
    assert get_frozen_dim() >= 17
    freeze_feature_dim(24)
    assert get_frozen_dim() == 24


def test_lr_clamp():
    import artibot.hyperparams as hp

    assert hp.mutate_lr(1e-4, 0.5) == 5e-4
    assert hp.mutate_lr(1e-4, -0.9) == 1e-5
