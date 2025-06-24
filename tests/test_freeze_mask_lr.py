def test_feature_pad_and_mask():
    from artibot.feature_store import freeze_feature_dim, get_frozen_dim

    assert freeze_feature_dim(16) == 16
    assert get_frozen_dim() >= 16


def test_lr_clamp():
    import artibot.hyperparams as hp

    assert hp.mutate_lr(1e-4, 0.5) == 5e-4
    assert hp.mutate_lr(1e-4, -0.9) == 1e-5
