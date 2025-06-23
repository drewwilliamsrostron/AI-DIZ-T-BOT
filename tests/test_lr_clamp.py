import artibot.hyperparams as hp


def test_mutate_lr_bounds():
    start = 1e-4
    big_delta = start * 0.5
    assert hp.mutate_lr(start, big_delta) <= hp.LR_MAX
    assert hp.mutate_lr(start, -big_delta) >= hp.LR_MIN
    up = hp.mutate_lr(start, big_delta)
    assert up - start <= start * hp.LR_FN_MAX_DELTA + 1e-12
    down = hp.mutate_lr(start, -big_delta)
    assert start - down <= start * hp.LR_FN_MAX_DELTA + 1e-12
    assert hp.mutate_lr(hp.LR_MIN, -1.0) == hp.LR_MIN
