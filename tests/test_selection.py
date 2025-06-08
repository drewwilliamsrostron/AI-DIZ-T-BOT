from artibot.ensemble import choose_best


def test_choose_best():
    assert choose_best([-0.1, 0.2, 0.05]) == 0.2
