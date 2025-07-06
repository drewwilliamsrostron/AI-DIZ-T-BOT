from artibot.core.device import get_device


def test_get_device():
    assert get_device().type in {"cuda", "cpu"}
