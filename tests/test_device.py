from artibot.core.device import get_device, enable_flash_sdp


def test_get_device():
    assert get_device().type in {"cuda", "cpu"}


def test_enable_flash_sdp_runs():
    assert isinstance(enable_flash_sdp(), bool)
