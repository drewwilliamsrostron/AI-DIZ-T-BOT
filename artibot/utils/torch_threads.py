import threading
import logging
import torch

_lock = threading.Lock()
_already_set = False


def set_threads(n: int) -> None:
    """Set Torch inter-op threads exactly once."""
    global _already_set
    n = max(1, int(n))
    with _lock:
        if _already_set:
            logging.info("Torch inter-op threads already set; ignoring")
            return
        torch.set_num_interop_threads(n)
        _already_set = True
        if torch.get_num_interop_threads() != n:
            logging.warning("Could not set inter-op threads to %d", n)
