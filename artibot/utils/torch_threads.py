import os
import threading
import logging
import torch

_once = threading.Lock()


def set_threads(n: int) -> None:
    """Set Torch inter-op threads exactly once."""
    n = max(1, int(n))
    with _once:
        torch.set_num_interop_threads(n)
        if torch.get_num_interop_threads() != n:
            logging.warning("Could not set inter-op threads to %d", n)

