import threading
import time

import numpy as np

import artibot.feature_store as fs


def _worker(out):
    for _ in range(100):
        out.append(fs.news_sentiment(1609459200))


def test_thread_safety():
    fs.upsert_news(1609459200, 0.42)
    vals1: list[np.float32] = []
    vals2: list[np.float32] = []
    t1 = threading.Thread(target=_worker, args=(vals1,))
    t2 = threading.Thread(target=_worker, args=(vals2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert all(v == np.float32(0.42) for v in vals1 + vals2)

