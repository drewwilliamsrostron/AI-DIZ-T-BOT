import threading
import artibot.feature_store as fs


def test_duckdb_multithread():
    fs.upsert_news(1609459200, 0.1)

    def worker():
        for _ in range(1000):
            val = fs.news_sentiment(1609459200)
            assert isinstance(val, float)

    ths = [threading.Thread(target=worker) for _ in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
