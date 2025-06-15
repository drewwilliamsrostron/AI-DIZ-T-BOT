import artibot.feature_store as fs
import time
import numpy as np


def test_feature_roundtrip():
    ts = int(time.time()) // 3600 * 3600
    fs.upsert_news(ts, 0.3)
    fs.upsert_macro(ts, -1.2)
    fs.upsert_rvol(ts, 0.45)
    assert np.isclose(fs.news_sentiment(ts), 0.3)
    assert np.isclose(fs.macro_surprise(ts), -1.2)
    assert np.isclose(fs.realised_vol(ts), 0.45)


def test_live_ingest_sim():
    """Smoke-test that the worker inserts non-zero rows."""

    import artibot.feature_ingest as fi

    fi.upd_sentiment()
    fi.upd_macro()
    fi.upd_rvol()
    ts = int(time.time()) // 3600 * 3600
    assert fs.news_sentiment(ts) != 0.0 or fs.macro_surprise(ts) != 0.0
