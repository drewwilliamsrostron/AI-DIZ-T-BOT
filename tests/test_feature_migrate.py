import subprocess
import sys
import time
import duckdb
import artibot.feature_store as fs


def test_auto_migrate(tmp_path, monkeypatch):
    db = tmp_path / "old.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE news_sentiment (ts TIMESTAMP PRIMARY KEY)")
    con.execute("CREATE TABLE macro_surprise (ts TIMESTAMP PRIMARY KEY)")
    con.execute("CREATE TABLE realised_vol (ts TIMESTAMP PRIMARY KEY)")

    old_con = fs._con
    old_con.close()
    monkeypatch.setattr(fs, "_con", con, raising=False)
    monkeypatch.setattr(fs, "_DB_PATH", str(db), raising=False)

    ts = int(time.time())
    # should not raise even though columns are missing
    assert fs.news_sentiment(ts) == 0.0
    assert fs.macro_surprise(ts) == 0.0
    assert fs.realised_vol(ts) == 0.0

    cols = [r[1] for r in con.execute("PRAGMA table_info('news_sentiment')").fetchall()]
    assert "score" in cols
    cols = [r[1] for r in con.execute("PRAGMA table_info('macro_surprise')").fetchall()]
    assert "surprise_z" in cols
    cols = [r[1] for r in con.execute("PRAGMA table_info('realised_vol')").fetchall()]
    assert "rvol" in cols

    con.close()


def test_repair_cli():
    fs._con.close()
    result = subprocess.run(
        [sys.executable, "-m", "artibot.feature_store", "--repair"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "schema verified" in result.stdout
