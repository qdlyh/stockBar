from datetime import date

import pandas as pd

from stockbar.datafeed.instruments import Board
from stockbar.datafeed.source import StockInfo
from stockbar.datafeed.store import LocalStore


def _bars(dates, closes):
    return pd.DataFrame({
        "date": dates,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [100] * len(dates), "amount": [1000.0] * len(dates),
    })


def test_save_and_load_bars(tmp_path):
    store = LocalStore(tmp_path)
    df = _bars([date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5])
    store.save_bars("600000", df)
    out = store.load_bars("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(out["close"]) == [10.0, 10.5]
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 3)]


def test_append_bars_dedupes_and_sorts(tmp_path):
    store = LocalStore(tmp_path)
    store.save_bars("600000", _bars([date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5]))
    # 追加：与 1/3 重叠 + 新增 1/4，重叠以新数据为准
    store.append_bars("600000", _bars([date(2024, 1, 3), date(2024, 1, 4)], [99.0, 11.0]))
    out = store.load_bars("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    assert list(out["close"]) == [10.0, 99.0, 11.0]


def test_last_bar_date(tmp_path):
    store = LocalStore(tmp_path)
    assert store.last_bar_date("600000") is None
    store.save_bars("600000", _bars([date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5]))
    assert store.last_bar_date("600000") == date(2024, 1, 3)


def test_load_bars_missing_returns_empty(tmp_path):
    store = LocalStore(tmp_path)
    out = store.load_bars("000001", date(2024, 1, 1), date(2024, 1, 31))
    assert out.empty


def test_save_and_load_stocks(tmp_path):
    store = LocalStore(tmp_path)
    stocks = [
        StockInfo("600000", "浦发银行", Board.MAIN, date(1999, 11, 10), False),
        StockInfo("688981", "中芯国际", Board.STAR, date(2020, 7, 16), False),
    ]
    store.save_stocks(stocks)
    loaded = store.load_stocks()
    assert {s.code for s in loaded} == {"600000", "688981"}
    star = next(s for s in loaded if s.code == "688981")
    assert star.board == Board.STAR and star.list_date == date(2020, 7, 16)


def test_save_and_load_calendar(tmp_path):
    store = LocalStore(tmp_path)
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    store.save_calendar(dates)
    assert store.load_calendar() == dates


# Fix 1: NaT / None dates must be dropped, not persisted
def test_save_bars_drops_nat_date(tmp_path):
    """save_bars + load_bars + last_bar_date must not raise when a row has NaT date,
    and that row is dropped."""
    store = LocalStore(tmp_path)
    df = _bars([date(2024, 1, 2), None, date(2024, 1, 4)], [10.0, 11.0, 12.0])
    # NaT row: replace None with pd.NaT for the date column
    df["date"] = df["date"].astype(object)
    df.loc[1, "date"] = pd.NaT
    store.save_bars("600000", df)
    out = store.load_bars("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 4)]
    assert list(out["close"]) == [10.0, 12.0]
    # last_bar_date must not crash
    assert store.last_bar_date("600000") == date(2024, 1, 4)


def test_append_bars_drops_nat_date(tmp_path):
    """append_bars must drop NaT date rows without raising."""
    store = LocalStore(tmp_path)
    store.save_bars("600000", _bars([date(2024, 1, 2)], [10.0]))
    df2 = _bars([None, date(2024, 1, 3)], [99.0, 11.0])
    df2["date"] = df2["date"].astype(object)
    df2.loc[0, "date"] = pd.NaT
    store.append_bars("600000", df2)
    out = store.load_bars("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 3)]
    # last_bar_date must not crash
    assert store.last_bar_date("600000") == date(2024, 1, 3)
