from datetime import date

import pandas as pd

from scripts.update_data import update_symbol
from stockbar.datafeed.store import LocalStore
from tests.fakes import FakeDataSource


def _bars(dates, closes):
    return pd.DataFrame({
        "date": dates, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [100] * len(dates), "amount": [1000.0] * len(dates),
    })


def test_update_symbol_fetches_full_when_empty(tmp_path):
    src = FakeDataSource(bars={"600000": _bars(
        [date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5])})
    store = LocalStore(tmp_path)
    n = update_symbol(src, store, "600000", until=date(2024, 1, 3))
    assert n == 2
    assert store.last_bar_date("600000") == date(2024, 1, 3)


def test_update_symbol_incremental_from_last(tmp_path):
    store = LocalStore(tmp_path)
    store.save_bars("600000", _bars([date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5]))
    # 源里有 1/2~1/5；增量应只抓 1/4、1/5（>last=1/3）
    src = FakeDataSource(bars={"600000": _bars(
        [date(2024, 1, d) for d in (2, 3, 4, 5)], [10.0, 10.5, 11.0, 11.5])})
    n = update_symbol(src, store, "600000", until=date(2024, 1, 5))
    assert n == 2
    assert store.last_bar_date("600000") == date(2024, 1, 5)
    full = store.load_bars("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(full["date"]) == [date(2024, 1, d) for d in (2, 3, 4, 5)]


def test_update_symbol_noop_when_up_to_date(tmp_path):
    store = LocalStore(tmp_path)
    store.save_bars("600000", _bars([date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5]))
    src = FakeDataSource(bars={"600000": _bars(
        [date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5])})
    n = update_symbol(src, store, "600000", until=date(2024, 1, 3))
    assert n == 0
