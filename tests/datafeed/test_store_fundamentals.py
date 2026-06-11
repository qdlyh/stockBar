from datetime import date

import pandas as pd

from stockbar.datafeed.store import LocalStore


def _funds(dates, pbs):
    return pd.DataFrame({"date": dates, "pe": [float(i) for i in range(len(dates))], "pb": pbs})


def test_save_and_load_fundamentals(tmp_path):
    store = LocalStore(tmp_path)
    df = _funds([date(2024, 1, 2), date(2024, 1, 3)], [1.0, 1.1])
    store.save_fundamentals("600000", df)
    out = store.load_fundamentals("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(out["pb"]) == [1.0, 1.1]
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 3)]


def test_append_fundamentals_dedupes(tmp_path):
    store = LocalStore(tmp_path)
    store.save_fundamentals("600000", _funds([date(2024, 1, 2), date(2024, 1, 3)], [1.0, 1.1]))
    store.append_fundamentals("600000", _funds([date(2024, 1, 3), date(2024, 1, 4)], [9.9, 1.2]))
    out = store.load_fundamentals("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    assert list(out["pb"]) == [1.0, 9.9, 1.2]


def test_last_fundamental_date(tmp_path):
    store = LocalStore(tmp_path)
    assert store.last_fundamental_date("600000") is None
    store.save_fundamentals("600000", _funds([date(2024, 1, 2)], [1.0]))
    assert store.last_fundamental_date("600000") == date(2024, 1, 2)


def test_load_fundamentals_missing_returns_empty(tmp_path):
    store = LocalStore(tmp_path)
    out = store.load_fundamentals("000001", date(2024, 1, 1), date(2024, 1, 31))
    assert out.empty
