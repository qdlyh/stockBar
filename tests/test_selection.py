from datetime import date, timedelta

import pandas as pd

from stockbar.datafeed.instruments import Board
from stockbar.datafeed.source import StockInfo
from stockbar.datafeed.store import LocalStore
from stockbar.selection import build_candidate_lists


def _bars(dates, closes, amt=5e7):
    n = len(dates)
    return pd.DataFrame({
        "date": dates, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000.0] * n, "amount": [amt] * n,
    })


def test_build_candidate_lists_separates_left_right(tmp_path):
    store = LocalStore(tmp_path)
    days = list(pd.date_range(end=pd.Timestamp("2024-04-01"), periods=70).date)
    as_of = days[-1]

    # 下跌股（左池候选）：持续下跌，低 pb
    down = [20.0 - i * 0.15 for i in range(70)]
    store.save_bars("600001", _bars(days, down))
    store.save_fundamentals("600001", pd.DataFrame({"date": [as_of], "pe": [5.0], "pb": [0.8]}))

    # 上涨股（右池候选）：持续上涨
    up = [10.0 + i * 0.2 for i in range(70)]
    store.save_bars("600002", _bars(days, up))
    store.save_fundamentals("600002", pd.DataFrame({"date": [as_of], "pe": [40.0], "pb": [6.0]}))

    stocks = [
        StockInfo("600001", "跌", Board.MAIN, date(2000, 1, 1), False),
        StockInfo("600002", "涨", Board.MAIN, date(2000, 1, 1), False),
    ]
    result = build_candidate_lists(store, stocks, as_of, top_n=5, lookback=120)

    assert "600001" in result.left
    assert "600002" in result.right
    assert "600001" not in result.right
    assert "600002" not in result.left


def test_build_candidate_lists_excludes_st(tmp_path):
    store = LocalStore(tmp_path)
    days = list(pd.date_range(end=pd.Timestamp("2024-04-01"), periods=70).date)
    as_of = days[-1]
    down = [20.0 - i * 0.15 for i in range(70)]
    store.save_bars("600001", _bars(days, down))
    store.save_fundamentals("600001", pd.DataFrame({"date": [as_of], "pe": [5.0], "pb": [0.8]}))
    stocks = [StockInfo("600001", "*ST跌", Board.MAIN, date(2000, 1, 1), True)]
    result = build_candidate_lists(store, stocks, as_of, top_n=5, lookback=120)
    assert "600001" not in result.left and "600001" not in result.right
