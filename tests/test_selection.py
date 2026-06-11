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


# Fix 2: fundamentals dated ~200 days before as_of (beyond 120-day price window)
# should still be loaded with the wider fund_lookback=450 default
def test_build_candidate_lists_fund_lookback_200_days(tmp_path):
    """A stock whose only fundamental row is ~200 days before as_of must still get
    a non-NaN pb (and thus appear in result.left) when fund_lookback=450 (default).
    Without Fix 2, pb would be NaN and select_top_n (after Fix 1) would drop it.
    """
    store = LocalStore(tmp_path)
    as_of = date(2024, 4, 1)
    # Price bars: 70 trading days ending at as_of (within 120-day price window)
    price_days = list(pd.date_range(end=pd.Timestamp(as_of), periods=70).date)
    # Down-trend so the stock ends up in the left pool
    down = [20.0 - i * 0.15 for i in range(70)]
    store.save_bars("600010", _bars(price_days, down))
    # Fundamental row placed ~200 days before as_of — outside 120-day price window
    fund_date = as_of - timedelta(days=200)
    store.save_fundamentals("600010", pd.DataFrame({"date": [fund_date], "pe": [8.0], "pb": [0.5]}))

    stocks = [StockInfo("600010", "远期报告", Board.MAIN, date(2000, 1, 1), False)]
    # With default fund_lookback=450 the fundamental should be found → pb != NaN
    result = build_candidate_lists(store, stocks, as_of, top_n=5, lookback=120)
    assert "600010" in result.left, (
        "Stock with fundamental row 200 days ago should appear in left pool "
        "when fund_lookback=450; got left=%s right=%s" % (result.left, result.right)
    )


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
