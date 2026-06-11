from datetime import date

import pandas as pd

from stockbar.datafeed.instruments import Board
from stockbar.datafeed.source import StockInfo
from stockbar.features import compute_features


def _bars(dates, closes, vols=None, amts=None):
    n = len(dates)
    vols = vols or [100.0] * n
    amts = amts or [1_000_000.0] * n
    return pd.DataFrame({
        "date": dates, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": vols, "amount": amts,
    })


def _trading_days(n, end=date(2024, 4, 1)):
    # 生成 n 个连续自然日（测试无需真实交易日历）
    return list(pd.date_range(end=pd.Timestamp(end), periods=n).date)


def test_features_basic_columns_and_suspended():
    days = _trading_days(70)
    closes = [10.0 + i * 0.1 for i in range(70)]   # 缓慢上涨
    panel = {"600000": _bars(days, closes)}
    funds = {"600000": pd.DataFrame({"date": [days[-1]], "pe": [15.0], "pb": [1.2]})}
    stocks = [StockInfo("600000", "浦发银行", Board.MAIN, date(1999, 11, 10), False)]
    feat = compute_features(stocks, panel, funds, as_of=days[-1])
    assert "600000" in feat.index
    row = feat.loc["600000"]
    assert row["bars_count"] == 70
    assert row["suspended"] == False
    assert row["is_st"] == False
    assert row["board"] == Board.MAIN
    assert abs(row["pb"] - 1.2) < 1e-9
    assert row["close"] == closes[-1]
    assert row["ma_bullish"] == True   # 单调上涨


def test_features_marks_suspended_when_no_bar_on_as_of():
    days = _trading_days(70)
    panel = {"600000": _bars(days, [10.0] * 70)}
    stocks = [StockInfo("600000", "X", Board.MAIN, date(2000, 1, 1), False)]
    later = days[-1] + pd.Timedelta(days=5)
    feat = compute_features(stocks, panel, {}, as_of=later.date() if hasattr(later, "date") else later)
    assert feat.loc["600000"]["suspended"] == True


def test_features_pb_uses_last_on_or_before_as_of():
    days = _trading_days(70)
    panel = {"600000": _bars(days, [10.0] * 70)}
    funds = {"600000": pd.DataFrame({
        "date": [days[-10], days[-3]], "pe": [10.0, 11.0], "pb": [1.0, 2.0]})}
    stocks = [StockInfo("600000", "X", Board.MAIN, date(2000, 1, 1), False)]
    feat = compute_features(stocks, panel, funds, as_of=days[-1])
    assert feat.loc["600000"]["pb"] == 2.0   # 取 as_of 前最近一条


def test_features_skips_codes_without_bars():
    stocks = [StockInfo("600000", "X", Board.MAIN, date(2000, 1, 1), False)]
    feat = compute_features(stocks, {}, {}, as_of=date(2024, 4, 1))
    assert feat.empty
