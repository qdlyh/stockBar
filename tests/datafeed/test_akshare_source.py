from datetime import date

import pandas as pd
import pytest

from stockbar.datafeed.akshare_source import (
    AkshareSource,
    _normalize_bars,
    _normalize_fundamentals,
    _is_st,
)


def test_normalize_bars_maps_chinese_columns():
    raw = pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03"],
        "开盘": [10.0, 10.5], "最高": [10.6, 11.1],
        "最低": [9.9, 10.4], "收盘": [10.5, 11.0],
        "成交量": [100, 120], "成交额": [1050.0, 1320.0],
        "涨跌幅": [1.0, 4.8],  # 多余列应被丢弃
    })
    out = _normalize_bars(raw)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert out["date"].iloc[0] == date(2024, 1, 2)
    assert out["close"].iloc[1] == 11.0


def test_normalize_bars_empty():
    out = _normalize_bars(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]


# Fix 2: missing columns in _normalize_bars should raise ValueError
def test_normalize_bars_missing_column_raises():
    """If a non-empty raw frame lacks 成交额 (amount), ValueError must be raised."""
    raw = pd.DataFrame({
        "日期": ["2024-01-02"],
        "开盘": [10.0], "最高": [10.6], "最低": [9.9], "收盘": [10.5],
        "成交量": [100],
        # 成交额 intentionally missing
    })
    with pytest.raises(ValueError, match="amount"):
        _normalize_bars(raw)


# Fix 5: _normalize_fundamentals pure helper
def test_normalize_fundamentals_basic():
    """_normalize_fundamentals renames, filters date range, drops extras, sorts."""
    raw = pd.DataFrame({
        "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "pe": [10.0, 11.0, 12.0, 13.0],
        "pb": [1.0, 1.1, 1.2, 1.3],
        "extra_col": ["a", "b", "c", "d"],
    })
    out = _normalize_fundamentals(raw, date(2024, 1, 2), date(2024, 1, 3))
    assert list(out.columns) == ["date", "pe", "pb"]
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 3)]
    assert list(out["pe"]) == [11.0, 12.0]


def test_normalize_fundamentals_empty_raw():
    """Empty raw frame returns empty DataFrame with FUNDAMENTAL_COLUMNS."""
    from stockbar.datafeed.source import FUNDAMENTAL_COLUMNS
    out = _normalize_fundamentals(pd.DataFrame(), date(2024, 1, 1), date(2024, 1, 31))
    assert out.empty
    assert list(out.columns) == FUNDAMENTAL_COLUMNS


def test_normalize_fundamentals_none_raw():
    """None raw frame returns empty DataFrame with FUNDAMENTAL_COLUMNS."""
    from stockbar.datafeed.source import FUNDAMENTAL_COLUMNS
    out = _normalize_fundamentals(None, date(2024, 1, 1), date(2024, 1, 31))
    assert out.empty
    assert list(out.columns) == FUNDAMENTAL_COLUMNS


# Fix 6: _is_st helper
@pytest.mark.parametrize("name,expected", [
    ("*ST信威", True),
    ("ST安信", True),
    ("浦发银行", False),
    ("中芯国际", False),
])
def test_is_st(name, expected):
    assert _is_st(name) == expected


@pytest.mark.integration
def test_akshare_get_daily_bars_live():
    src = AkshareSource()
    df = src.get_daily_bars("600000", date(2024, 1, 2), date(2024, 1, 31))
    assert not df.empty
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert df["date"].is_monotonic_increasing


@pytest.mark.integration
def test_akshare_trading_dates_live():
    src = AkshareSource()
    dates = src.get_trading_dates(date(2024, 1, 1), date(2024, 1, 31))
    assert date(2024, 1, 2) in dates
    assert date(2024, 1, 6) not in dates  # 周六
