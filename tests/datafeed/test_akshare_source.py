from datetime import date

import pandas as pd
import pytest

from stockbar.datafeed.akshare_source import (
    AkshareSource,
    _normalize_bars,
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
