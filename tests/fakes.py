"""测试用内存数据源。"""
from __future__ import annotations

from datetime import date

import pandas as pd

from stockbar.datafeed.source import (
    BAR_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    DataSource,
    StockInfo,
)


class FakeDataSource(DataSource):
    def __init__(
        self,
        stocks: list[StockInfo] | None = None,
        bars: dict[str, pd.DataFrame] | None = None,
        fundamentals: dict[str, pd.DataFrame] | None = None,
        trading_dates: list[date] | None = None,
    ):
        self._stocks = stocks or []
        self._bars = bars or {}
        self._fundamentals = fundamentals or {}
        self._trading_dates = sorted(trading_dates or [])

    def list_stocks(self) -> list[StockInfo]:
        return list(self._stocks)

    def get_daily_bars(self, code, start, end) -> pd.DataFrame:
        df = self._bars.get(code, pd.DataFrame(columns=BAR_COLUMNS))
        if df.empty:
            return df.copy()
        m = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[m].reset_index(drop=True)

    def get_fundamentals(self, code, start, end) -> pd.DataFrame:
        df = self._fundamentals.get(code, pd.DataFrame(columns=FUNDAMENTAL_COLUMNS))
        if df.empty:
            return df.copy()
        m = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[m].reset_index(drop=True)

    def get_trading_dates(self, start, end) -> list[date]:
        return [d for d in self._trading_dates if start <= d <= end]
