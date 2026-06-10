"""数据源抽象接口与公共数据类型。

上层（选股池/因子/回测）只依赖 DataSource，不直接依赖 AKShare，
便于未来替换为 Tushare 等其它实现。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd

from stockbar.datafeed.instruments import Board

# 行情 DataFrame 标准列（前复权），按 date 升序：
BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]
# 财务 DataFrame 标准列，按 date 升序：
FUNDAMENTAL_COLUMNS = ["date", "pe", "pb"]


@dataclass(frozen=True)
class StockInfo:
    code: str          # 6 位代码
    name: str          # 名称（含 ST 前缀时 is_st=True）
    board: Board
    list_date: date    # 上市日期
    is_st: bool


class DataSource(ABC):
    """A股 数据源抽象接口。实现需返回符合上述标准列的 DataFrame。"""

    @abstractmethod
    def list_stocks(self) -> list[StockInfo]:
        """返回全市场股票基础信息。"""

    @abstractmethod
    def get_daily_bars(self, code: str, start: date, end: date) -> pd.DataFrame:
        """前复权日线，列为 BAR_COLUMNS，按 date 升序；无数据返回空 DataFrame。"""

    @abstractmethod
    def get_fundamentals(self, code: str, start: date, end: date) -> pd.DataFrame:
        """每日估值指标，列为 FUNDAMENTAL_COLUMNS，按 date 升序。"""

    @abstractmethod
    def get_trading_dates(self, start: date, end: date) -> list[date]:
        """[start, end] 区间内的交易日（升序）。"""
