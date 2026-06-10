"""AKShare 数据源适配器。

注意：AKShare 接口列名/函数可能随版本变动。集成测试失败时优先核对
akshare 文档的返回列名，再调整 _normalize_* 映射。
"""
from __future__ import annotations

from datetime import date

import akshare as ak
import pandas as pd

from stockbar.datafeed.instruments import Board, classify_board
from stockbar.datafeed.source import (
    BAR_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    DataSource,
    StockInfo,
)

_BAR_MAP = {
    "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
    "收盘": "close", "成交量": "volume", "成交额": "amount",
}


def _normalize_bars(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = raw.rename(columns=_BAR_MAP)
    df = df[[c for c in BAR_COLUMNS if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


class AkshareSource(DataSource):
    def list_stocks(self) -> list[StockInfo]:
        # 代码+名称
        names = ak.stock_info_a_code_name()  # 列: code, name
        out: list[StockInfo] = []
        for _, row in names.iterrows():
            code = str(row["code"]).zfill(6)
            try:
                board = classify_board(code)
            except ValueError:
                continue
            name = str(row["name"])
            is_st = "ST" in name.upper()
            # 上市日期：逐只查较慢，这里给占位最早日期，update_data 脚本可后补。
            out.append(StockInfo(code, name, board, date(1990, 12, 19), is_st))
        return out

    def get_daily_bars(self, code: str, start: date, end: date) -> pd.DataFrame:
        raw = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
        return _normalize_bars(raw)

    def get_fundamentals(self, code: str, start: date, end: date) -> pd.DataFrame:
        raw = ak.stock_a_indicator_lg(symbol=code)  # 列含 trade_date, pe, pb
        if raw is None or raw.empty:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        df = raw.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        keep = [c for c in FUNDAMENTAL_COLUMNS if c in df.columns]
        return df[keep].sort_values("date").reset_index(drop=True)

    def get_trading_dates(self, start: date, end: date) -> list[date]:
        cal = ak.tool_trade_date_hist_sina()  # 列: trade_date
        s = pd.to_datetime(cal["trade_date"]).dt.date
        return sorted(d for d in s if start <= d <= end)
