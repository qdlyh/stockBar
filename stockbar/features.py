"""把一个截面日的候选股算成特征表（index=code）。"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stockbar.datafeed.source import StockInfo
from stockbar import indicators as ind

FEATURE_COLUMNS = [
    "board", "is_st", "bars_count", "suspended", "close",
    "ma5", "ma10", "ma20", "ma60", "rsi", "bias20", "bias60",
    "ret20", "ret60", "new_high60", "ma_bullish", "vol_price_up",
    "pb", "pe", "avg_amount20",
]


def _last_on_or_before(df: pd.DataFrame, col: str, as_of: date):
    if df is None or df.empty:
        return np.nan
    sub = df[df["date"] <= as_of]
    if sub.empty:
        return np.nan
    return sub.sort_values("date").iloc[-1][col]


def _feature_row(info: StockInfo, bars: pd.DataFrame, funds: pd.DataFrame, as_of: date) -> dict:
    bars = bars[bars["date"] <= as_of].sort_values("date")
    close = bars["close"].reset_index(drop=True)
    volume = bars["volume"].reset_index(drop=True)
    last_bar_date = bars["date"].iloc[-1]
    return {
        "board": info.board,
        "is_st": info.is_st,
        "bars_count": len(bars),
        "suspended": last_bar_date != as_of,
        "close": float(close.iloc[-1]),
        "ma5": float(ind.sma(close, 5).iloc[-1]),
        "ma10": float(ind.sma(close, 10).iloc[-1]),
        "ma20": float(ind.sma(close, 20).iloc[-1]),
        "ma60": float(ind.sma(close, 60).iloc[-1]),
        "rsi": float(ind.rsi(close, 14).iloc[-1]),
        "bias20": float(ind.bias(close, 20).iloc[-1]),
        "bias60": float(ind.bias(close, 60).iloc[-1]),
        "ret20": float(ind.ret_n(close, 20).iloc[-1]),
        "ret60": float(ind.ret_n(close, 60).iloc[-1]),
        "new_high60": ind.is_new_high(close, 60),
        "ma_bullish": ind.is_ma_bullish(close),
        "vol_price_up": ind.is_vol_price_up(close, volume),
        "pb": float(_last_on_or_before(funds, "pb", as_of)),
        "pe": float(_last_on_or_before(funds, "pe", as_of)),
        "avg_amount20": float(bars["amount"].tail(20).mean()),
    }


def compute_features(
    stocks: list[StockInfo],
    panel: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    as_of: date,
) -> pd.DataFrame:
    rows = {}
    for info in stocks:
        bars = panel.get(info.code)
        if bars is None or bars.empty:
            continue
        if bars[bars["date"] <= as_of].empty:
            continue
        rows[info.code] = _feature_row(info, bars, fundamentals.get(info.code), as_of)
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.DataFrame.from_dict(rows, orient="index")[FEATURE_COLUMNS]
