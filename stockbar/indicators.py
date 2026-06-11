"""价量技术指标（作用于按时间升序的 pd.Series）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0.0, 100.0)   # 全涨：avg_loss=0 → 100
    out = out.where(avg_gain != 0.0, 0.0)      # 全跌：avg_gain=0 → 0
    out[avg_gain.isna()] = np.nan              # 数据不足保持 NaN
    return out


def bias(close: pd.Series, n: int) -> pd.Series:
    """乖离率 (close - ma_n) / ma_n。"""
    ma = sma(close, n)
    return (close - ma) / ma


def ret_n(close: pd.Series, n: int) -> pd.Series:
    """近 n 期收益率 close[t]/close[t-n] - 1。"""
    return close.pct_change(n)


def is_new_high(close: pd.Series, n: int) -> bool:
    """最新收盘是否为近 n 期（含当日）最高。数据不足返回 False。"""
    window = close.tail(n + 1)
    if len(window) < n + 1 or window.isna().any():
        # 至少需要 n+1 个点才能说"近 n 期新高"；不足时用现有窗口判断
        window = close.dropna()
        if window.empty:
            return False
    return bool(window.iloc[-1] >= window.max())


def is_ma_bullish(close: pd.Series) -> bool:
    """均线多头排列 ma5 > ma10 > ma20（取最新值）。任一为 NaN 返回 False。"""
    m5 = sma(close, 5).iloc[-1]
    m10 = sma(close, 10).iloc[-1]
    m20 = sma(close, 20).iloc[-1]
    if any(pd.isna(x) for x in (m5, m10, m20)):
        return False
    return bool(m5 > m10 > m20)


def is_vol_price_up(close: pd.Series, volume: pd.Series, n: int = 5) -> bool:
    """量价齐升：最新收盘较上一日上涨，且最新成交量 > 近 n 日均量。"""
    if len(close) < 2:
        return False
    price_up = bool(close.iloc[-1] > close.iloc[-2])
    vol_ma = sma(volume, n).iloc[-1]
    vol_up = (not pd.isna(vol_ma)) and bool(volume.iloc[-1] > vol_ma)
    return price_up and vol_up
