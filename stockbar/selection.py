"""每日选股整合：数据 → 特征 → 选股池 → 左右打分排名。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from stockbar.datafeed.source import StockInfo
from stockbar.datafeed.store import LocalStore
from stockbar.features import compute_features
from stockbar.universe import build_tradable, split_pools
from stockbar.factors.left import score_left
from stockbar.factors.right import score_right
from stockbar.factors.base import select_top_n


@dataclass(frozen=True)
class CandidateLists:
    as_of: date
    left: list[str]    # 左侧榜前 N
    right: list[str]   # 右侧榜前 N


def build_candidate_lists(
    store: LocalStore,
    stocks: list[StockInfo],
    as_of: date,
    top_n: int = 5,
    lookback: int = 120,
    min_amount: float = 1e7,
    min_bars: int = 63,
    fund_lookback: int = 450,
) -> CandidateLists:
    """从缓存读取近 lookback 自然日数据，输出左右榜前 top_n 候选。

    价格数据使用 lookback 窗口；财务数据（PB/PE）使用更宽的 fund_lookback 窗口，
    以确保季报数据在 lookback 之前时仍能被正确加载。
    """
    start = as_of - timedelta(days=lookback)
    fund_start = as_of - timedelta(days=fund_lookback)
    panel: dict[str, pd.DataFrame] = {}
    funds: dict[str, pd.DataFrame] = {}
    for info in stocks:
        bars = store.load_bars(info.code, start, as_of)
        if not bars.empty:
            panel[info.code] = bars
            funds[info.code] = store.load_fundamentals(info.code, fund_start, as_of)

    features = compute_features(stocks, panel, funds, as_of)
    tradable = build_tradable(features, min_amount=min_amount, min_bars=min_bars)
    left_codes, right_codes = split_pools(tradable)

    left_score = score_left(tradable, left_codes)
    right_score = score_right(tradable, right_codes)

    return CandidateLists(
        as_of=as_of,
        left=select_top_n(left_score, top_n),
        right=select_top_n(right_score, top_n),
    )
