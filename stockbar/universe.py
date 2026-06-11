"""选股池：第一层基础可交易过滤 + 第二层按市场状态拆左/右池。"""
from __future__ import annotations

import pandas as pd

from stockbar.datafeed.instruments import Board

# 左池：估值后 1/3
LEFT_PB_QUANTILE = 1.0 / 3.0
# 左池：近 20 日跌幅阈值
LEFT_RET20_MAX = -0.08


def build_tradable(features: pd.DataFrame, min_amount: float = 1e7, min_bars: int = 60) -> pd.DataFrame:
    """第一层：剔除 ST、北交所、停牌、上市不足、流动性过低。返回过滤后的特征表。"""
    if features.empty:
        return features
    m = (
        (~features["is_st"])
        & (features["board"] != Board.BSE)
        & (~features["suspended"])
        & (features["bars_count"] >= min_bars)
        & (features["avg_amount20"] >= min_amount)
    )
    return features[m]


def split_pools(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """第二层：返回 (左池 codes, 右池 codes)。输入应为已过可交易过滤的特征表。"""
    if features.empty:
        return [], []

    # 左池：弱势 + 超跌 + 低估（pb 后 1/3）
    weak = (features["close"] < features["ma60"]) & (features["ret20"] <= LEFT_RET20_MAX)
    if weak.any():
        pb_threshold = features.loc[weak, "pb"].quantile(LEFT_PB_QUANTILE)
        left_mask = weak & (features["pb"] <= pb_threshold)
    else:
        left_mask = weak

    # 右池：强势 + 趋势
    right_mask = (
        (features["close"] > features["ma20"])
        & (features["ma_bullish"])
        & (features["ret20"] > 0)
    )

    return list(features[left_mask].index), list(features[right_mask].index)
