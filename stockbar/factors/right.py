"""右侧（顺势/动量/趋势）因子：越强势、越趋势越好。"""
from __future__ import annotations

import pandas as pd

from stockbar.factors.base import FactorSpec, score_pool

RIGHT_FACTORS = [
    FactorSpec("ret20", 1.0),         # 20 日动量
    FactorSpec("ret60", 1.0),         # 60 日动量
    FactorSpec("new_high60", 1.0),    # 创 60 日新高
    FactorSpec("ma_bullish", 1.0),    # 均线多头排列
    FactorSpec("vol_price_up", 1.0),  # 量价齐升
]


def score_right(features: pd.DataFrame, codes: list[str]) -> pd.Series:
    return score_pool(features, codes, RIGHT_FACTORS)
