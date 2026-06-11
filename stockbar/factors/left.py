"""左侧（逆势/价值/均值回归）因子：越便宜、越超跌越好。"""
from __future__ import annotations

import pandas as pd

from stockbar.factors.base import FactorSpec, score_pool

LEFT_FACTORS = [
    FactorSpec("pb", -1.0),       # 低估值
    FactorSpec("pe", -1.0),       # 低估值
    FactorSpec("bias20", -1.0),   # 距 MA20 负乖离越大（越超跌）越好
    FactorSpec("rsi", -1.0),      # RSI 越低（越超卖）越好
    FactorSpec("ret20", -1.0),    # 近 20 日跌幅越大越好
]


def score_left(features: pd.DataFrame, codes: list[str]) -> pd.Series:
    return score_pool(features, codes, LEFT_FACTORS)
