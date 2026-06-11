"""因子框架：标准化、合成打分、选前 N。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    column: str        # 特征表中的列名
    direction: float   # +1 越大越好；-1 越小越好
    weight: float = 1.0


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def standardize(s: pd.Series) -> pd.Series:
    return zscore(winsorize(s))


def score_pool(features: pd.DataFrame, codes: list[str], specs: list[FactorSpec]) -> pd.Series:
    """对 codes 子集，按 specs 截面标准化+合成，返回合成分（index=code）。"""
    codes = [c for c in codes if c in features.index]
    if not codes:
        return pd.Series(dtype="float64")
    sub = features.loc[codes]
    total = pd.Series(0.0, index=sub.index)
    for spec in specs:
        col = sub[spec.column].astype("float64")
        total = total + spec.direction * spec.weight * standardize(col)
    return total


def select_top_n(score: pd.Series, n: int) -> list[str]:
    """合成分降序取前 n 的 codes。"""
    return list(score.sort_values(ascending=False).head(n).index)
