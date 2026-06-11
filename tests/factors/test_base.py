import numpy as np
import pandas as pd

from stockbar.factors.base import (
    FactorSpec, winsorize, zscore, standardize, score_pool, select_top_n,
)


def test_winsorize_clips_tails():
    s = pd.Series([1.0, 2, 3, 4, 100])
    out = winsorize(s, lower=0.0, upper=0.5)   # upper 50% 分位=3
    assert out.max() == 3.0


def test_zscore_mean0_std1():
    s = pd.Series([1.0, 2, 3, 4, 5])
    z = zscore(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


def test_zscore_constant_returns_zeros():
    s = pd.Series([5.0, 5, 5])
    assert (zscore(s) == 0.0).all()


def test_standardize_winsorizes_then_zscores():
    s = pd.Series([1.0, 2, 3, 4, 1000])
    out = standardize(s)
    assert not out.isna().any()
    assert out.idxmax() == 4   # 极端值裁剪后仍最大


def test_score_pool_direction_and_rank():
    feat = pd.DataFrame({
        "pb":  [1.0, 2.0, 3.0],     # 越小越好 → -1
        "mom": [0.1, 0.2, 0.3],     # 越大越好 → +1
    }, index=["A", "B", "C"])
    specs = [FactorSpec("pb", -1.0), FactorSpec("mom", +1.0)]
    score = score_pool(feat, ["A", "B", "C"], specs)
    # A: 低pb(好) + 低mom(差); C: 高pb(差) + 高mom(好) → 对称，B 居中
    assert set(score.index) == {"A", "B", "C"}
    assert abs(score["B"]) < 1e-9


def test_score_pool_subsets_to_codes():
    feat = pd.DataFrame({"mom": [0.1, 0.2, 0.3]}, index=["A", "B", "C"])
    score = score_pool(feat, ["A", "B"], [FactorSpec("mom", 1.0)])
    assert set(score.index) == {"A", "B"}


def test_select_top_n():
    score = pd.Series({"A": 3.0, "B": 1.0, "C": 2.0})
    assert select_top_n(score, 2) == ["A", "C"]


def test_score_pool_empty():
    feat = pd.DataFrame({"mom": [0.1]}, index=["A"])
    score = score_pool(feat, [], [FactorSpec("mom", 1.0)])
    assert score.empty
