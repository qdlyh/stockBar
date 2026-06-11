import numpy as np
import pandas as pd

from stockbar.indicators import (
    sma, rsi, bias, ret_n, is_new_high, is_ma_bullish, is_vol_price_up,
)


def test_sma_last_value():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert sma(s, 3).iloc[-1] == 4.0   # (3+4+5)/3
    assert np.isnan(sma(s, 10).iloc[-1])  # 数据不足


def test_rsi_all_up_is_100_all_down_is_0():
    up = pd.Series([float(i) for i in range(1, 20)])     # 单调上涨
    down = pd.Series([float(i) for i in range(20, 1, -1)])  # 单调下跌
    assert rsi(up, 14).iloc[-1] == 100.0
    assert rsi(down, 14).iloc[-1] == 0.0


def test_rsi_insufficient_returns_nan():
    s = pd.Series([1.0, 2, 3])
    assert np.isnan(rsi(s, 14).iloc[-1])


def test_bias_relative_to_ma():
    s = pd.Series([10.0] * 4 + [12.0])   # ma5 = (10*4+12)/5 = 10.4
    b = bias(s, 5).iloc[-1]
    assert abs(b - (12.0 - 10.4) / 10.4) < 1e-9


def test_ret_n():
    s = pd.Series([10.0, 11, 12, 9])  # ret over 3 = 9/10 - 1 = -0.1
    assert abs(ret_n(s, 3).iloc[-1] - (-0.1)) < 1e-9


def test_is_new_high():
    assert is_new_high(pd.Series([1.0, 2, 3, 5]), 3) is True   # 5 是近3日(3,5)及自身最高
    assert is_new_high(pd.Series([5.0, 4, 3, 2]), 3) is False


def test_is_ma_bullish():
    # 多头排列 ma5>ma10>ma20 用上升序列构造
    s = pd.Series([float(i) for i in range(1, 40)])
    assert is_ma_bullish(s) is True
    s2 = pd.Series([float(i) for i in range(40, 1, -1)])
    assert is_ma_bullish(s2) is False


def test_is_vol_price_up():
    close = pd.Series([10.0, 10, 10, 10, 10, 11])      # 末日上涨
    vol = pd.Series([100.0, 100, 100, 100, 100, 300])  # 末日放量 > 5日均量
    assert is_vol_price_up(close, vol) is True
    assert is_vol_price_up(pd.Series([11.0, 10]), pd.Series([100.0, 300])) is False  # 末日下跌
