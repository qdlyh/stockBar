import pandas as pd

from stockbar.datafeed.instruments import Board
from stockbar.universe import build_tradable, split_pools


def _features(rows: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index")


def test_build_tradable_filters():
    feat = _features({
        "OK":   dict(board=Board.MAIN, is_st=False, bars_count=100, suspended=False, avg_amount20=5e7),
        "ST":   dict(board=Board.MAIN, is_st=True,  bars_count=100, suspended=False, avg_amount20=5e7),
        "BSE":  dict(board=Board.BSE,  is_st=False, bars_count=100, suspended=False, avg_amount20=5e7),
        "SUSP": dict(board=Board.MAIN, is_st=False, bars_count=100, suspended=True,  avg_amount20=5e7),
        "NEW":  dict(board=Board.MAIN, is_st=False, bars_count=30,  suspended=False, avg_amount20=5e7),
        "ILLIQ":dict(board=Board.MAIN, is_st=False, bars_count=100, suspended=False, avg_amount20=1e6),
    })
    out = build_tradable(feat, min_amount=1e7, min_bars=60)
    assert set(out.index) == {"OK"}


def test_split_pools_left_and_right():
    # 左池：close<ma60 且 ret20<=-0.08 且 pb 处于后1/3
    # 右池：close>ma20 且 ma_bullish 且 ret20>0
    feat = _features({
        "LEFT":  dict(close=8.0,  ma20=9.0, ma60=10.0, ret20=-0.12, ma_bullish=False, pb=1.0),
        "RIGHT": dict(close=12.0, ma20=10.0, ma60=9.0, ret20=0.10,  ma_bullish=True,  pb=5.0),
        "MID":   dict(close=10.0, ma20=10.0, ma60=10.0, ret20=0.0,  ma_bullish=False, pb=3.0),
    })
    left, right = split_pools(feat)
    assert "LEFT" in left and "LEFT" not in right
    assert "RIGHT" in right and "RIGHT" not in left
    assert "MID" not in left and "MID" not in right


def test_split_pools_left_requires_low_pb_tercile():
    # 三只都满足 close<ma60 且 ret20<=-0.08，但只有 pb 后1/3 进左池
    feat = _features({
        "A": dict(close=8.0, ma20=9, ma60=10, ret20=-0.1, ma_bullish=False, pb=1.0),  # 最低 pb
        "B": dict(close=8.0, ma20=9, ma60=10, ret20=-0.1, ma_bullish=False, pb=5.0),
        "C": dict(close=8.0, ma20=9, ma60=10, ret20=-0.1, ma_bullish=False, pb=9.0),
    })
    left, _ = split_pools(feat)
    assert "A" in left and "B" not in left and "C" not in left
