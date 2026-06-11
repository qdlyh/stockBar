import pandas as pd

from stockbar.factors.left import LEFT_FACTORS, score_left
from stockbar.factors.right import RIGHT_FACTORS, score_right


def _feat(rows: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index")


def test_left_factors_columns():
    cols = {f.column for f in LEFT_FACTORS}
    assert cols == {"pb", "pe", "bias20", "rsi", "ret20"}
    assert all(f.direction == -1.0 for f in LEFT_FACTORS)   # 左侧全是"越小越好"


def test_right_factors_columns():
    cols = {f.column for f in RIGHT_FACTORS}
    assert cols == {"ret20", "ret60", "new_high60", "ma_bullish", "vol_price_up"}
    assert all(f.direction == 1.0 for f in RIGHT_FACTORS)    # 右侧全是"越大越好"


def test_score_left_prefers_cheaper_more_oversold():
    feat = _feat({
        "CHEAP": dict(pb=1.0, pe=5.0,  bias20=-0.2, rsi=20.0, ret20=-0.15),
        "RICH":  dict(pb=5.0, pe=50.0, bias20=0.05, rsi=70.0, ret20=-0.02),
    })
    score = score_left(feat, ["CHEAP", "RICH"])
    assert score["CHEAP"] > score["RICH"]


def test_score_right_prefers_stronger_trend():
    feat = _feat({
        "STRONG": dict(ret20=0.2, ret60=0.4, new_high60=True,  ma_bullish=True,  vol_price_up=True),
        "WEAK":   dict(ret20=0.0, ret60=0.0, new_high60=False, ma_bullish=False, vol_price_up=False),
    })
    score = score_right(feat, ["STRONG", "WEAK"])
    assert score["STRONG"] > score["WEAK"]
