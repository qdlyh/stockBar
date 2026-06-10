import pytest
from stockbar.datafeed.instruments import Board, classify_board


@pytest.mark.parametrize("code,expected", [
    ("600000", Board.MAIN),   # 沪主板
    ("601318", Board.MAIN),
    ("603259", Board.MAIN),
    ("605499", Board.MAIN),
    ("000001", Board.MAIN),   # 深主板
    ("001979", Board.MAIN),
    ("002594", Board.MAIN),   # 原中小板归主板
    ("003816", Board.MAIN),
    ("300750", Board.GEM),    # 创业板
    ("301029", Board.GEM),
    ("688981", Board.STAR),   # 科创板
    ("689009", Board.STAR),
    ("830799", Board.BSE),    # 北交所
    ("871981", Board.BSE),
    ("920819", Board.BSE),
])
def test_classify_board(code, expected):
    assert classify_board(code) == expected


def test_classify_board_rejects_bad_code():
    with pytest.raises(ValueError):
        classify_board("12345")    # 非6位
    with pytest.raises(ValueError):
        classify_board("999999")   # 未知前缀
