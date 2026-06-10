"""A股板块分类（按证券代码前缀）。"""
from __future__ import annotations

from enum import Enum


class Board(Enum):
    MAIN = "main"   # 主板（沪：600/601/603/605；深：000/001/002/003）
    GEM = "gem"     # 创业板（300/301）
    STAR = "star"   # 科创板（688/689）
    BSE = "bse"     # 北交所（4/8 开头、920）


def classify_board(code: str) -> Board:
    """按 6 位证券代码前缀返回所属板块。无法识别时抛 ValueError。"""
    if not (isinstance(code, str) and len(code) == 6 and code.isdigit()):
        raise ValueError(f"非法证券代码: {code!r}")
    if code.startswith(("688", "689")):
        return Board.STAR
    if code.startswith(("300", "301")):
        return Board.GEM
    if code.startswith(("4", "8", "920")):
        return Board.BSE
    if code.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return Board.MAIN
    raise ValueError(f"未知板块前缀: {code!r}")
