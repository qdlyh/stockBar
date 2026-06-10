from datetime import date

import pytest

from stockbar.datafeed.calendar import TradingCalendar

# 模拟交易日：2024-01-02,03,04,05 (周二~周五), 跳过周末, 08,09
DATES = [date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9)]


def cal():
    return TradingCalendar(DATES)


def test_is_trading_day():
    c = cal()
    assert c.is_trading_day(date(2024, 1, 3)) is True
    assert c.is_trading_day(date(2024, 1, 6)) is False  # 周六


def test_next_trading_day_skips_weekend():
    c = cal()
    assert c.next_trading_day(date(2024, 1, 5)) == date(2024, 1, 8)
    assert c.next_trading_day(date(2024, 1, 6)) == date(2024, 1, 8)


def test_prev_trading_day():
    c = cal()
    assert c.prev_trading_day(date(2024, 1, 8)) == date(2024, 1, 5)


def test_trading_days_between_inclusive():
    c = cal()
    assert c.trading_days_between(date(2024, 1, 3), date(2024, 1, 8)) == [
        date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)
    ]


def test_offset_forward_and_back():
    c = cal()
    assert c.offset(date(2024, 1, 3), 2) == date(2024, 1, 5)
    assert c.offset(date(2024, 1, 8), -2) == date(2024, 1, 4)


def test_offset_out_of_range_raises():
    c = cal()
    with pytest.raises(IndexError):
        c.offset(date(2024, 1, 9), 5)
