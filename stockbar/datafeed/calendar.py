"""A股交易日历。"""
from __future__ import annotations

import bisect
from datetime import date


class TradingCalendar:
    def __init__(self, trading_dates: list[date]):
        self._dates = sorted(set(trading_dates))
        self._set = set(self._dates)

    def is_trading_day(self, d: date) -> bool:
        return d in self._set

    def next_trading_day(self, d: date) -> date:
        """严格大于 d 的下一个交易日。"""
        i = bisect.bisect_right(self._dates, d)
        if i >= len(self._dates):
            raise IndexError(f"{d} 之后无交易日数据")
        return self._dates[i]

    def prev_trading_day(self, d: date) -> date:
        """严格小于 d 的上一个交易日。"""
        i = bisect.bisect_left(self._dates, d)
        if i == 0:
            raise IndexError(f"{d} 之前无交易日数据")
        return self._dates[i - 1]

    def trading_days_between(self, start: date, end: date) -> list[date]:
        lo = bisect.bisect_left(self._dates, start)
        hi = bisect.bisect_right(self._dates, end)
        return self._dates[lo:hi]

    def offset(self, d: date, n: int) -> date:
        """从交易日 d 向前(n>0)/向后(n<0)偏移 n 个交易日。d 必须是交易日。"""
        if d not in self._set:
            raise ValueError(f"{d} 不是交易日")
        i = self._dates.index(d) + n
        if i < 0 or i >= len(self._dates):
            raise IndexError(f"偏移越界: {d} offset {n}")
        return self._dates[i]
