from datetime import date

import pandas as pd

from stockbar.datafeed.instruments import Board
from stockbar.datafeed.source import StockInfo
from tests.fakes import FakeDataSource


def test_fake_filters_bars_by_date_range():
    bars = pd.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        "open": [10.0, 10.5, 11.0], "high": [10.6, 11.1, 11.2],
        "low": [9.9, 10.4, 10.8], "close": [10.5, 11.0, 10.9],
        "volume": [100, 120, 90], "amount": [1050.0, 1320.0, 981.0],
    })
    src = FakeDataSource(bars={"600000": bars})
    out = src.get_daily_bars("600000", date(2024, 1, 3), date(2024, 1, 4))
    assert list(out["date"]) == [date(2024, 1, 3), date(2024, 1, 4)]


def test_fake_trading_dates_range():
    src = FakeDataSource(trading_dates=[date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)])
    assert src.get_trading_dates(date(2024, 1, 3), date(2024, 1, 10)) == [
        date(2024, 1, 3), date(2024, 1, 4)
    ]


def test_fake_list_stocks():
    s = StockInfo("600000", "浦发银行", Board.MAIN, date(1999, 11, 10), False)
    src = FakeDataSource(stocks=[s])
    assert src.list_stocks()[0].code == "600000"
