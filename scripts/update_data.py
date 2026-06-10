"""增量更新本地缓存：行情、股票列表、交易日历。

用法（已 pip install -e . 且联网）：
    python scripts/update_data.py --root data --until 2024-12-31
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from stockbar.datafeed.akshare_source import AkshareSource
from stockbar.datafeed.source import DataSource
from stockbar.datafeed.store import LocalStore

_EPOCH = date(1990, 12, 19)  # A股 开市日，全量抓取起点


def update_symbol(src: DataSource, store: LocalStore, code: str, until: date) -> int:
    """把 code 的行情增量更新到 until。返回新增条数。"""
    last = store.last_bar_date(code)
    start = (last + timedelta(days=1)) if last is not None else _EPOCH
    if start > until:
        return 0
    df = src.get_daily_bars(code, start, until)
    if df.empty:
        return 0
    if last is None:
        store.save_bars(code, df)
    else:
        store.append_bars(code, df)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--until", default=date.today().isoformat())
    args = parser.parse_args()

    until = date.fromisoformat(args.until)
    src = AkshareSource()
    store = LocalStore(Path(args.root))

    print("更新交易日历…")
    store.save_calendar(src.get_trading_dates(_EPOCH, until))

    print("更新股票列表…")
    stocks = src.list_stocks()
    store.save_stocks(stocks)

    print(f"更新 {len(stocks)} 只股票行情…")
    for i, s in enumerate(stocks, 1):
        try:
            n = update_symbol(src, store, s.code, until)
        except Exception as e:  # 单只失败不影响整体
            print(f"  [{i}/{len(stocks)}] {s.code} 失败: {e}")
            continue
        if n:
            print(f"  [{i}/{len(stocks)}] {s.code} +{n}")


if __name__ == "__main__":
    main()
