# Phase 1：数据地基（datafeed）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭好数据层——抽象数据源接口、AKShare 实现、本地 Parquet/SQLite 缓存、交易日历、板块分类，使后续层能拿到干净、可缓存、离线可读的 A股 日线行情/财务/日历数据。

**Architecture:** 定义 `DataSource` 抽象接口（上层只依赖它），AKShare 作为一个适配器实现。所有可单元测试的逻辑（板块分类、交易日历、缓存读写）用内存 `FakeDataSource` / 临时目录做 TDD；AKShare 适配器只做轻量、可跳过的联网集成测试。数据落地为本地缓存后，回测/影子盘全程读本地。

**Tech Stack:** Python 3.11 · pandas · numpy · akshare · pyarrow(Parquet) · sqlite3(标准库) · pytest

设计文档：`docs/superpowers/specs/2026-06-10-ashare-dual-line-shadow-design.md`（第 3 节数据层）

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 包定义、依赖、pytest 配置 |
| `stockbar/__init__.py` | 包入口 |
| `stockbar/datafeed/__init__.py` | 子包入口，导出公共类型 |
| `stockbar/datafeed/instruments.py` | `Board` 枚举 + `classify_board(code)` 按代码前缀分板块 |
| `stockbar/datafeed/source.py` | `StockInfo` 数据类 + `DataSource` 抽象接口 |
| `stockbar/datafeed/calendar.py` | `TradingCalendar`：交易日判断与偏移 |
| `stockbar/datafeed/store.py` | `LocalStore`：Parquet 行情/财务 + SQLite 元数据缓存 |
| `stockbar/datafeed/akshare_source.py` | `AkshareSource`：`DataSource` 的 AKShare 实现 |
| `scripts/update_data.py` | 增量更新本地缓存的命令行脚本 |
| `tests/fakes.py` | `FakeDataSource`：测试用内存数据源 |
| `tests/datafeed/test_*.py` | 各模块单元测试 |

---

## Task 0：项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `stockbar/__init__.py`
- Create: `stockbar/datafeed/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/datafeed/__init__.py`

- [ ] **Step 1: 创建 `pyproject.toml`**

```toml
[project]
name = "stockbar"
version = "0.1.0"
description = "个人 A股 多因子左右双线影子盘系统"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "akshare>=1.12",
    "pyarrow>=14.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["stockbar*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: 联网集成测试（默认运行，离线可用 -m 'not integration' 跳过）",
]
```

- [ ] **Step 2: 创建空包文件**

`stockbar/__init__.py`：
```python
"""个人 A股 多因子左右双线影子盘系统。"""
```

`stockbar/datafeed/__init__.py`、`tests/__init__.py`、`tests/datafeed/__init__.py` 各写一行 docstring：
```python
"""datafeed: 数据源、缓存、交易日历、板块分类。"""
```
（后两个测试包写 `"""tests."""`）

- [ ] **Step 3: 建虚拟环境并安装**

Windows PowerShell：
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```
Expected: 安装成功，`pytest --version` 可运行。

- [ ] **Step 4: 确认 pytest 能收集到 0 用例**

Run: `pytest -q`
Expected: `no tests ran`（无报错）

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml stockbar/ tests/
git commit -m "chore: phase1 项目脚手架与依赖"
```

---

## Task 1：板块分类 `instruments.py`

**Files:**
- Create: `stockbar/datafeed/instruments.py`
- Test: `tests/datafeed/test_instruments.py`

- [ ] **Step 1: 写失败测试**

`tests/datafeed/test_instruments.py`：
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/datafeed/test_instruments.py -q`
Expected: FAIL（`ModuleNotFoundError: stockbar.datafeed.instruments`）

- [ ] **Step 3: 实现 `instruments.py`**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/datafeed/test_instruments.py -q`
Expected: PASS（全部参数化用例通过）

- [ ] **Step 5: Commit**

```bash
git add stockbar/datafeed/instruments.py tests/datafeed/test_instruments.py
git commit -m "feat: 板块分类 classify_board"
```

---

## Task 2：数据源接口 `source.py` + 测试用 FakeDataSource

**Files:**
- Create: `stockbar/datafeed/source.py`
- Create: `tests/fakes.py`
- Test: `tests/datafeed/test_fakes.py`

- [ ] **Step 1: 写 `source.py`（接口先行，无需测试驱动——纯声明）**

```python
"""数据源抽象接口与公共数据类型。

上层（选股池/因子/回测）只依赖 DataSource，不直接依赖 AKShare，
便于未来替换为 Tushare 等其它实现。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd

from stockbar.datafeed.instruments import Board

# 行情 DataFrame 标准列（前复权），按 date 升序：
BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]
# 财务 DataFrame 标准列，按 date 升序：
FUNDAMENTAL_COLUMNS = ["date", "pe", "pb"]


@dataclass(frozen=True)
class StockInfo:
    code: str          # 6 位代码
    name: str          # 名称（含 ST 前缀时 is_st=True）
    board: Board
    list_date: date    # 上市日期
    is_st: bool


class DataSource(ABC):
    """A股 数据源抽象接口。实现需返回符合上述标准列的 DataFrame。"""

    @abstractmethod
    def list_stocks(self) -> list[StockInfo]:
        """返回全市场股票基础信息。"""

    @abstractmethod
    def get_daily_bars(self, code: str, start: date, end: date) -> pd.DataFrame:
        """前复权日线，列为 BAR_COLUMNS，按 date 升序；无数据返回空 DataFrame。"""

    @abstractmethod
    def get_fundamentals(self, code: str, start: date, end: date) -> pd.DataFrame:
        """每日估值指标，列为 FUNDAMENTAL_COLUMNS，按 date 升序。"""

    @abstractmethod
    def get_trading_dates(self, start: date, end: date) -> list[date]:
        """[start, end] 区间内的交易日（升序）。"""
```

- [ ] **Step 2: 写 `tests/fakes.py`（内存实现，供全项目测试复用）**

```python
"""测试用内存数据源。"""
from __future__ import annotations

from datetime import date

import pandas as pd

from stockbar.datafeed.source import (
    BAR_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    DataSource,
    StockInfo,
)


class FakeDataSource(DataSource):
    def __init__(
        self,
        stocks: list[StockInfo] | None = None,
        bars: dict[str, pd.DataFrame] | None = None,
        fundamentals: dict[str, pd.DataFrame] | None = None,
        trading_dates: list[date] | None = None,
    ):
        self._stocks = stocks or []
        self._bars = bars or {}
        self._fundamentals = fundamentals or {}
        self._trading_dates = sorted(trading_dates or [])

    def list_stocks(self) -> list[StockInfo]:
        return list(self._stocks)

    def get_daily_bars(self, code, start, end) -> pd.DataFrame:
        df = self._bars.get(code, pd.DataFrame(columns=BAR_COLUMNS))
        if df.empty:
            return df.copy()
        m = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[m].reset_index(drop=True)

    def get_fundamentals(self, code, start, end) -> pd.DataFrame:
        df = self._fundamentals.get(code, pd.DataFrame(columns=FUNDAMENTAL_COLUMNS))
        if df.empty:
            return df.copy()
        m = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[m].reset_index(drop=True)

    def get_trading_dates(self, start, end) -> list[date]:
        return [d for d in self._trading_dates if start <= d <= end]
```

- [ ] **Step 3: 写测试验证 Fake 行为**

`tests/datafeed/test_fakes.py`：
```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/datafeed/test_fakes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stockbar/datafeed/source.py tests/fakes.py tests/datafeed/test_fakes.py
git commit -m "feat: DataSource 抽象接口与 FakeDataSource"
```

---

## Task 3：交易日历 `calendar.py`

**Files:**
- Create: `stockbar/datafeed/calendar.py`
- Test: `tests/datafeed/test_calendar.py`

- [ ] **Step 1: 写失败测试**

`tests/datafeed/test_calendar.py`：
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/datafeed/test_calendar.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `calendar.py`**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/datafeed/test_calendar.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stockbar/datafeed/calendar.py tests/datafeed/test_calendar.py
git commit -m "feat: 交易日历 TradingCalendar"
```

---

## Task 4：本地缓存 `store.py`

**Files:**
- Create: `stockbar/datafeed/store.py`
- Test: `tests/datafeed/test_store.py`

`LocalStore` 把行情/财务存为 Parquet（每只一文件），股票列表与交易日历存 SQLite。支持增量 append 与"最后缓存日期"查询。

- [ ] **Step 1: 写失败测试**

`tests/datafeed/test_store.py`：
```python
from datetime import date

import pandas as pd

from stockbar.datafeed.instruments import Board
from stockbar.datafeed.source import StockInfo
from stockbar.datafeed.store import LocalStore


def _bars(dates, closes):
    return pd.DataFrame({
        "date": dates,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [100] * len(dates), "amount": [1000.0] * len(dates),
    })


def test_save_and_load_bars(tmp_path):
    store = LocalStore(tmp_path)
    df = _bars([date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5])
    store.save_bars("600000", df)
    out = store.load_bars("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(out["close"]) == [10.0, 10.5]
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 3)]


def test_append_bars_dedupes_and_sorts(tmp_path):
    store = LocalStore(tmp_path)
    store.save_bars("600000", _bars([date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5]))
    # 追加：与 1/3 重叠 + 新增 1/4，重叠以新数据为准
    store.append_bars("600000", _bars([date(2024, 1, 3), date(2024, 1, 4)], [99.0, 11.0]))
    out = store.load_bars("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    assert list(out["close"]) == [10.0, 99.0, 11.0]


def test_last_bar_date(tmp_path):
    store = LocalStore(tmp_path)
    assert store.last_bar_date("600000") is None
    store.save_bars("600000", _bars([date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5]))
    assert store.last_bar_date("600000") == date(2024, 1, 3)


def test_load_bars_missing_returns_empty(tmp_path):
    store = LocalStore(tmp_path)
    out = store.load_bars("000001", date(2024, 1, 1), date(2024, 1, 31))
    assert out.empty


def test_save_and_load_stocks(tmp_path):
    store = LocalStore(tmp_path)
    stocks = [
        StockInfo("600000", "浦发银行", Board.MAIN, date(1999, 11, 10), False),
        StockInfo("688981", "中芯国际", Board.STAR, date(2020, 7, 16), False),
    ]
    store.save_stocks(stocks)
    loaded = store.load_stocks()
    assert {s.code for s in loaded} == {"600000", "688981"}
    star = next(s for s in loaded if s.code == "688981")
    assert star.board == Board.STAR and star.list_date == date(2020, 7, 16)


def test_save_and_load_calendar(tmp_path):
    store = LocalStore(tmp_path)
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    store.save_calendar(dates)
    assert store.load_calendar() == dates
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/datafeed/test_store.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `store.py`**

```python
"""本地缓存：行情/财务用 Parquet，股票列表与交易日历用 SQLite。"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from stockbar.datafeed.instruments import Board
from stockbar.datafeed.source import BAR_COLUMNS, StockInfo


def _to_date(v) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    return pd.Timestamp(v).date()


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.bars_dir = self.root / "bars"
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "meta.sqlite"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS stocks ("
                "code TEXT PRIMARY KEY, name TEXT, board TEXT, "
                "list_date TEXT, is_st INTEGER)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS calendar (d TEXT PRIMARY KEY)"
            )

    # ---- 行情 ----
    def _bars_path(self, code: str) -> Path:
        return self.bars_dir / f"{code}.parquet"

    def save_bars(self, code: str, df: pd.DataFrame) -> None:
        out = df[BAR_COLUMNS].copy()
        out["date"] = out["date"].map(_to_date)
        out = out.sort_values("date").reset_index(drop=True)
        out.to_parquet(self._bars_path(code), index=False)

    def append_bars(self, code: str, df: pd.DataFrame) -> None:
        existing = self._read_all_bars(code)
        combined = pd.concat([existing, df[BAR_COLUMNS]], ignore_index=True)
        combined["date"] = combined["date"].map(_to_date)
        combined = (
            combined.drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        combined.to_parquet(self._bars_path(code), index=False)

    def _read_all_bars(self, code: str) -> pd.DataFrame:
        path = self._bars_path(code)
        if not path.exists():
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = pd.read_parquet(path)
        df["date"] = df["date"].map(_to_date)
        return df

    def load_bars(self, code: str, start: date, end: date) -> pd.DataFrame:
        df = self._read_all_bars(code)
        if df.empty:
            return df
        m = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[m].sort_values("date").reset_index(drop=True)

    def last_bar_date(self, code: str) -> date | None:
        df = self._read_all_bars(code)
        if df.empty:
            return None
        return max(df["date"])

    # ---- 股票列表 ----
    def save_stocks(self, stocks: list[StockInfo]) -> None:
        rows = [
            (s.code, s.name, s.board.value, s.list_date.isoformat(), int(s.is_st))
            for s in stocks
        ]
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM stocks")
            con.executemany(
                "INSERT INTO stocks VALUES (?, ?, ?, ?, ?)", rows
            )

    def load_stocks(self) -> list[StockInfo]:
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("SELECT code, name, board, list_date, is_st FROM stocks")
            return [
                StockInfo(
                    code=r[0], name=r[1], board=Board(r[2]),
                    list_date=date.fromisoformat(r[3]), is_st=bool(r[4]),
                )
                for r in cur.fetchall()
            ]

    # ---- 交易日历 ----
    def save_calendar(self, dates: list[date]) -> None:
        rows = [(d.isoformat(),) for d in sorted(set(dates))]
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM calendar")
            con.executemany("INSERT INTO calendar VALUES (?)", rows)

    def load_calendar(self) -> list[date]:
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("SELECT d FROM calendar ORDER BY d")
            return [date.fromisoformat(r[0]) for r in cur.fetchall()]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/datafeed/test_store.py -q`
Expected: PASS（6 个用例）

- [ ] **Step 5: Commit**

```bash
git add stockbar/datafeed/store.py tests/datafeed/test_store.py
git commit -m "feat: 本地缓存 LocalStore (Parquet 行情 + SQLite 元数据)"
```

---

## Task 5：AKShare 适配器 `akshare_source.py`

**Files:**
- Create: `stockbar/datafeed/akshare_source.py`
- Test: `tests/datafeed/test_akshare_source.py`

AKShare 返回中文列名，需映射到标准列。AKShare 联网且接口偶有变动，故：核心**列映射逻辑**做纯函数单元测试（不联网），整体抓取做 `@pytest.mark.integration` 集成测试（离线可跳过）。

- [ ] **Step 1: 写失败测试（列映射纯函数 + 集成测试）**

`tests/datafeed/test_akshare_source.py`：
```python
from datetime import date

import pandas as pd
import pytest

from stockbar.datafeed.akshare_source import (
    AkshareSource,
    _normalize_bars,
)


def test_normalize_bars_maps_chinese_columns():
    raw = pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03"],
        "开盘": [10.0, 10.5], "最高": [10.6, 11.1],
        "最低": [9.9, 10.4], "收盘": [10.5, 11.0],
        "成交量": [100, 120], "成交额": [1050.0, 1320.0],
        "涨跌幅": [1.0, 4.8],  # 多余列应被丢弃
    })
    out = _normalize_bars(raw)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert out["date"].iloc[0] == date(2024, 1, 2)
    assert out["close"].iloc[1] == 11.0


def test_normalize_bars_empty():
    out = _normalize_bars(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]


@pytest.mark.integration
def test_akshare_get_daily_bars_live():
    src = AkshareSource()
    df = src.get_daily_bars("600000", date(2024, 1, 2), date(2024, 1, 31))
    assert not df.empty
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert df["date"].is_monotonic_increasing


@pytest.mark.integration
def test_akshare_trading_dates_live():
    src = AkshareSource()
    dates = src.get_trading_dates(date(2024, 1, 1), date(2024, 1, 31))
    assert date(2024, 1, 2) in dates
    assert date(2024, 1, 6) not in dates  # 周六
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/datafeed/test_akshare_source.py -q -m "not integration"`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `akshare_source.py`**

```python
"""AKShare 数据源适配器。

注意：AKShare 接口列名/函数可能随版本变动。集成测试失败时优先核对
akshare 文档的返回列名，再调整 _normalize_* 映射。
"""
from __future__ import annotations

from datetime import date

import akshare as ak
import pandas as pd

from stockbar.datafeed.instruments import Board, classify_board
from stockbar.datafeed.source import (
    BAR_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    DataSource,
    StockInfo,
)

_BAR_MAP = {
    "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
    "收盘": "close", "成交量": "volume", "成交额": "amount",
}


def _normalize_bars(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = raw.rename(columns=_BAR_MAP)
    df = df[[c for c in BAR_COLUMNS if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


class AkshareSource(DataSource):
    def list_stocks(self) -> list[StockInfo]:
        # 代码+名称
        names = ak.stock_info_a_code_name()  # 列: code, name
        out: list[StockInfo] = []
        for _, row in names.iterrows():
            code = str(row["code"]).zfill(6)
            try:
                board = classify_board(code)
            except ValueError:
                continue
            name = str(row["name"])
            is_st = "ST" in name.upper()
            # 上市日期：逐只查较慢，这里给占位最早日期，update_data 脚本可后补。
            out.append(StockInfo(code, name, board, date(1990, 12, 19), is_st))
        return out

    def get_daily_bars(self, code: str, start: date, end: date) -> pd.DataFrame:
        raw = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
        return _normalize_bars(raw)

    def get_fundamentals(self, code: str, start: date, end: date) -> pd.DataFrame:
        raw = ak.stock_a_indicator_lg(symbol=code)  # 列含 trade_date, pe, pb
        if raw is None or raw.empty:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        df = raw.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        keep = [c for c in FUNDAMENTAL_COLUMNS if c in df.columns]
        return df[keep].sort_values("date").reset_index(drop=True)

    def get_trading_dates(self, start: date, end: date) -> list[date]:
        cal = ak.tool_trade_date_hist_sina()  # 列: trade_date
        s = pd.to_datetime(cal["trade_date"]).dt.date
        return sorted(d for d in s if start <= d <= end)
```

- [ ] **Step 4: 运行单元测试（不联网）确认通过**

Run: `pytest tests/datafeed/test_akshare_source.py -q -m "not integration"`
Expected: PASS（`_normalize_bars` 两个用例）

- [ ] **Step 5: 运行集成测试（联网）**

Run: `pytest tests/datafeed/test_akshare_source.py -q -m integration`
Expected: PASS（若 akshare 列名有变则按错误信息调 `_BAR_MAP` / 列名映射）

- [ ] **Step 6: Commit**

```bash
git add stockbar/datafeed/akshare_source.py tests/datafeed/test_akshare_source.py
git commit -m "feat: AKShare 数据源适配器"
```

---

## Task 6：增量更新脚本 `scripts/update_data.py`

**Files:**
- Create: `scripts/update_data.py`
- Test: `tests/datafeed/test_update_data.py`

把"更新一只股票行情到缓存"的核心逻辑做成可测纯函数 `update_symbol`，脚本 `main` 只做编排。

- [ ] **Step 1: 写失败测试**

`tests/datafeed/test_update_data.py`：
```python
from datetime import date

import pandas as pd

from scripts.update_data import update_symbol
from stockbar.datafeed.store import LocalStore
from tests.fakes import FakeDataSource


def _bars(dates, closes):
    return pd.DataFrame({
        "date": dates, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [100] * len(dates), "amount": [1000.0] * len(dates),
    })


def test_update_symbol_fetches_full_when_empty(tmp_path):
    src = FakeDataSource(bars={"600000": _bars(
        [date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5])})
    store = LocalStore(tmp_path)
    n = update_symbol(src, store, "600000", until=date(2024, 1, 3))
    assert n == 2
    assert store.last_bar_date("600000") == date(2024, 1, 3)


def test_update_symbol_incremental_from_last(tmp_path):
    store = LocalStore(tmp_path)
    store.save_bars("600000", _bars([date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5]))
    # 源里有 1/2~1/5；增量应只抓 1/4、1/5（>last=1/3）
    src = FakeDataSource(bars={"600000": _bars(
        [date(2024, 1, d) for d in (2, 3, 4, 5)], [10.0, 10.5, 11.0, 11.5])})
    n = update_symbol(src, store, "600000", until=date(2024, 1, 5))
    assert n == 2
    assert store.last_bar_date("600000") == date(2024, 1, 5)
    full = store.load_bars("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(full["date"]) == [date(2024, 1, d) for d in (2, 3, 4, 5)]


def test_update_symbol_noop_when_up_to_date(tmp_path):
    store = LocalStore(tmp_path)
    store.save_bars("600000", _bars([date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5]))
    src = FakeDataSource(bars={"600000": _bars(
        [date(2024, 1, 2), date(2024, 1, 3)], [10.0, 10.5])})
    n = update_symbol(src, store, "600000", until=date(2024, 1, 3))
    assert n == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/datafeed/test_update_data.py -q`
Expected: FAIL（`ModuleNotFoundError: scripts.update_data`）

- [ ] **Step 3: 实现 `scripts/update_data.py`**

需要 `scripts/__init__.py`（空文件，docstring `"""scripts."""`）以便测试导入。

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/datafeed/test_update_data.py -q`
Expected: PASS（3 个用例）

- [ ] **Step 5: 全量单元测试回归**

Run: `pytest -q -m "not integration"`
Expected: PASS（所有非集成用例）

- [ ] **Step 6: Commit**

```bash
git add scripts/ tests/datafeed/test_update_data.py
git commit -m "feat: 增量更新脚本 update_data"
```

---

## Phase 1 验收标准

- [ ] `pytest -q -m "not integration"` 全绿
- [ ] `pytest -q -m integration` 在联网时全绿（验证 AKShare 真实可用）
- [ ] 手动冒烟：`python scripts/update_data.py --root data --until <近一周某日>`，确认 `data/bars/` 生成 parquet、`data/meta.sqlite` 有 stocks/calendar 记录
- [ ] 后续 Phase 2 能通过 `LocalStore` 离线读到行情/股票/日历

---

## 自检记录

- **Spec 覆盖**：对应设计文档第 3 节（数据源/缓存/复权/交易日历）；板块分类提前到本阶段（后续撮合规则依赖）。第 4 节起的选股池/因子/撮合/回测/runner 属 Phase 2–5。
- **类型一致性**：`StockInfo`、`Board`、`BAR_COLUMNS`/`FUNDAMENTAL_COLUMNS` 全程统一；`DataSource` 抽象方法签名与 `FakeDataSource`/`AkshareSource` 实现一致；`update_symbol(src, store, code, until)` 签名在测试与实现一致。
- **占位符扫描**：无 TBD/TODO；每步含可运行代码与命令。`list_stocks` 的 `list_date` 占位最早日期一点是已知简化（AKShare 逐只查上市日太慢），已在注释标注，Phase 2 若需精确上市日再补。
