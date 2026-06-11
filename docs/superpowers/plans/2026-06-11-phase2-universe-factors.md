# Phase 2：选股池 + 因子 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Phase 1 数据地基之上，实现「每日选股」：构建两层选股池（基础可交易过滤 + 按市场状态拆成左/右两个独立池子），对左右池各用本侧因子打分排名，输出左侧榜与右侧榜候选清单。

**Architecture:** 纯函数为主、数据访问在边缘。技术指标（`indicators.py`）作用于单只股票的价量序列；`features.py` 把一个截面日的所有候选股算成一张「特征表」（index=code）；`universe.py` 在特征表上做两层过滤拆池；`factors/` 在池内做截面标准化+合成+排名；`selection.py` 把上述串起来并对接 Phase 1 的 `LocalStore`。所有逻辑用合成数据 TDD，无需联网。

**Tech Stack:** Python 3.11 · pandas · numpy · pytest（沿用 Phase 1 环境 `.venv`）

设计文档：`docs/superpowers/specs/2026-06-10-ashare-dual-line-shadow-design.md`（第 4 选股池、第 5 因子库、第 6 每日选股流程、第 7.1 买入条件的池内排名部分）。

依赖 Phase 1 已有：`stockbar/datafeed/{instruments(Board),source(StockInfo,BAR_COLUMNS,FUNDAMENTAL_COLUMNS,DataSource),store(LocalStore),calendar,akshare_source}`、`tests/fakes.FakeDataSource`。

---

## 约定与默认参数

- **标准化**：去极值用分位裁剪（默认下 1% / 上 99%），再做截面 z-score（`(x-mean)/std`，std=0 时返回 0）。
- **因子方向**：每个因子配 `direction ∈ {+1,-1}`。`+1` = 原值越大越好；`-1` = 原值越小越好。合成分 = `Σ direction_i * zscore(winsorize(value_i)) * weight_i`，默认 `weight=1`。
- **RSI**：周期 14，简单均值法；全涨→100，全跌→0，数据不足→NaN。
- **"上市≥60交易日"** 用「缓存K线条数 ≥ 60」代理。
- **停牌**：as_of 当日无K线（最新K线日期 < as_of）。
- **特征表列**（`features.py` 产出，index=code）：
  `board, is_st, bars_count, suspended, close, ma5, ma10, ma20, ma60, rsi, bias20, bias60, ret20, ret60, new_high60, ma_bullish, vol_price_up, pb, pe, avg_amount20`

## 文件结构

| 文件 | 职责 |
|---|---|
| `stockbar/datafeed/store.py`（改） | 增加 `save_fundamentals/append_fundamentals/load_fundamentals/last_fundamental_date` |
| `scripts/update_data.py`（改） | 增量更新里补抓 PB/PE 财务 |
| `stockbar/indicators.py` | 价量技术指标纯函数 |
| `stockbar/features.py` | `compute_features(...)` 构建截面特征表 |
| `stockbar/universe.py` | `build_tradable`（第一层）+ `split_pools`（第二层） |
| `stockbar/factors/__init__.py` | 子包入口 |
| `stockbar/factors/base.py` | `winsorize/zscore/standardize/FactorSpec/score_pool/select_top_n` |
| `stockbar/factors/left.py` | 左侧因子规格 `LEFT_FACTORS` + `score_left` |
| `stockbar/factors/right.py` | 右侧因子规格 `RIGHT_FACTORS` + `score_right` |
| `stockbar/selection.py` | `build_candidate_lists(store, stocks, as_of, ...)` 整合 |
| `tests/...` | 各模块单测 |

---

## Task 1：store 增加财务缓存

**Files:**
- Modify: `stockbar/datafeed/store.py`
- Test: `tests/datafeed/test_store_fundamentals.py`

财务（PB/PE）按 Parquet 每只一文件存于 `funds/{code}.parquet`，接口与行情对称。

- [ ] **Step 1: 写失败测试** — `tests/datafeed/test_store_fundamentals.py`:

```python
from datetime import date

import pandas as pd

from stockbar.datafeed.store import LocalStore


def _funds(dates, pbs):
    return pd.DataFrame({"date": dates, "pe": [float(i) for i in range(len(dates))], "pb": pbs})


def test_save_and_load_fundamentals(tmp_path):
    store = LocalStore(tmp_path)
    df = _funds([date(2024, 1, 2), date(2024, 1, 3)], [1.0, 1.1])
    store.save_fundamentals("600000", df)
    out = store.load_fundamentals("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(out["pb"]) == [1.0, 1.1]
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 3)]


def test_append_fundamentals_dedupes(tmp_path):
    store = LocalStore(tmp_path)
    store.save_fundamentals("600000", _funds([date(2024, 1, 2), date(2024, 1, 3)], [1.0, 1.1]))
    store.append_fundamentals("600000", _funds([date(2024, 1, 3), date(2024, 1, 4)], [9.9, 1.2]))
    out = store.load_fundamentals("600000", date(2024, 1, 1), date(2024, 1, 31))
    assert list(out["date"]) == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    assert list(out["pb"]) == [1.0, 9.9, 1.2]


def test_last_fundamental_date(tmp_path):
    store = LocalStore(tmp_path)
    assert store.last_fundamental_date("600000") is None
    store.save_fundamentals("600000", _funds([date(2024, 1, 2)], [1.0]))
    assert store.last_fundamental_date("600000") == date(2024, 1, 2)


def test_load_fundamentals_missing_returns_empty(tmp_path):
    store = LocalStore(tmp_path)
    out = store.load_fundamentals("000001", date(2024, 1, 1), date(2024, 1, 31))
    assert out.empty
```

- [ ] **Step 2: 运行确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/datafeed/test_store_fundamentals.py -q`
Expected: FAIL（`AttributeError: ... save_fundamentals`）

- [ ] **Step 3: 修改 `store.py`** — 在文件顶部 import 处补 `FUNDAMENTAL_COLUMNS`：

把
```python
from stockbar.datafeed.source import BAR_COLUMNS, StockInfo
```
改为
```python
from stockbar.datafeed.source import BAR_COLUMNS, FUNDAMENTAL_COLUMNS, StockInfo
```

在 `__init__` 里，`self.bars_dir` 之后补一个 funds 目录：
```python
        self.funds_dir = self.root / "funds"
        self.funds_dir.mkdir(parents=True, exist_ok=True)
```

在「# ---- 行情 ----」段落之后、「# ---- 股票列表 ----」之前，插入财务方法（与行情逻辑对称，复用 `_to_date`）：
```python
    # ---- 财务（PB/PE） ----
    def _funds_path(self, code: str) -> Path:
        return self.funds_dir / f"{code}.parquet"

    def save_fundamentals(self, code: str, df: pd.DataFrame) -> None:
        out = df[FUNDAMENTAL_COLUMNS].copy()
        out["date"] = out["date"].map(_to_date)
        out = out[out["date"].map(lambda d: not pd.isna(d))]
        out = out.sort_values("date").reset_index(drop=True)
        out.to_parquet(self._funds_path(code), index=False)

    def append_fundamentals(self, code: str, df: pd.DataFrame) -> None:
        existing = self._read_all_funds(code)
        combined = pd.concat([existing, df[FUNDAMENTAL_COLUMNS]], ignore_index=True)
        combined["date"] = combined["date"].map(_to_date)
        combined = combined[combined["date"].map(lambda d: not pd.isna(d))]
        combined = (
            combined.drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        combined.to_parquet(self._funds_path(code), index=False)

    def _read_all_funds(self, code: str) -> pd.DataFrame:
        path = self._funds_path(code)
        if not path.exists():
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        df = pd.read_parquet(path)
        df["date"] = df["date"].map(_to_date)
        return df

    def load_fundamentals(self, code: str, start: date, end: date) -> pd.DataFrame:
        df = self._read_all_funds(code)
        if df.empty:
            return df
        m = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[m].sort_values("date").reset_index(drop=True)

    def last_fundamental_date(self, code: str) -> date | None:
        df = self._read_all_funds(code)
        if df.empty:
            return None
        return max(df["date"])
```

- [ ] **Step 4: 运行确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/datafeed/test_store_fundamentals.py -q`
Expected: PASS（4 用例）

- [ ] **Step 5: Commit**

```bash
git add stockbar/datafeed/store.py tests/datafeed/test_store_fundamentals.py
git commit -m "feat: LocalStore 增加 PB/PE 财务缓存"
```

---

## Task 2：技术指标 `indicators.py`

**Files:**
- Create: `stockbar/indicators.py`
- Test: `tests/test_indicators.py`

全部作用于 `pd.Series`（按时间升序），返回标量或 Series。

- [ ] **Step 1: 写失败测试** — `tests/test_indicators.py`:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_indicators.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `indicators.py`**

```python
"""价量技术指标（作用于按时间升序的 pd.Series）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0.0, 100.0)   # 全涨：avg_loss=0 → 100
    out = out.where(avg_gain != 0.0, 0.0)      # 全跌：avg_gain=0 → 0
    out[avg_gain.isna()] = np.nan              # 数据不足保持 NaN
    return out


def bias(close: pd.Series, n: int) -> pd.Series:
    """乖离率 (close - ma_n) / ma_n。"""
    ma = sma(close, n)
    return (close - ma) / ma


def ret_n(close: pd.Series, n: int) -> pd.Series:
    """近 n 期收益率 close[t]/close[t-n] - 1。"""
    return close.pct_change(n)


def is_new_high(close: pd.Series, n: int) -> bool:
    """最新收盘是否为近 n 期（含当日）最高。数据不足返回 False。"""
    window = close.tail(n + 1)
    if len(window) < n + 1 or window.isna().any():
        # 至少需要 n+1 个点才能说"近 n 期新高"；不足时用现有窗口判断
        window = close.dropna()
        if window.empty:
            return False
    return bool(window.iloc[-1] >= window.max())


def is_ma_bullish(close: pd.Series) -> bool:
    """均线多头排列 ma5 > ma10 > ma20（取最新值）。任一为 NaN 返回 False。"""
    m5 = sma(close, 5).iloc[-1]
    m10 = sma(close, 10).iloc[-1]
    m20 = sma(close, 20).iloc[-1]
    if any(pd.isna(x) for x in (m5, m10, m20)):
        return False
    return bool(m5 > m10 > m20)


def is_vol_price_up(close: pd.Series, volume: pd.Series, n: int = 5) -> bool:
    """量价齐升：最新收盘较上一日上涨，且最新成交量 > 近 n 日均量。"""
    if len(close) < 2:
        return False
    price_up = bool(close.iloc[-1] > close.iloc[-2])
    vol_ma = sma(volume, n).iloc[-1]
    vol_up = (not pd.isna(vol_ma)) and bool(volume.iloc[-1] > vol_ma)
    return price_up and vol_up
```

- [ ] **Step 4: 运行确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_indicators.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stockbar/indicators.py tests/test_indicators.py
git commit -m "feat: 技术指标 indicators"
```

---

## Task 3：截面特征表 `features.py`

**Files:**
- Create: `stockbar/features.py`
- Test: `tests/test_features.py`

把一个截面日 `as_of` 的所有候选股算成一张特征表（index=code）。输入：`stocks`（StockInfo 列表）、`panel`（{code: 行情DataFrame}）、`fundamentals`（{code: 财务DataFrame}）。

- [ ] **Step 1: 写失败测试** — `tests/test_features.py`:

```python
from datetime import date

import pandas as pd

from stockbar.datafeed.instruments import Board
from stockbar.datafeed.source import StockInfo
from stockbar.features import compute_features


def _bars(dates, closes, vols=None, amts=None):
    n = len(dates)
    vols = vols or [100.0] * n
    amts = amts or [1_000_000.0] * n
    return pd.DataFrame({
        "date": dates, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": vols, "amount": amts,
    })


def _trading_days(n, end=date(2024, 4, 1)):
    # 生成 n 个连续自然日（测试无需真实交易日历）
    return list(pd.date_range(end=pd.Timestamp(end), periods=n).date)


def test_features_basic_columns_and_suspended():
    days = _trading_days(70)
    closes = [10.0 + i * 0.1 for i in range(70)]   # 缓慢上涨
    panel = {"600000": _bars(days, closes)}
    funds = {"600000": pd.DataFrame({"date": [days[-1]], "pe": [15.0], "pb": [1.2]})}
    stocks = [StockInfo("600000", "浦发银行", Board.MAIN, date(1999, 11, 10), False)]
    feat = compute_features(stocks, panel, funds, as_of=days[-1])
    assert "600000" in feat.index
    row = feat.loc["600000"]
    assert row["bars_count"] == 70
    assert row["suspended"] == False
    assert row["is_st"] == False
    assert row["board"] == Board.MAIN
    assert abs(row["pb"] - 1.2) < 1e-9
    assert row["close"] == closes[-1]
    assert row["ma_bullish"] == True   # 单调上涨


def test_features_marks_suspended_when_no_bar_on_as_of():
    days = _trading_days(70)
    panel = {"600000": _bars(days, [10.0] * 70)}
    stocks = [StockInfo("600000", "X", Board.MAIN, date(2000, 1, 1), False)]
    later = days[-1] + pd.Timedelta(days=5)
    feat = compute_features(stocks, panel, {}, as_of=later.date() if hasattr(later, "date") else later)
    assert feat.loc["600000"]["suspended"] == True


def test_features_pb_uses_last_on_or_before_as_of():
    days = _trading_days(70)
    panel = {"600000": _bars(days, [10.0] * 70)}
    funds = {"600000": pd.DataFrame({
        "date": [days[-10], days[-3]], "pe": [10.0, 11.0], "pb": [1.0, 2.0]})}
    stocks = [StockInfo("600000", "X", Board.MAIN, date(2000, 1, 1), False)]
    feat = compute_features(stocks, panel, funds, as_of=days[-1])
    assert feat.loc["600000"]["pb"] == 2.0   # 取 as_of 前最近一条


def test_features_skips_codes_without_bars():
    stocks = [StockInfo("600000", "X", Board.MAIN, date(2000, 1, 1), False)]
    feat = compute_features(stocks, {}, {}, as_of=date(2024, 4, 1))
    assert feat.empty
```

- [ ] **Step 2: 运行确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_features.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `features.py`**

```python
"""把一个截面日的候选股算成特征表（index=code）。"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stockbar.datafeed.source import StockInfo
from stockbar import indicators as ind

FEATURE_COLUMNS = [
    "board", "is_st", "bars_count", "suspended", "close",
    "ma5", "ma10", "ma20", "ma60", "rsi", "bias20", "bias60",
    "ret20", "ret60", "new_high60", "ma_bullish", "vol_price_up",
    "pb", "pe", "avg_amount20",
]


def _last_on_or_before(df: pd.DataFrame, col: str, as_of: date):
    if df is None or df.empty:
        return np.nan
    sub = df[df["date"] <= as_of]
    if sub.empty:
        return np.nan
    return sub.sort_values("date").iloc[-1][col]


def _feature_row(info: StockInfo, bars: pd.DataFrame, funds: pd.DataFrame, as_of: date) -> dict:
    bars = bars[bars["date"] <= as_of].sort_values("date")
    close = bars["close"].reset_index(drop=True)
    volume = bars["volume"].reset_index(drop=True)
    last_bar_date = bars["date"].iloc[-1]
    return {
        "board": info.board,
        "is_st": info.is_st,
        "bars_count": len(bars),
        "suspended": last_bar_date != as_of,
        "close": float(close.iloc[-1]),
        "ma5": float(ind.sma(close, 5).iloc[-1]),
        "ma10": float(ind.sma(close, 10).iloc[-1]),
        "ma20": float(ind.sma(close, 20).iloc[-1]),
        "ma60": float(ind.sma(close, 60).iloc[-1]),
        "rsi": float(ind.rsi(close, 14).iloc[-1]),
        "bias20": float(ind.bias(close, 20).iloc[-1]),
        "bias60": float(ind.bias(close, 60).iloc[-1]),
        "ret20": float(ind.ret_n(close, 20).iloc[-1]),
        "ret60": float(ind.ret_n(close, 60).iloc[-1]),
        "new_high60": ind.is_new_high(close, 60),
        "ma_bullish": ind.is_ma_bullish(close),
        "vol_price_up": ind.is_vol_price_up(close, volume),
        "pb": float(_last_on_or_before(funds, "pb", as_of)),
        "pe": float(_last_on_or_before(funds, "pe", as_of)),
        "avg_amount20": float(bars["amount"].tail(20).mean()),
    }


def compute_features(
    stocks: list[StockInfo],
    panel: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    as_of: date,
) -> pd.DataFrame:
    rows = {}
    for info in stocks:
        bars = panel.get(info.code)
        if bars is None or bars.empty:
            continue
        if bars[bars["date"] <= as_of].empty:
            continue
        rows[info.code] = _feature_row(info, bars, fundamentals.get(info.code), as_of)
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.DataFrame.from_dict(rows, orient="index")[FEATURE_COLUMNS]
```

- [ ] **Step 4: 运行确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_features.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stockbar/features.py tests/test_features.py
git commit -m "feat: 截面特征表 compute_features"
```

---

## Task 4：选股池 `universe.py`

**Files:**
- Create: `stockbar/universe.py`
- Test: `tests/test_universe.py`

在特征表上做两层：`build_tradable`（第一层基础可交易过滤）+ `split_pools`（第二层按市场状态拆左/右池）。

- [ ] **Step 1: 写失败测试** — `tests/test_universe.py`:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_universe.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `universe.py`**

```python
"""选股池：第一层基础可交易过滤 + 第二层按市场状态拆左/右池。"""
from __future__ import annotations

import pandas as pd

from stockbar.datafeed.instruments import Board

# 左池：估值后 1/3
LEFT_PB_QUANTILE = 1.0 / 3.0
# 左池：近 20 日跌幅阈值
LEFT_RET20_MAX = -0.08


def build_tradable(features: pd.DataFrame, min_amount: float = 1e7, min_bars: int = 60) -> pd.DataFrame:
    """第一层：剔除 ST、北交所、停牌、上市不足、流动性过低。返回过滤后的特征表。"""
    if features.empty:
        return features
    m = (
        (~features["is_st"])
        & (features["board"] != Board.BSE)
        & (~features["suspended"])
        & (features["bars_count"] >= min_bars)
        & (features["avg_amount20"] >= min_amount)
    )
    return features[m]


def split_pools(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """第二层：返回 (左池 codes, 右池 codes)。输入应为已过可交易过滤的特征表。"""
    if features.empty:
        return [], []

    # 左池：弱势 + 超跌 + 低估（pb 后 1/3）
    weak = (features["close"] < features["ma60"]) & (features["ret20"] <= LEFT_RET20_MAX)
    if weak.any():
        pb_threshold = features.loc[weak, "pb"].quantile(LEFT_PB_QUANTILE)
        left_mask = weak & (features["pb"] <= pb_threshold)
    else:
        left_mask = weak

    # 右池：强势 + 趋势
    right_mask = (
        (features["close"] > features["ma20"])
        & (features["ma_bullish"])
        & (features["ret20"] > 0)
    )

    return list(features[left_mask].index), list(features[right_mask].index)
```

- [ ] **Step 4: 运行确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_universe.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stockbar/universe.py tests/test_universe.py
git commit -m "feat: 选股池 build_tradable + split_pools"
```

---

## Task 5：因子框架 `factors/base.py`

**Files:**
- Create: `stockbar/factors/__init__.py`（docstring `"""factors."""`）
- Create: `stockbar/factors/base.py`
- Test: `tests/factors/__init__.py`（docstring）、`tests/factors/test_base.py`

- [ ] **Step 1: 写失败测试** — `tests/factors/test_base.py`:

```python
import numpy as np
import pandas as pd

from stockbar.factors.base import (
    FactorSpec, winsorize, zscore, standardize, score_pool, select_top_n,
)


def test_winsorize_clips_tails():
    s = pd.Series([1.0, 2, 3, 4, 100])
    out = winsorize(s, lower=0.0, upper=0.5)   # upper 50% 分位=3
    assert out.max() == 3.0


def test_zscore_mean0_std1():
    s = pd.Series([1.0, 2, 3, 4, 5])
    z = zscore(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


def test_zscore_constant_returns_zeros():
    s = pd.Series([5.0, 5, 5])
    assert (zscore(s) == 0.0).all()


def test_standardize_winsorizes_then_zscores():
    s = pd.Series([1.0, 2, 3, 4, 1000])
    out = standardize(s)
    assert not out.isna().any()
    assert out.idxmax() == 4   # 极端值裁剪后仍最大


def test_score_pool_direction_and_rank():
    feat = pd.DataFrame({
        "pb":  [1.0, 2.0, 3.0],     # 越小越好 → -1
        "mom": [0.1, 0.2, 0.3],     # 越大越好 → +1
    }, index=["A", "B", "C"])
    specs = [FactorSpec("pb", -1.0), FactorSpec("mom", +1.0)]
    score = score_pool(feat, ["A", "B", "C"], specs)
    # A: 低pb(好) + 低mom(差); C: 高pb(差) + 高mom(好) → 对称，B 居中
    assert set(score.index) == {"A", "B", "C"}
    assert abs(score["B"]) < 1e-9


def test_score_pool_subsets_to_codes():
    feat = pd.DataFrame({"mom": [0.1, 0.2, 0.3]}, index=["A", "B", "C"])
    score = score_pool(feat, ["A", "B"], [FactorSpec("mom", 1.0)])
    assert set(score.index) == {"A", "B"}


def test_select_top_n():
    score = pd.Series({"A": 3.0, "B": 1.0, "C": 2.0})
    assert select_top_n(score, 2) == ["A", "C"]


def test_score_pool_empty():
    feat = pd.DataFrame({"mom": [0.1]}, index=["A"])
    score = score_pool(feat, [], [FactorSpec("mom", 1.0)])
    assert score.empty
```

- [ ] **Step 2: 运行确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/factors/test_base.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `factors/base.py`**（先建 `stockbar/factors/__init__.py`、`tests/factors/__init__.py`）

```python
"""因子框架：标准化、合成打分、选前 N。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    column: str        # 特征表中的列名
    direction: float   # +1 越大越好；-1 越小越好
    weight: float = 1.0


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def standardize(s: pd.Series) -> pd.Series:
    return zscore(winsorize(s))


def score_pool(features: pd.DataFrame, codes: list[str], specs: list[FactorSpec]) -> pd.Series:
    """对 codes 子集，按 specs 截面标准化+合成，返回合成分（index=code）。"""
    codes = [c for c in codes if c in features.index]
    if not codes:
        return pd.Series(dtype="float64")
    sub = features.loc[codes]
    total = pd.Series(0.0, index=sub.index)
    for spec in specs:
        col = sub[spec.column].astype("float64")
        total = total + spec.direction * spec.weight * standardize(col)
    return total


def select_top_n(score: pd.Series, n: int) -> list[str]:
    """合成分降序取前 n 的 codes。"""
    return list(score.sort_values(ascending=False).head(n).index)
```

- [ ] **Step 4: 运行确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/factors/test_base.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stockbar/factors/__init__.py stockbar/factors/base.py tests/factors/
git commit -m "feat: 因子框架 base (标准化/合成/选top)"
```

---

## Task 6：左右因子 `factors/left.py` + `factors/right.py`

**Files:**
- Create: `stockbar/factors/left.py`
- Create: `stockbar/factors/right.py`
- Test: `tests/factors/test_left_right.py`

- [ ] **Step 1: 写失败测试** — `tests/factors/test_left_right.py`:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/factors/test_left_right.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `factors/left.py`**

```python
"""左侧（逆势/价值/均值回归）因子：越便宜、越超跌越好。"""
from __future__ import annotations

import pandas as pd

from stockbar.factors.base import FactorSpec, score_pool

LEFT_FACTORS = [
    FactorSpec("pb", -1.0),       # 低估值
    FactorSpec("pe", -1.0),       # 低估值
    FactorSpec("bias20", -1.0),   # 距 MA20 负乖离越大（越超跌）越好
    FactorSpec("rsi", -1.0),      # RSI 越低（越超卖）越好
    FactorSpec("ret20", -1.0),    # 近 20 日跌幅越大越好
]


def score_left(features: pd.DataFrame, codes: list[str]) -> pd.Series:
    return score_pool(features, codes, LEFT_FACTORS)
```

实现 `factors/right.py`:

```python
"""右侧（顺势/动量/趋势）因子：越强势、越趋势越好。"""
from __future__ import annotations

import pandas as pd

from stockbar.factors.base import FactorSpec, score_pool

RIGHT_FACTORS = [
    FactorSpec("ret20", 1.0),         # 20 日动量
    FactorSpec("ret60", 1.0),         # 60 日动量
    FactorSpec("new_high60", 1.0),    # 创 60 日新高
    FactorSpec("ma_bullish", 1.0),    # 均线多头排列
    FactorSpec("vol_price_up", 1.0),  # 量价齐升
]


def score_right(features: pd.DataFrame, codes: list[str]) -> pd.Series:
    return score_pool(features, codes, RIGHT_FACTORS)
```

> 注：`score_pool` 内部对每列 `.astype("float64")`，bool 列（new_high60/ma_bullish/vol_price_up）会被转成 1.0/0.0 参与标准化，符合预期。

- [ ] **Step 4: 运行确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/factors/test_left_right.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stockbar/factors/left.py stockbar/factors/right.py tests/factors/test_left_right.py
git commit -m "feat: 左右因子 left/right 规格与打分"
```

---

## Task 7：每日选股整合 `selection.py`

**Files:**
- Create: `stockbar/selection.py`
- Test: `tests/test_selection.py`

把数据加载（从 `LocalStore`）→ 特征表 → 两层选股池 → 左右打分排名串起来，产出左右榜候选清单。

- [ ] **Step 1: 写失败测试** — `tests/test_selection.py`:

```python
from datetime import date, timedelta

import pandas as pd

from stockbar.datafeed.instruments import Board
from stockbar.datafeed.source import StockInfo
from stockbar.datafeed.store import LocalStore
from stockbar.selection import build_candidate_lists


def _bars(dates, closes, amt=5e7):
    n = len(dates)
    return pd.DataFrame({
        "date": dates, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000.0] * n, "amount": [amt] * n,
    })


def test_build_candidate_lists_separates_left_right(tmp_path):
    store = LocalStore(tmp_path)
    days = list(pd.date_range(end=pd.Timestamp("2024-04-01"), periods=70).date)
    as_of = days[-1]

    # 下跌股（左池候选）：持续下跌，低 pb
    down = [20.0 - i * 0.15 for i in range(70)]
    store.save_bars("600001", _bars(days, down))
    store.save_fundamentals("600001", pd.DataFrame({"date": [as_of], "pe": [5.0], "pb": [0.8]}))

    # 上涨股（右池候选）：持续上涨
    up = [10.0 + i * 0.2 for i in range(70)]
    store.save_bars("600002", _bars(days, up))
    store.save_fundamentals("600002", pd.DataFrame({"date": [as_of], "pe": [40.0], "pb": [6.0]}))

    stocks = [
        StockInfo("600001", "跌", Board.MAIN, date(2000, 1, 1), False),
        StockInfo("600002", "涨", Board.MAIN, date(2000, 1, 1), False),
    ]
    result = build_candidate_lists(store, stocks, as_of, top_n=5, lookback=120)

    assert "600001" in result.left
    assert "600002" in result.right
    assert "600001" not in result.right
    assert "600002" not in result.left


def test_build_candidate_lists_excludes_st(tmp_path):
    store = LocalStore(tmp_path)
    days = list(pd.date_range(end=pd.Timestamp("2024-04-01"), periods=70).date)
    as_of = days[-1]
    down = [20.0 - i * 0.15 for i in range(70)]
    store.save_bars("600001", _bars(days, down))
    store.save_fundamentals("600001", pd.DataFrame({"date": [as_of], "pe": [5.0], "pb": [0.8]}))
    stocks = [StockInfo("600001", "*ST跌", Board.MAIN, date(2000, 1, 1), True)]
    result = build_candidate_lists(store, stocks, as_of, top_n=5, lookback=120)
    assert "600001" not in result.left and "600001" not in result.right
```

- [ ] **Step 2: 运行确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_selection.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `selection.py`**

```python
"""每日选股整合：数据 → 特征 → 选股池 → 左右打分排名。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from stockbar.datafeed.source import StockInfo
from stockbar.datafeed.store import LocalStore
from stockbar.features import compute_features
from stockbar.universe import build_tradable, split_pools
from stockbar.factors.left import score_left
from stockbar.factors.right import score_right
from stockbar.factors.base import select_top_n


@dataclass(frozen=True)
class CandidateLists:
    as_of: date
    left: list[str]    # 左侧榜前 N
    right: list[str]   # 右侧榜前 N


def build_candidate_lists(
    store: LocalStore,
    stocks: list[StockInfo],
    as_of: date,
    top_n: int = 5,
    lookback: int = 120,
    min_amount: float = 1e7,
    min_bars: int = 60,
) -> CandidateLists:
    """从缓存读取近 lookback 自然日数据，输出左右榜前 top_n 候选。"""
    start = as_of - timedelta(days=lookback)
    panel: dict[str, pd.DataFrame] = {}
    funds: dict[str, pd.DataFrame] = {}
    for info in stocks:
        bars = store.load_bars(info.code, start, as_of)
        if not bars.empty:
            panel[info.code] = bars
            funds[info.code] = store.load_fundamentals(info.code, start, as_of)

    features = compute_features(stocks, panel, funds, as_of)
    tradable = build_tradable(features, min_amount=min_amount, min_bars=min_bars)
    left_codes, right_codes = split_pools(tradable)

    left_score = score_left(tradable, left_codes)
    right_score = score_right(tradable, right_codes)

    return CandidateLists(
        as_of=as_of,
        left=select_top_n(left_score, top_n),
        right=select_top_n(right_score, top_n),
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_selection.py -q`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `.\.venv\Scripts\python.exe -m pytest -q -m "not integration"`
Expected: PASS（Phase 1 + Phase 2 全部单测）

- [ ] **Step 6: Commit**

```bash
git add stockbar/selection.py tests/test_selection.py
git commit -m "feat: 每日选股整合 build_candidate_lists"
```

---

## Phase 2 验收标准

- [ ] `.\.venv\Scripts\python.exe -m pytest -q -m "not integration"` 全绿（Phase 1 + Phase 2）
- [ ] `build_candidate_lists` 能从 `LocalStore` 缓存出发，对一组股票在指定交易日产出左/右两个互斥的候选清单
- [ ] 左池股票（弱势/超跌/低估）不出现在右榜，右池股票（强势/趋势）不出现在左榜
- [ ] ST / 北交所 / 停牌 / 上市不足 / 流动性过低 的股票被正确剔除

---

## 自检记录

- **Spec 覆盖**：第 4 节两层选股池 → `universe.py`；第 5 节左右因子库 → `indicators.py`+`features.py`+`factors/`；第 6 节每日选股流程步骤 1-3（出左右榜）→ `selection.py`。步骤 4-6（结合账户状态决定买/加/卖、算股数、出操作清单）属 Phase 3/4/5，本阶段不做。
- **数据缺口**：Phase 1 未缓存财务，Task 1 在 `LocalStore` 补 PB/PE 缓存，`selection` 才能读到 pb/pe；`scripts/update_data.py` 的财务抓取留到需要真实跑批时补（本阶段以单测为准，不强制改脚本）。
- **类型一致性**：`FactorSpec(column,direction,weight)`、`compute_features` 产出的 `FEATURE_COLUMNS` 与 `LEFT/RIGHT_FACTORS` 引用列名一致（pb/pe/bias20/rsi/ret20/ret60/new_high60/ma_bullish/vol_price_up）；`build_candidate_lists` 返回 `CandidateLists(as_of,left,right)`。
- **占位符扫描**：无 TBD/TODO；每步含完整代码与命令。
- **已知简化**：`update_data.py` 脚本本阶段不强制补财务抓取（store 接口已就绪）；`is_new_high` 在数据少于 n+1 时退化为现有窗口判断；估值后 1/3 用 `quantile(1/3)`，小样本下为近似。
