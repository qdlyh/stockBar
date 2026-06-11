"""因子 Rank IC 验证 — 纯本地，读 Parquet 缓存。

用法: python scripts/validate_ic.py [--root data] [--horizon 10]
"""
from __future__ import annotations

import math, sys, argparse
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np

# bro: add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stockbar.datafeed.store import LocalStore


# ---------- 指标 ----------
def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _rsi(close: pd.Series, n=14):
    d = close.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.rolling(n).mean(); al = l.rolling(n).mean()
    out = 100 - 100/(1 + ag/al)
    out = out.where(al != 0, 100.0); out = out.where(ag != 0, 0.0)
    out[(ag == 0) & (al == 0)] = 50.0; out[ag.isna()] = np.nan
    return out

def _bias(close, n):
    ma = _sma(close, n); return (close - ma) / ma

def _ret_n(close, n):
    return close.pct_change(n)


def _factor_values(df: pd.DataFrame, as_of: date) -> dict[str, float]:
    """单股截至 as_of 的因子值。df 已是 <= as_of 的子集。"""
    if len(df) < 63:
        return {}
    # 停牌: 最新K线日 ≠ as_of
    last_date = df["date"].iloc[-1]
    if hasattr(last_date, "date"):
        last_date = last_date.date()
    if last_date != as_of:
        return {}
    close = df["close"].reset_index(drop=True)
    return {
        "bias20": float(_bias(close, 20).iloc[-1]),
        "rsi":    float(_rsi(close, 14).iloc[-1]),
        "ret20":  float(_ret_n(close, 20).iloc[-1]),
        "ret60":  float(_ret_n(close, 60).iloc[-1]),
    }


# ---------- IC ----------
def _rank_ic(xs: list[float], ys: list[float]) -> float:
    """Spearman = Pearson on ranks. 并列取平均排名。"""
    n = len(xs)
    if n < 10:
        return float("nan")

    def _rk(arr):
        order = sorted(range(n), key=lambda k: arr[k])
        res = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                res[order[k]] = avg
            i = j + 1
        return res

    rx, ry = _rk(xs), _rk(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    sx = sum((x - mx) ** 2 for x in rx)
    sy = sum((y - my) ** 2 for y in ry)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(sx * sy)


def _summary(ics: list[float], label: str) -> str:
    valid = [x for x in ics if not math.isnan(x)]
    if len(valid) < 20:
        return f"  {label:12s} 样本不足 n={len(valid)}"
    m = sum(valid) / len(valid)
    v = sum((x - m) ** 2 for x in valid) / len(valid)
    sd = math.sqrt(v) if v > 0 else 0.0
    ir = m / sd if sd > 0 else 0.0
    wr = sum(1 for x in valid if x > 0) / len(valid)
    return f"  {label:12s} RankIC={m:+.4f}  IR={ir:+.2f}  胜率={wr:.1%}  n={len(valid):,}"


# ---------- main ----------
def main() -> None:
    args = parser.parse_args()
    root = Path(args.root)
    store = LocalStore(root)

    # 加载股票列表
    stocks = store.load_stocks()
    if not stocks:
        print("无股票缓存，请先 python scripts/update_data.py --root data --until 2024-06-30")
        sys.exit(1)

    print(f"缓存: {len(stocks)} 只股票")
    # 预加载所有K线到内存
    print("加载K线到内存...", end="", flush=True)
    all_bars: dict[str, pd.DataFrame] = {}
    for s in stocks:
        df = store.load_bars(s.code, date(1990, 1, 1), date(2025, 12, 31))
        if len(df) >= 250:
            # 确保 date 列是 date 对象
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.date
            all_bars[s.code] = df
    print(f" {len(all_bars)} 只有效(>=250根K线)")

    # 收集所有交易日
    all_dates: set[date] = set()
    for df in all_bars.values():
        for d in df["date"]:
            all_dates.add(d)
    dates = sorted(d for d in all_dates if date(2019, 1, 1) <= d <= date(2024, 6, 30))
    # 隔日采样(减少计算量)
    dates = dates[::2]
    print(f"评估: {len(dates)} 个交易日(隔日采样)")

    for horizon in args.horizon:
        ics: dict[str, list[float]] = {k: [] for k in ["bias20", "rsi", "ret20", "ret60"]}
        n_ok = 0

        for di, d in enumerate(dates):
            if di % 200 == 0:
                print(f"  h{horizon} [{di}/{len(dates)}] {d}...", flush=True)

            # 逐只因子值 + 前向收益
            fv = {}
            rets = {}
            for code, df in all_bars.items():
                sub = df[df["date"] <= d]
                vals = _factor_values(sub, d)
                if not vals:
                    continue
                fv[code] = vals
                # 前向收益
                idxs = df.index[df["date"] == d]
                if len(idxs) == 0:
                    continue
                i = int(df.index.get_loc(idxs[0]))
                if i + horizon >= len(df):
                    continue
                bp = float(df["close"].iloc[i])
                sp = float(df["close"].iloc[i + horizon])
                if bp > 0:
                    rets[code] = sp / bp - 1.0

            common = sorted(set(fv) & set(rets))
            if len(common) < 10:
                continue
            n_ok += 1
            rr = [rets[c] for c in common]

            for fn in ics:
                vc = [c for c in common if not math.isnan(fv[c].get(fn, float("nan")))]
                if len(vc) < 10:
                    continue
                ic = _rank_ic([fv[c][fn] for c in vc], [rets[c] for c in vc])
                ics[fn].append(ic)

        print(f"\n=== 前向 {horizon} 日收益 | Rank IC ({n_ok} 个有效评估日) ===")
        for fn in ["ret60", "ret20", "bias20", "rsi"]:
            print(_summary(ics[fn], fn))


parser = argparse.ArgumentParser()
parser.add_argument("--root", default="data")
parser.add_argument("--horizon", "-hz", nargs="+", type=int, default=[5, 10, 20])
if __name__ == "__main__":
    main()
