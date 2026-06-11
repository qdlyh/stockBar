"""因子验证 V2 — 全均值回归方向，2025-2026/5 样本外，动态(点-in-time)池。

方向: 全部 -1 (越小越好 — 均值回归)。
池子: 每个评估日，只纳入截至当日>=250根K线的活跃股(不向前偷看)。
数据源: baostock (2022-01~2026-05，覆盖2025-2026评估)。

用法: python scripts/validate_v2.py [--n 120]
"""
from __future__ import annotations

import math, sys, time, threading, argparse
import baostock as bs
import pandas as pd
import numpy as np

# ── 代表股池(各板块覆盖) ──
SEEDS = [
    # 沪主板
    "600000","600036","600519","600887","601318","601857","603259","600276",
    "600030","600585","601166","601012","600900","601088","600809","601899",
    "600028","600048","600196","600309","600346","600406","600436","600570",
    "600690","600745","600763","600872","600893","601138","601225","601390",
    "601688","601766","601800","601888","603160","603501","603986","605499",
    # 深主板
    "000001","000002","000858","000333","000651","000568","000725","000063",
    "000100","000157","000338","000425","000538","000625","000661","000776",
    "000800","000876","000895","000938","002142","002230","002304","002415",
    "002475","002594","002714","002007","002050","002129","002236","002352",
    "002371","002410","002459","002460","002466","002475","002714","003816",
    # 创业板
    "300750","300059","300124","300015","300274","300760","300502","300014",
    "300033","300122","300142","300347","300408","300413","300450","300496",
    "300529","300601","300661","300750","300782","301269",
    # 科创板(2020后上市，2022起有足够历史)
    "688981","688111","688036","688012","688005","688008","688256","688561",
    "688188","688223","688390","688536","688599","688981","688126",
]

def _sym(code: str) -> str:
    c = str(code).zfill(6)
    return f"sh.{c}" if c.startswith("6") else f"sz.{c}"

def _pull(bs_obj, code: str, timeout: float = 12.0) -> pd.DataFrame | None:
    """拉单只日K(前复权)，12秒超时(防卡死)。失败返回 None。"""
    result = {}
    def _work():
        try:
            rs = bs_obj.query_history_k_data_plus(
                _sym(code), "date,open,high,low,close,volume,amount",
                start_date="2022-01-01", end_date="2026-05-31",
                frequency="d", adjustflag="2")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume","amount"])
                for col in ("open","high","low","close","volume","amount"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                df = df[df["close"] > 0]
                df["date"] = df["date"].astype(str).str[:10]
                if len(df) >= 250:
                    result["df"] = df.reset_index(drop=True)
        except Exception:
            pass

    t = threading.Thread(target=_work, daemon=True)
    t.start(); t.join(timeout)
    return result.get("df")


# ── 指标 ──
def _rsi(close, n=14):
    d = close.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.rolling(n).mean(); al = l.rolling(n).mean()
    out = 100 - 100/(1 + ag/al)
    out = out.where(al != 0, 100.0); out = out.where(ag != 0, 0.0)
    out[(ag == 0) & (al == 0)] = 50.0; out[ag.isna()] = np.nan
    return out

def _sma(s, n): return s.rolling(n).mean()
def _bias(close, n): return (close - _sma(close, n)) / _sma(close, n)
def _ret_n(close, n): return close.pct_change(n)
def _vol20(amt): return float(amt.tail(20).mean())

def factor_vals(df: pd.DataFrame, as_of: str) -> dict[str, float]:
    """截至 as_of 的因子值(全均值回归方向：越小越好)。"""
    sub = df[df["date"] <= as_of]
    if len(sub) < 250:       # 至少250根才进入动态池
        return {}
    # 停牌: 最新日 ≠ as_of
    if sub["date"].iloc[-1] != as_of:
        return {}
    sub = sub.sort_values("date").reset_index(drop=True)
    close = sub["close"]
    volume = sub["volume"]
    return {
        # 左侧因子
        "bias20":  float(_bias(close, 20).iloc[-1]),
        "rsi":     float(_rsi(close, 14).iloc[-1]),
        "ret20":   float(_ret_n(close, 20).iloc[-1]),
        # 右侧因子 (现也朝均值回归：动量弱=好)
        "ret60":   float(_ret_n(close, 60).iloc[-1]),
        # 共用
        "vol20":   _vol20(sub["amount"]),
        "close":   float(close.iloc[-1]),
        "bars":    len(sub),
    }


# ── IC ──
def _rank_ic(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 15: return float("nan")
    def _rk(arr):
        idx = sorted(range(n), key=lambda i: arr[i])
        res = [0.0]*n; i = 0
        while i < n:
            j = i
            while j+1<n and arr[idx[j+1]]==arr[idx[i]]: j+=1
            for k in range(i,j+1): res[idx[k]]=(i+j)/2
            i=j+1
        return res
    rx, ry = _rk(xs), _rk(ys)
    mx=sum(rx)/n; my=sum(ry)/n
    sx=sum((x-mx)**2 for x in rx); sy=sum((y-my)**2 for y in ry)
    if sx==0 or sy==0: return float("nan")
    return sum((x-mx)*(y-my) for x,y in zip(rx,ry))/math.sqrt(sx*sy)


def summ(ics, label):
    valid = [x for x in ics if not math.isnan(x)]
    if len(valid) < 15:
        return f"  {label:12s} 样本不足 n={len(valid)}"
    m = sum(valid)/len(valid); v = sum((x-m)**2 for x in valid)/len(valid)
    sd = math.sqrt(v) if v>0 else 0.0
    ir = m/sd if sd>0 else 0.0
    wr = sum(1 for x in valid if x>0)/len(valid)
    return f"  {label:12s} RankIC={m:+.4f} IR={ir:+.2f} 胜率={wr:.1%} n={len(valid):,}"


# ── main ──
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=120)
args = parser.parse_args()

# 1. 拉数据
dedup = list(dict.fromkeys(SEEDS))   # 去重保序
sample = dedup[:args.n]
print(f"baostock 登录...", flush=True)
bs.login()
print(f"拉 {len(sample)} 只 K线(2022-01~2026-05, 12秒/只超时)...", flush=True)
klines = {}
t0 = time.time()
n_fail = 0
for i, code in enumerate(sample):
    df = _pull(bs, code)
    if df is not None:
        klines[code] = df
    else:
        n_fail += 1
    if (i+1) % 20 == 0:
        print(f"  [{i+1}/{len(sample)}] ok={len(klines)} fail={n_fail} {time.time()-t0:.0f}s", flush=True)
bs.logout()
print(f"有效池: {len(klines)} 只 (失败{n_fail}只, {time.time()-t0:.0f}s)\n", flush=True)

if len(klines) < 20:
    print("股票太少, 退出"); sys.exit(1)

# 2. 评估: 2025-01-01 ~ 2026-05-31
# 动态池: 每评估日只纳入>=250根K线且当日非停牌的股票
all_dates = sorted({d for df in klines.values() for d in df["date"]
                    if "2025-01-01" <= d <= "2026-05-31"})
eval_dates = all_dates[::2]  # 隔日采样
print(f"评估: {len(eval_dates)} 个交易日(2025-01~2026-05), 池子每日动态更新\n", flush=True)

for horizon in [5, 10, 20]:
    ics = {k: [] for k in ["bias20","rsi","ret20","ret60"]}
    pool_sizes = []
    n_eval = 0

    for di, d in enumerate(eval_dates):
        if di % 80 == 0:
            print(f"  h{horizon} [{di}/{len(eval_dates)}] {d}...", flush=True)

        # 动态池: >=250 bars at d, non-suspended
        pool: dict[str, dict] = {}
        for code, df in klines.items():
            fv = factor_vals(df, d)
            if fv:
                pool[code] = fv
        pool_sizes.append(len(pool))
        if len(pool) < 10:
            continue

        # 前向收益
        codes = list(pool)
        rets = {}
        for code in codes:
            df = klines[code]
            idxs = df.index[df["date"] == d]
            if len(idxs) == 0: continue
            i = df.index.get_loc(idxs[0])
            if i + horizon >= len(df): continue
            bp = float(df["close"].iloc[i])
            sp = float(df["close"].iloc[i + horizon])
            if bp > 0: rets[code] = sp/bp - 1

        common = sorted(set(pool) & set(rets))
        if len(common) < 15: continue
        n_eval += 1
        rr = [rets[c] for c in common]

        for fn in ics:
            vc = [c for c in common if not math.isnan(pool[c].get(fn, float("nan")))]
            if len(vc) < 15: continue
            ic = _rank_ic([pool[c][fn] for c in vc], [rets[c] for c in vc])
            ics[fn].append(ic)

    avg_pool = sum(pool_sizes)/len(pool_sizes) if pool_sizes else 0
    print(f"\n=== horizon={horizon}日 | 评估日={n_eval} | 日均池子={avg_pool:.0f}只 ===")
    print("  解读: 全部方向=-1(越小越好)。")
    print("  正 RankIC → 因子值小(动量弱/超卖/负乖离)的股前向收益好 ≈ 均值回归有效。")
    for fn in ["ret60","ret20","bias20","rsi"]:
        print(summ(ics[fn], fn))
    print(flush=True)
