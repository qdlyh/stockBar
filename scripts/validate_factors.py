"""因子 Rank IC 验证 — 用 baostock（免费、无代理）直连，不依赖 LocalStore/akshare。

评估: 2018-2024，反幸存者(含退市股)，逐日横截面 RankIC(Spearman)。
因子: 我们的左/右因子，直接算，不走特征表(无需 PE/PB)。

用法: python scripts/validate_factors.py [--n-stocks 80]
"""
from __future__ import annotations

import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import baostock as bs
import pandas as pd
import numpy as np

BAR_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]

# ---------- baostock ----------

def _bs_sym(code: str) -> str:
    """6位代码 → baostock 符号。修复: 689xxx 正确映射 sh (科创板)。"""
    c = str(code).zfill(6)
    # 沪市: 60xxxx, 688xxx, 689xxx
    if c.startswith("6"):
        return f"sh.{c}"
    # 北交所(baostock 多数不支持,先走 bj,后面会因无数据被自然淘汰)
    if c.startswith(("4", "8", "92")):
        return f"bj.{c}"
    # 其余: 深市 (00xxxx, 002xxx, 003xxx, 30xxxx, 301xxx)
    return f"sz.{c}"


def _kline(code: str, bs_obj, start="2017-01-01", end="2024-12-31") -> pd.DataFrame:
    rs = bs_obj.query_history_k_data_plus(
        _bs_sym(code), "date,open,high,low,close,volume,amount",
        start_date=start, end_date=end, frequency="d", adjustflag="2")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=BAR_COLS)
    df = pd.DataFrame(rows, columns=BAR_COLS)
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0].reset_index(drop=True)
    df["date"] = df["date"].astype(str).str[:10]
    return df


def _build_pool(n_stocks: int = 80) -> dict[str, pd.DataFrame]:
    """抽 ~n 只主板+创业板+科创板(含退市股)，反幸存者。返回 {code: kline_df}。"""
    bs_obj = bs
    bs.login()

    # 1. 现存股
    live_codes = []
    rs = bs_obj.query_all_stock(day="2024-12-31")
    while rs.error_code == "0" and rs.next():
        row = rs.get_row_data()
        if len(row) >= 2 and row[1] == "1":  # tradeStatus=1
            live_codes.append(row[0].split(".")[-1])

    # 2. 退市股 (用 akshare 东财退市名单——这一步可能被代理墙挡,
    #    能拿到最好，拿不到就只用现存股，IC 验证仍然有效只是幸存者偏差稍大)
    delisted = set()
    try:
        import akshare as ak
        for fn in [ak.stock_info_sz_delist, ak.stock_info_sh_delist]:
            try:
                raw = fn(symbol="终止上市公司" if fn == ak.stock_info_sz_delist else "全部")
                for _, r in raw.iterrows():
                    c = str(r[raw.columns[0]]).zfill(6)
                    if fn == ak.stock_info_sh_delist:
                        c = str(r[raw.columns[0]]).zfill(6)
                    delisted.add(c)
            except Exception:
                pass
    except Exception:
        pass

    all_codes = sorted(set(live_codes) | delisted)
    # 只取 A股(去指数、去B股)且主板/创业板/科创板
    a_codes = [c for c in all_codes
               if c.isdigit() and len(c) == 6
               and not c.startswith(("9",))     # 900xxx B股 / 920xxx 北交所
               and not c.startswith(("4", "8"))  # 4xxxxx 三板 / 8xxxxx 北交所
               and not c.startswith("200")]      # 200xxx 深B股(不误伤 002xxx)
    print(f"候选池: 现存{len(live_codes)} + 退市{len(delisted)} → A股{len(a_codes)}只, 抽样{n_stocks}只")
    sample = a_codes[:: max(1, len(a_codes) // n_stocks)]

    klines: dict[str, pd.DataFrame] = {}
    for i, code in enumerate(sample):
        try:
            df = _kline(code, bs_obj)
            if len(df) >= 250:
                klines[code] = df
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  拉K线 {i+1}/{len(sample)}, 有效{len(klines)}只")

    bs.logout()
    return klines


# ---------- 指标（复用我们 indicators.py 的逻辑）----------

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    out = out.where(~flat, 50.0)
    out[avg_gain.isna()] = np.nan
    return out


def _bias(close: pd.Series, n: int) -> pd.Series:
    ma = _sma(close, n)
    return (close - ma) / ma


def _ret_n(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(n)


def _is_new_high(close: pd.Series, n: int) -> bool:
    """最新收盘是否为近 n 期最高。数据不足 → False。"""
    window = close.dropna()
    if len(window) < n + 1:
        # 数据不足: 用现有窗口近似判断
        if window.empty:
            return False
    else:
        window = window.tail(n + 1)
    return bool(window.iloc[-1] >= window.max())


def _is_ma_bullish(close: pd.Series) -> bool:
    m5 = _sma(close, 5).iloc[-1]
    m10 = _sma(close, 10).iloc[-1]
    m20 = _sma(close, 20).iloc[-1]
    if any(pd.isna(x) for x in (m5, m10, m20)):
        return False
    return bool(m5 > m10 > m20)


def _avg_amount(amount: pd.Series, n: int = 20) -> float:
    return float(amount.tail(n).mean())


# ---------- 因子值（单个值，as_of 截面）----------

def _factor_values(df: pd.DataFrame, as_of: str) -> dict:
    """给定一只股票截至 as_of 的K线，返回 {因子名: 值}。
    所有因子值都是原始量(不标准化)，IC 算排名相关所以不受量纲影响。
    df["date"] 是字符串 'YYYY-MM-DD'。
    """
    sub = df[df["date"] <= as_of].copy()
    if len(sub) < 63:  # ret60 / new_high60 等需要至少 63 根
        return {}
    sub = sub.sort_values("date").reset_index(drop=True)
    close = sub["close"]
    volume = sub["volume"]
    amount = sub["amount"]

    closest_date = sub["date"].iloc[-1]
    suspended = (closest_date != as_of)

    return {
        # --- 左侧因子 (越小越好: 低估值、超跌、超卖) ---
        # PE/PB 暂缺(无财务数据)，用纯价量代理:
        "bias20": float(_bias(close, 20).iloc[-1]),          # 负乖离 → 超跌
        "rsi":    float(_rsi(close, 14).iloc[-1]),           # RSI 越低越超卖
        "ret20":  float(_ret_n(close, 20).iloc[-1]),         # 近20日跌幅越大越好
        # --- 右侧因子 (越大越好: 动量、趋势) ---
        "ret60":       float(_ret_n(close, 60).iloc[-1]),    # 60日动量
        "new_high60":  1.0 if _is_new_high(close, 60) else 0.0,
        "ma_bullish":  1.0 if _is_ma_bullish(close) else 0.0,
        "vol_price_up": 1.0 if (_is_ma_bullish(close) and
                                float(volume.iloc[-1]) > float(volume.tail(5).mean())) else 0.0,
        # --- 共用 ---
        "suspended": 1.0 if suspended else 0.0,              # 停牌标记(IC计算时剔除)
        "close": float(close.iloc[-1]),
    }


# ---------- IC 计算 ----------

def _pearson(xs: list[float], ys: list[float]) -> float:
    """皮尔逊相关。样本<2 或零方差 → NaN。"""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx)**2 for x in xs)
    sy = sum((y - my)**2 for y in ys)
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sx * sy)


def _rank(xs: list[float]) -> list[float]:
    """排名(0~n-1), 并列取平均排位(修复参考项目不处理并列的 bug)。"""
    n = len(xs)
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    result = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j+1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            result[indexed[k][0]] = avg_rank
        i = j + 1
    return result


def _rank_ic(factor_vals: list[float], fwd_rets: list[float]) -> float:
    """Rank IC (Spearman 排名相关)。"""
    return _pearson(_rank(factor_vals), _rank(fwd_rets))


def _summary(ics: list[float], label: str) -> str:
    ics = [x for x in ics if not math.isnan(x)]
    if len(ics) < 10:
        return f"  {label:20s} 样本不足 (n={len(ics)})"
    mean = sum(ics) / len(ics)
    var = sum((x - mean)**2 for x in ics) / len(ics)
    sd = math.sqrt(var) if var > 0 else 0.0
    ir = mean / sd if sd > 0 else 0.0
    win = sum(1 for x in ics if x > 0) / len(ics)
    return (f"  {label:20s} RankIC={mean:+.4f}  IR={ir:+.2f}  "
            f"胜率={win:.1%}  n={len(ics):,}")


# ---------- 主流程 ----------

def validate(klines: dict[str, pd.DataFrame], horizon: int = 10):
    """对所有交易日逐日横截面 Rank IC。horizon=10(默认10日前向收益)。"""
    # 收集所有候选日
    all_dates: set[str] = set()
    for df in klines.values():
        all_dates.update(df["date"].tolist())
    dates = sorted(d for d in all_dates if "2018-01-01" <= d <= "2024-12-31")
    print(f"\n评估: {len(dates)} 个交易日, horizon={horizon}日, {len(klines)} 只股票")

    factor_ics: dict[str, list[float]] = {
        "bias20": [], "rsi": [], "ret20": [],
        "ret60": [], "new_high60": [], "ma_bullish": [], "vol_price_up": [],
    }
    n_dates_evaluated = 0

    for di, d in enumerate(dates):
        if di % 100 == 0:
            print(f"  [{di}/{len(dates)}] {d}...")

        # 逐只取因子值 + 前向收益
        factor_rows: dict[str, dict[str, float]] = {}  # code -> {factor: val}
        fwd_rets: dict[str, float] = {}                # code -> fwd return

        for code, df in klines.items():
            fv = _factor_values(df, d)
            if not fv or fv.get("suspended", 0) > 0.5:
                continue
            # 前向收益: d 日后第 horizon 个交易日
            idxs = df.index[df["date"] == d]
            if len(idxs) == 0:
                continue
            i = df.index.get_loc(idxs[0])
            if i + horizon >= len(df):
                continue
            buy_price = float(df["close"].iloc[i])
            sell_price = float(df["close"].iloc[i + horizon])
            if buy_price <= 0:
                continue
            factor_rows[code] = fv
            fwd_rets[code] = sell_price / buy_price - 1.0

        if len(factor_rows) < 10:  # 当日有效票太少,跳过
            continue
        n_dates_evaluated += 1

        rets_list = [fwd_rets[c] for c in factor_rows]
        for factor_name in factor_ics:
            vals = [factor_rows[c].get(factor_name, float("nan")) for c in factor_rows]
            vals = [v for v in vals if not math.isnan(v)]
            if len(vals) < 10:
                continue
            # 需要 vals 和 rets 对齐:只取两个都有效的 code
            valid_codes = [c for c in factor_rows
                           if not math.isnan(factor_rows[c].get(factor_name, float("nan")))
                           and c in fwd_rets]
            if len(valid_codes) < 10:
                continue
            aligned_vals = [factor_rows[c][factor_name] for c in valid_codes]
            aligned_rets = [fwd_rets[c] for c in valid_codes]
            ic = _rank_ic(aligned_vals, aligned_rets)
            factor_ics[factor_name].append(ic)

    # --- 报告 ---
    print(f"\n{'='*65}")
    print(f"因子 Rank IC (Spearman), horizon={horizon}日, "
          f"{len(klines)}只股票, {n_dates_evaluated}个评估日")
    print(f"解读: 正 RankIC 说明因子值大的股票前向收益好(右侧对,左侧反)")
    print(f"{'='*65}")
    for factor_name in ["ret60", "new_high60", "ma_bullish", "vol_price_up",
                         "bias20", "rsi", "ret20"]:
        print(_summary(factor_ics[factor_name], factor_name))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    pool = _build_pool(n)
    if len(pool) < 20:
        print("股票太少,退出"); sys.exit(1)
    validate(pool)
