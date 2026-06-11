"""因子验证 V3 — 修正因子集：R剔ret20/ret60，新增 bias5 + 波动率收缩。

方向: 全部 -1 (均值回归)。
池子: 动态(point-in-time) >=250根K线、当日非停牌。
2025-01 ~ 2026-05 样本外。
"""
from __future__ import annotations

import math, sys, time, threading, pickle
from pathlib import Path

import baostock as bs
import pandas as pd
import numpy as np

SEEDS = [
    "600000","600036","600519","600887","601318","601857","603259","600276",
    "600030","600585","601166","601012","600900","601088","600809","601899",
    "600028","600048","600196","600309","600346","600406","600436","600570",
    "600690","600745","600763","600872","600893","601138","601225","601390",
    "601688","601766","601800","601888","603160","603501","603986","605499",
    "000001","000002","000858","000333","000651","000568","000725","000063",
    "000100","000157","000338","000425","000538","000625","000661","000776",
    "000800","000876","000895","000938","002142","002230","002304","002415",
    "002475","002594","002714","002007","002050","002129","002236","002352",
    "002371","002410","002459","002460","002466","002475","002714","003816",
    "300750","300059","300124","300015","300274","300760","300502","300014",
    "300033","300122","300142","300347","300408","300413","300450","300496",
    "300529","300601","300661","300750","300782","301269",
    "688981","688111","688036","688012","688005","688008","688256","688561",
    "688188","688223","688390","688536","688599","688981","688126",
]
CACHE = Path("D:/stockBar/data/baostock_cache.pkl")

def _sym(code): return f"sh.{code}" if str(code).zfill(6).startswith("6") else f"sz.{code}"

def _load_or_pull(force=False):
    if not force and CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    print("baostock 登录...", flush=True); bs.login()
    klines = {}
    dedup = list(dict.fromkeys(SEEDS))
    t0 = time.time()
    for i, code in enumerate(dedup):
        result = {}
        def _work():
            try:
                rs = bs.query_history_k_data_plus(
                    _sym(code), "date,open,high,low,close,volume,amount",
                    start_date="2022-01-01", end_date="2026-05-31",
                    frequency="d", adjustflag="2")
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume","amount"])
                    for c in ("open","high","low","close","volume","amount"):
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    df = df.dropna(subset=["close"]); df = df[df["close"] > 0]
                    df["date"] = df["date"].astype(str).str[:10]
                    if len(df) >= 250:
                        result["df"] = df.reset_index(drop=True)
            except Exception: pass
        t = threading.Thread(target=_work, daemon=True); t.start(); t.join(12)
        if "df" in result:
            klines[code] = result["df"]
        if (i+1) % 25 == 0:
            print(f"  [{i+1}/{len(dedup)}] ok={len(klines)} {time.time()-t0:.0f}s", flush=True)
    bs.logout()
    print(f"有效池: {len(klines)}只, 落盘缓存", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f: pickle.dump(klines, f)
    return klines


# ── 指标 ──
def _sma(s,n): return s.rolling(n).mean()
def _rsi(close, n=14):
    d = close.diff(); g = d.clip(lower=0); l=(-d).clip(lower=0)
    ag=g.rolling(n).mean(); al=l.rolling(n).mean()
    out=100-100/(1+ag/al)
    out=out.where(al!=0,100.0); out=out.where(ag!=0,0.0)
    out[(ag==0)&(al==0)]=50.0; out[ag.isna()]=np.nan
    return out
def _bias(close,n): return (close-_sma(close,n))/_sma(close,n)
def _vol_contract(kline):
    """波动率收缩: 1/(1+ATR20/close), 越小越好(已取倒数再取负)。直接用 ATR20/close(越小=越收缩=好)"""
    if len(kline) < 21: return float("nan")
    h=kline["high"].astype(float); l=kline["low"].astype(float); c=kline["close"].astype(float)
    pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=float(tr.tail(20).mean())
    price=float(c.iloc[-1])
    return atr/price if price>0 else float("nan")

def factor_vals(df, as_of):
    sub=df[df["date"]<=as_of]
    if len(sub)<250: return {}
    if sub["date"].iloc[-1]!=as_of: return {}
    sub=sub.sort_values("date").reset_index(drop=True)
    close=sub["close"]; volume=sub["volume"]
    vc=_vol_contract(sub)
    r20=close.pct_change(20).iloc[-1]
    return {
        # ─ 左侧 ─
        "bias20": float(_bias(close,20).iloc[-1]),
        "rsi":    float(_rsi(close,14).iloc[-1]),
        # ─ 右侧(新) ─
        "bias5":  float(_bias(close,5).iloc[-1]),
        "atr_pct": float(vc) if not math.isnan(vc) else float("nan"),
        # ─ 筛选(killer) ─
        "ret60":  float(close.pct_change(60).iloc[-1]),
        "close":  float(close.iloc[-1]),
        "bars":   len(sub),
    }

# ── IC (同V2) ──
def _rank_ic(xs, ys):
    n=len(xs)
    if n<15: return float("nan")
    def _rk(arr):
        idx=sorted(range(n),key=lambda i:arr[i])
        res=[0.0]*n; i=0
        while i<n:
            j=i
            while j+1<n and arr[idx[j+1]]==arr[idx[i]]: j+=1
            for k in range(i,j+1): res[idx[k]]=(i+j)/2
            i=j+1
        return res
    rx,ry=_rk(xs),_rk(ys)
    mx=sum(rx)/n; my=sum(ry)/n
    sx=sum((x-mx)**2 for x in rx); sy=sum((y-my)**2 for y in ry)
    if sx==0 or sy==0: return float("nan")
    return sum((x-mx)*(y-my) for x,y in zip(rx,ry))/math.sqrt(sx*sy)

def summ(ics,label):
    v=[x for x in ics if not math.isnan(x)]
    if len(v)<15: return f"  {label:12s} 样本不足 n={len(v)}"
    m=sum(v)/len(v); vv=sum((x-m)**2 for x in v)/len(v)
    sd=math.sqrt(vv) if vv>0 else 0.0
    ir=m/sd if sd>0 else 0.0
    wr=sum(1 for x in v if x>0)/len(v)
    return f"  {label:12s} RankIC={m:+.4f} IR={ir:+.2f} 胜率={wr:.1%} n={len(v):,}"


# ── main ──
klines=_load_or_pull(force="--force" in sys.argv)
eval_dates=sorted({d for df in klines.values() for d in df["date"]
                    if "2025-01-01"<=d<="2026-05-31"})[::2]
print(f"\n评估: {len(eval_dates)}日 (2025-01~2026-05)",flush=True)

# 新因子集
FACTORS = ["bias20","rsi","bias5","atr_pct"]
# 逐日 killer: ret60越深跌(极负)→直接剔; 也看连跌
for horizon in [5,10,20]:
    ics={k:[] for k in FACTORS+["ret60_killed"]}
    ret60_killed_ic=[]
    n_eval=0
    for di,d in enumerate(eval_dates):
        if di%60==0: print(f"  h{horizon} [{di}/{len(eval_dates)}] {d}...",flush=True)
        pool={}
        for code,df in klines.items():
            fv=factor_vals(df,d)
            if not fv: continue
            # Killer: ret60 < -0.40 (腰斩级) → 剔除
            if fv.get("ret60",0) < -0.40: continue
            pool[code]=fv
        if len(pool)<15: continue
        # 前向收益
        rets={}
        for code in pool:
            df=klines[code]
            idxs=df.index[df["date"]==d]
            if len(idxs)==0: continue
            i=df.index.get_loc(idxs[0])
            if i+horizon>=len(df): continue
            bp=float(df["close"].iloc[i]); sp=float(df["close"].iloc[i+horizon])
            if bp>0: rets[code]=sp/bp-1
        common=sorted(set(pool)&set(rets))
        if len(common)<15: continue
        n_eval+=1
        rr=[rets[c] for c in common]
        for fn in FACTORS:
            vc=[c for c in common if not math.isnan(pool[c].get(fn,float("nan")))]
            if len(vc)<15: continue
            ics[fn].append(_rank_ic([pool[c][fn] for c in vc],[rets[c] for c in vc]))
        # Killer 有效性的间接验证: 被杀的股票前向收益均值
        killed_ret=[]
        for code in pool:
            if pool[code].get("ret60",0)<-0.40:
                if code in rets:
                    killed_ret.append(rets[code])
        if killed_ret: ret60_killed_ic.append(np.mean(killed_ret))

    print(f"\n=== horizon={horizon}日 | n={n_eval} ===")
    print("  正 RankIC → 因子值小 → 前向收益好 ≈ 均值回归有效",flush=True)
    for fn in FACTORS:
        print(summ(ics[fn],fn),flush=True)
    if ret60_killed_ic:
        km=np.mean(ret60_killed_ic)
        print(f"  {'ret60_kill':12s} 被剔除股均收益={km*100:+.2f}% (负=剔除正确)",flush=True)
    print(flush=True)
