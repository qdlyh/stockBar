"""因子验证 V4 — 真实前向收益 + 动态池证明。

每评估日选 top-N，算实际持有收益，对比 benchmark(全池等权)。
验证动态池: 每日池子大小/成分变化。

用法: python scripts/validate_v4.py [--top-n 5]
"""
from __future__ import annotations

import math, sys, pickle
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

CACHE = Path("D:/stockBar/data/baostock_cache.pkl")

if not CACHE.exists():
    print("请先跑 validate_v3.py 生成缓存")
    sys.exit(1)

with open(CACHE, "rb") as f:
    klines = pickle.load(f)

print(f"缓存: {len(klines)} 只股票\n")

# ── 指标(同V3) ──
def _sma(s,n): return s.rolling(n).mean()
def _rsi(close, n=14):
    d=close.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    ag=g.rolling(n).mean(); al=l.rolling(n).mean()
    out=100-100/(1+ag/al)
    out=out.where(al!=0,100.0); out=out.where(ag!=0,0.0)
    out[(ag==0)&(al==0)]=50.0; out[ag.isna()]=np.nan
    return out
def _bias(close,n): return (close-_sma(close,n))/_sma(close,n)

def _atr_pct(kline):
    if len(kline)<21: return float("nan")
    h=kline["high"].astype(float); l=kline["low"].astype(float); c=kline["close"].astype(float)
    pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=float(tr.tail(20).mean())
    price=float(c.iloc[-1])
    return atr/price if price>0 else float("nan")

def compute(df, as_of):
    sub=df[df["date"]<=as_of]
    if len(sub)<250: return None
    if sub["date"].iloc[-1]!=as_of: return None
    sub=sub.sort_values("date").reset_index(drop=True)
    c=sub["close"]
    return {
        "bias20": float(_bias(c,20).iloc[-1]),
        "rsi":    float(_rsi(c,14).iloc[-1]),
        "atr_pct": float(v) if not math.isnan(v:=_atr_pct(sub)) else float("nan"),
        "close":  float(c.iloc[-1]),
        "bars":   len(sub),
    }

def forward_return(df, as_of, horizon):
    """前向收益: as_of 收盘买入 → as_of+horizon 交易日收盘卖出。"""
    idxs=df.index[df["date"]==as_of]
    if len(idxs)==0: return None
    i=df.index.get_loc(idxs[0])
    if i+horizon>=len(df): return None
    bp=float(df["close"].iloc[i]); sp=float(df["close"].iloc[i+horizon])
    return sp/bp-1 if bp>0 else None


# ── 主循环 ──
args = [a for a in sys.argv if a.startswith("--top-n=")]
TOP_N = int(args[0].split("=")[1]) if args else 5

eval_dates=sorted({d for df in klines.values() for d in df["date"]
                    if "2025-01-01"<=d<="2026-05-31"})[::2]
print(f"评估日: {len(eval_dates)} (2025-01~2026-05, 隔日)")
print(f"top-N: {TOP_N}\n")

# 动态池统计
pool_sizes=[]
pool_turnover=[]
prev_pool=set()

for horizon in [5,10,20]:
    factors=["bias20","rsi","atr_pct"]
    # 每因子: 收集 top-N 收益 + bottom-N 收益 + benchmark
    top_rets: dict[str,list[float]] = {f:[] for f in factors}
    bot_rets: dict[str,list[float]] = {f:[] for f in factors}
    bench_rets: list[float] = []
    pool_stats: list[int] = []

    for di,d in enumerate(eval_dates):
        if di%60==0: print(f"  h{horizon} [{di}/{len(eval_dates)}] {d}",flush=True)

        # 动态池
        pool={}
        today_codes=set()
        for code,df in klines.items():
            fv=compute(df,d)
            if fv:
                pool[code]=fv
                today_codes.add(code)
        pool_sizes.append(len(pool))
        pool_turnover.append(len(today_codes-prev_pool))
        prev_pool=today_codes

        if len(pool)<TOP_N+10: continue
        pool_stats.append(len(pool))

        # 前向收益
        valids={c:forward_return(klines[c],d,horizon) for c in pool}
        valids={c:r for c,r in valids.items() if r is not None}
        common=sorted(set(pool)&set(valids))
        if len(common)<TOP_N+10: continue

        # Benchmark: 全池等权
        bench_rets.append(np.mean([valids[c] for c in common]))

        for fn in factors:
            scored=[]
            for c in common:
                v=pool[c].get(fn)
                if v is not None and not math.isnan(v):
                    scored.append((c,v))
            # direction=-1: 值越小越好
            scored.sort(key=lambda x:x[1])  # 升序
            if len(scored)<TOP_N+10: continue
            top=[c for c,_ in scored[:TOP_N]]
            bot=[c for c,_ in scored[-TOP_N:]]
            top_rets[fn].append(np.mean([valids[c] for c in top]))
            bot_rets[fn].append(np.mean([valids[c] for c in bot]))

    print(f"\n{'='*70}")
    print(f"持有 {horizon} 交易日 | top-{TOP_N} 选股 | {len(pool_stats)} 个有效评估日")
    print(f"日均池子: {np.mean(pool_stats):.0f}只 (min={min(pool_stats)}, max={max(pool_stats)})")
    print(f"全池等权基准: {np.mean(bench_rets)*100:+.2f}% (胜率={sum(1 for r in bench_rets if r>0)/len(bench_rets)*100:.0f}%)")
    print(f"{'='*70}")

    for fn in factors:
        tr=top_rets[fn]; br=bot_rets[fn]
        if not tr: continue
        tm=np.mean(tr); bm=np.mean(br)
        tw=sum(1 for r in tr if r>0)/len(tr)
        bw=sum(1 for r in br if r>0)/len(br)
        spread=tm-bm
        # 年化
        ann_tm=tm*(252/horizon); ann_bm=bm*(252/horizon)
        print(f"  [{fn:8s}] top{TOP_N:2d}: {tm*100:+.2f}% ({tw*100:.0f}%胜) | "
              f"bot: {bm*100:+.2f}% ({bw*100:.0f}%胜) | "
              f"多空: {spread*100:+.2f}% | "
              f"top年化≈{ann_tm*100:+.0f}%")

    # 多因子合成 (等权等向,全=-1)
    synth=[]
    for di,d in enumerate(eval_dates):
        pool={}
        for code,df in klines.items():
            fv=compute(df,d)
            if fv:
                # 剔除任何因子 NaN
                if any(math.isnan(fv.get(f,float("nan"))) for f in factors): continue
                pool[code]=fv
        if len(pool)<TOP_N+10: continue
        valids={c:forward_return(klines[c],d,horizon) for c in pool}
        valids={c:r for c,r in valids.items() if r is not None}
        common=sorted(set(pool)&set(valids))
        if len(common)<TOP_N+10: continue
        # 合成: 每个因子标准化后等权
        scores={c:0.0 for c in common}
        for fn in factors:
            vals=[pool[c][fn] for c in common]
            mn=np.mean(vals); sd=np.std(vals)
            if sd==0: continue
            for i,c in enumerate(common):
                scores[c]+=(vals[i]-mn)/sd  # z-score, 越小越好
        ranked=sorted(scores.items(),key=lambda x:x[1])  # 值小=好
        top=[c for c,_ in ranked[:TOP_N]]
        synth.append(np.mean([valids[c] for c in top]))

    if synth:
        sm=np.mean(synth); sw=sum(1 for r in synth if r>0)/len(synth)
        print(f"  [合成   ] top{TOP_N:2}: {sm*100:+.2f}% ({sw*100:.0f}%胜) | 年化≈{sm*(252/horizon)*100:+.0f}%",flush=True)
    print(flush=True)

# ── 动态池证明 ──
print(f"\n{'='*70}")
print("动态池验证")
print(f"日均池子: {np.mean(pool_sizes):.0f}只 (min={min(pool_sizes)} max={max(pool_sizes)} std={np.std(pool_sizes):.0f})")
print(f"日均新增(退出): {np.mean(pool_turnover):.1f}只")
# 池子成分变化: 比较首日和末日
first_codes=set()
last_codes=set()
for code,df in klines.items():
    fv_first=compute(df, eval_dates[0])
    if fv_first: first_codes.add(code)
    fv_last=compute(df, eval_dates[-1])
    if fv_last: last_codes.add(code)
new_entrants=last_codes-first_codes
exits=first_codes-last_codes
print(f"首日池: {len(first_codes)}只, 末日池: {len(last_codes)}只")
print(f"新增(期间上市/满250根): {len(new_entrants)}只")
print(f"退出(退市/停牌): {len(exits)}只")
print(f"成分变化率: {len(new_entrants|exits)/max(len(first_codes),1)*100:.0f}%")
print(f"→ {len(new_entrants)}只新进入(IPO/满250根), {len(exits)}只退出")
