"""本地缓存：行情/财务用 Parquet，股票列表与交易日历用 SQLite。"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from stockbar.datafeed.instruments import Board
from stockbar.datafeed.source import BAR_COLUMNS, StockInfo


def _to_date(v) -> date:
    if pd.isna(v):
        return None  # caller must filter these out
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    return pd.Timestamp(v).date()


def _drop_null_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the 'date' column is null/NaT."""
    mask = df["date"].map(lambda v: not pd.isna(v))
    return df.loc[mask].reset_index(drop=True)


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
        out = _drop_null_dates(out)
        out["date"] = out["date"].map(_to_date)
        out = out.sort_values("date").reset_index(drop=True)
        out.to_parquet(self._bars_path(code), index=False)

    def append_bars(self, code: str, df: pd.DataFrame) -> None:
        existing = self._read_all_bars(code)
        new = _drop_null_dates(df[BAR_COLUMNS].copy())
        combined = pd.concat([existing, new], ignore_index=True)
        combined["date"] = combined["date"].map(_to_date)
        combined = _drop_null_dates(combined)
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
