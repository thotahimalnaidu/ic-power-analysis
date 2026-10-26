"""ic-data/russell3000_data.csv (daily) -> data/panel.npz (monthly returns)."""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("IC_DATA", ROOT / "ic-data")) / "russell3000_data.csv"
OUT = ROOT / "data" / "panel.npz"


def month_end_closes(path):
    """last close per (ticker, month)"""
    last = {}
    with open(path) as f:
        f.readline()  # header: Date,Open,High,Low,Close,Volume,Ticker,Sector
        for line in f:
            p = line.split(",")
            last[(p[6], p[0][:7])] = p[4]
    return last


def build():
    t0 = time.time()
    if not RAW.exists():
        sys.exit(f"raw data not found at {RAW}\nrun make download, or set IC_DATA to the directory holding it")
    last = month_end_closes(RAW)

    tickers = sorted({t for t, _ in last})
    months = sorted({m for _, m in last})
    ti = {t: i for i, t in enumerate(tickers)}
    mi = {m: i for i, m in enumerate(months)}
    prices = np.full((len(months), len(tickers)), np.nan)
    for (t, m), c in last.items():
        prices[mi[m], ti[t]] = float(c)

    rets = prices[1:] / prices[:-1] - 1.0  # Close is adjusted, so total return
    months = months[1:]

    # last month is partial (file ends 2026-03-11)
    rets, months = rets[:-1], months[:-1]

    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT, rets=rets.astype(np.float32), tickers=np.array(tickers), months=np.array(months)
    )
    n = np.isfinite(rets).sum(1)
    print(
        f"panel: {rets.shape[0]} months {months[0]}..{months[-1]}, {len(tickers)} tickers, "
        f"median {int(np.median(n))} names/month -> {OUT.name} "
        f"({OUT.stat().st_size/1e6:.1f} MB, {time.time()-t0:.1f}s)"
    )


def load():
    if not OUT.exists():
        sys.exit(f"{OUT} not found; run make download, then make panel")
    z = np.load(OUT, allow_pickle=False)
    return z["rets"].astype(np.float64), z["tickers"], z["months"]


if __name__ == "__main__":
    build()
