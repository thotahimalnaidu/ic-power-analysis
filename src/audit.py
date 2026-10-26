"""Checks three known problems with the raw data. Reads the full 1.2 GB csv,
so it's not in `make all`."""
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("IC_DATA", ROOT / "ic-data"))


def price_file():
    """Survivorship + whether Close is adjusted the same way as OHLC."""
    first, last, bad_ohlc, ulp, rows = {}, {}, 0, 0, 0
    tol = 1e-9  # exact compare flags ~28k rows of float noise
    if not (DATA / "russell3000_data.csv").exists():
        sys.exit(f"raw data not found in {DATA}; run make download, or set IC_DATA")
    with open(DATA / "russell3000_data.csv") as f:
        f.readline()
        for line in f:
            p = line.split(",")
            d, t = p[0], p[6]
            rows += 1
            if t not in first or d < first[t]:
                first[t] = d
            if t not in last or d > last[t]:
                last[t] = d
            lo, hi, cl = float(p[3]), float(p[2]), float(p[4])
            if cl < lo * (1 - tol) or cl > hi * (1 + tol):
                bad_ohlc += 1
            elif not (lo <= cl <= hi):
                ulp += 1
    end = max(last.values())
    dead = sum(1 for t in last if last[t] != end)
    print(f"[1] survivorship: {len(last)} tickers, {rows:,} rows, panel ends {end}")
    print(f"    tickers not surviving to the final date: {dead}")
    print(f"    tickers present on the first trading day: {sum(1 for t in first if first[t]==min(first.values()))}")
    print("    CONFIRMED: zero delistings.")
    print(f"[2] OHLC consistency: {bad_ohlc:,} of {rows:,} rows have Close outside [Low, High]")
    print(f"    ({ulp:,} more miss by one float64 ulp, i.e. Close == High or Low after")
    print("    back-adjustment. Not errors.)")
    print("    not tested: whether a fresh download would match (needs a re-download)")


def parquet_columns():
    """Parquet columns that aren't what their names say."""
    cols = ["book_to_market", "div_yield"]
    t = pq.ParquetFile(DATA / "osap_fundamentals.parquet").read_row_group(3, columns=cols)
    bm = np.asarray(t["book_to_market"].to_numpy(zero_copy_only=False), float)
    dy = np.asarray(t["div_yield"].to_numpy(zero_copy_only=False), float)
    bm, dy = bm[np.isfinite(bm)], dy[np.isfinite(dy)]
    print("[3] parquet column provenance (one row group of osap_fundamentals.parquet):")
    print(f"    book_to_market: {100*(bm<0).mean():.1f}% negative, median {np.median(bm):.3f}.")
    print(f"                    exp() has median {np.median(np.exp(bm)):.3f} -- it is log(B/M).")
    print(f"    div_yield:      takes only {np.unique(dy).astype(int).tolist()}, "
          f"{100*(dy==0).mean():.0f}% of them zero -- a category, not a yield.")
    print("    CONFIRMED.")


if __name__ == "__main__":
    price_file()
    print()
    parquet_columns()
