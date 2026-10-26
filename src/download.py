"""data/tickers.csv -> ic-data/russell3000_data.csv (daily prices from Yahoo via yfinance).

Yahoo's data is for personal use, so the csv stays out of the repo. Yahoo re-adjusts old
closes after every dividend or split, and some tickers in the list (25 of 2,584 in the
paper's pull) come back empty, so a fresh pull won't reproduce the paper's numbers exactly.
Tickers with no data are listed in ic-data/download_missing.txt.
"""
import csv
import os
import time
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ROOT / "data" / "tickers.csv"
OUT = Path(os.environ.get("IC_DATA", ROOT / "ic-data")) / "russell3000_data.csv"
MISSING = OUT.parent / "download_missing.txt"
START, END = "2000-01-01", "2026-03-12"  # END is exclusive; the paper's file ends 2026-03-11
CHUNK = 100


def fetch(w, tickers, sector, chunk):
    """write rows for each ticker, return the ones Yahoo gave nothing for"""
    missing = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        df = yf.download(
            batch, start=START, end=END, auto_adjust=True,  # adjusted Close, as panel.py assumes
            group_by="ticker", threads=True, progress=False,
        )
        for t in batch:
            d = df[t].dropna() if t in df.columns.get_level_values(0) else None
            if d is None or d.empty:
                missing.append(t)
                continue
            for day, r in d.iterrows():
                w.writerow([day.strftime("%Y-%m-%d"), r.Open, r.High, r.Low, r.Close,
                            int(r.Volume), t, sector[t]])
        print(f"{min(i + chunk, len(tickers))}/{len(tickers)} tickers")
    return missing


def build():
    t0 = time.time()
    with open(TICKERS) as f:
        sector = {r["Ticker"]: r["Sector"] for r in csv.DictReader(f)}
    tickers = sorted(sector)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".csv.part")  # so an interrupted run can't pass for a full file
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"])
        missing = fetch(w, tickers, sector, CHUNK)
        if missing:  # yfinance swallows errors, including rate limits, so try once more
            print(f"retrying {len(missing)} tickers")
            time.sleep(30)
            missing = fetch(w, missing, sector, 10)
    os.replace(tmp, OUT)

    MISSING.write_text("".join(t + "\n" for t in missing))
    print(f"download: {len(tickers) - len(missing)}/{len(tickers)} tickers -> {OUT} "
          f"({OUT.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s)")
    if missing:
        print(f"no data for {len(missing)}: {' '.join(missing)} (listed in {MISSING.name})")


if __name__ == "__main__":
    build()
