# How much data does an information coefficient need?

Code for [`paper/paper.pdf`](paper/paper.pdf). The price data isn't included (see below).

People usually treat a cross-sectional IC as if the N stocks were independent, so the
standard error falls like 1/sqrt(N). But a real signal is correlated with factor
loadings, so each month's IC picks up that month's factor return, which is the same
for every stock and doesn't average out. In this data a cross-section of 1,552 US
stocks is worth about 50 independent observations, and the 465,750 stock-months in
the panel are worth about 15,078 (31x fewer).

So a true IC of 0.03 needs 14.5 years of monthly data to detect at 80% power, or 27.2
years at a t > 3 hurdle. Over any 10-year window the smallest detectable IC is 0.0361.
12-1 momentum here has mean IC 0.0123, t = 1.51, i.e. not significant.

The floor itself is Qian and Hua's (2004) "strategy risk", and Ding and Martin (2017)
show it bounds the information ratio. What the paper adds is the inference side:
effective sample size, required T, and the exaggeration ratio for published ICs.

## Running it

The daily prices come from Yahoo Finance, whose terms allow personal use only, so they
aren't in the repo and you have to pull them yourself. The stock list in
`data/tickers.csv` is the holdings of the iShares Russell 3000 ETF (IWV) on the day I
downloaded the data. `ic-data/` can live elsewhere if you point `$IC_DATA` at it.

```bash
make venv      # venv + requirements.txt
make download  # data/tickers.csv -> ic-data/russell3000_data.csv
make panel     # daily csv -> data/panel.npz (~6s)
make all       # all numbers and figures, ~15s
make paper     # build paper/paper.pdf (tectonic or pdflatex)
make audit     # check three known problems with the raw data (~11s); also needs
               # ic-data/osap_fundamentals.parquet, which download.py does not fetch
```

A fresh download won't match my file exactly. Yahoo re-adjusts old closes after every
dividend, and 25 of the 2,584 tickers in `data/tickers.csv` had no data in my pull either.

## Files

- `src/ic.py`: IC, N_eff, power, Type M, the analytic floor
- `src/download.py`: Yahoo daily prices for `data/tickers.csv`
- `src/panel.py`: daily csv -> monthly returns (`data/panel.npz`)
- `src/simulate.py`: Monte Carlo check of the derivation
- `src/empirical.py`: real IC series, breadth curve, placebo, robustness checks
- `src/figures.py`: the six figures
- `src/audit.py`: data checks
- `tools/make_numbers.py`: `data/results.json` -> `paper/numbers.tex`

The scripts write their results to `data/results.json`, `make_numbers.py` turns that
into LaTeX macros, and the paper uses only those macros, so no estimate in the paper is
typed in by hand.

## Limitations

Every one of the 2,559 tickers survives to the end of the file (no delistings), so
the mean ICs here don't mean much. The paper is about sd(IC), which should be much less
affected, and I checked: on the cohort present from the first month the mean IC moves
from 0.0123 to 0.0040 but the sd only moves from 0.141 to 0.154.

It's also one market, one 25-year window and two price signals. sigma_F ~ 0.14 is
a single measurement. See section 10 of the paper.
