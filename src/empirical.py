"""Real-data part. Reads data/panel.npz, writes the "emp" section of results.json."""
import time

import numpy as np

import ic
import panel

# stop at 1000, the thinnest month has 1083 names
SIZES = [50, 100, 200, 400, 800, 1000]
DRAWS = 5
PERMS = 20
# placebo pairs with 0 <= t - p <= LOOKBACK overlap the signal's own returns
LOOKBACK = 12
# single test / HLZ 2016 / bonferroni over ~300 anomalies
HURDLES = {"single": 1.96, "hlz": 3.00, "bonferroni300": 3.76}


def summarise(v, ns):
    T, mu, sd = len(v), float(v.mean()), float(v.std(ddof=1))
    return dict(T=T, mean=mu, sd=sd, t_iid=mu / sd * np.sqrt(T), t_nw=mu / ic.nw_se(v),
                ar1=float(np.corrcoef(v[:-1], v[1:])[0, 1]),
                kurtosis=float(((v - mu) ** 4).mean() / sd ** 4),
                median_N=float(np.median(ns)), n_eff=float(ic.n_eff(sd ** 2)))


def curve(s, fwd, rng, randomize=False):
    return [float(np.mean([ic.rank_ic(s, fwd, sub=N, randomize=randomize, rng=rng,
                                      pool_min=max(SIZES))[0].std(ddof=1)
                           for _ in range(DRAWS)])) for N in SIZES]


def main():
    t0 = time.time()
    rets, tickers, months = panel.load()
    sig, fwd = ic.signals(rets)
    rng = np.random.default_rng(20260810)
    out = {"months": [str(months[0]), str(months[-1])], "n_tickers": len(tickers)}

    print("Realised IC (Spearman; signal known at end of t, return earned over t+1)")
    print("  signal        T   mean     sd    t(iid)  t(NW)    AR1  kurt   medN  N_eff")
    for name, s in sig.items():
        v, ns = ic.rank_ic(s, fwd, rng=rng)
        d = summarise(v, ns)
        out[name] = d
        out[name]["series"] = v.tolist()
        out[name]["t_index"] = [t for t in range(len(fwd)) if
                                (np.isfinite(s[t]) & np.isfinite(fwd[t])).sum() >= ic.MIN_N]
        print("  %-9s %4d %7.4f %6.4f %7.2f %6.2f %6.2f %5.1f %6d %6.0f"
              % (name, d["T"], d["mean"], d["sd"], d["t_iid"], d["t_nw"], d["ar1"],
                 d["kurtosis"], d["median_N"], d["n_eff"]))

    print("\nsd(IC) by cross-section size")
    print("      N   mom12_1     rev1   random signal   iid theory")
    curves = {n: curve(s, fwd, rng) for n, s in sig.items()}
    curves["random"] = curve(sig["mom12_1"], fwd, rng, randomize=True)
    iid = [1 / np.sqrt(N - 1) for N in SIZES]
    for i, N in enumerate(SIZES):
        print("  %5d   %7.4f  %7.4f   %9.4f   %10.4f"
              % (N, curves["mom12_1"][i], curves["rev1"][i], curves["random"][i], iid[i]))
    for name in ("mom12_1", "rev1", "random"):
        fv, c = ic.fit_floor(SIZES, np.array(curves[name]) ** 2)
        out.setdefault("fit", {})[name] = dict(floor_var=fv, c=c, sigma_F=float(np.sqrt(max(fv, 0))))
        print("  fit %-8s Var = %+.6f + %.3f/N   ->  sigma_F = %.4f, N_eff ceiling = %.0f"
              % (name, fv, c, np.sqrt(max(fv, 0)), 1 / fv if fv > 0 else np.inf))
    out["curves"] = curves | {"iid": iid, "sizes": SIZES}

    # placebo: pair each signal with some other month's returns. lower bound on the floor
    print("\nmisaligned-month placebo (%d permutations)" % PERMS)
    t = np.arange(len(fwd))
    for name, s in sig.items():
        pl = []
        for _ in range(PERMS):
            p = rng.permutation(len(fwd))
            shuffled = fwd[p]
            shuffled[(t - p >= 0) & (t - p <= LOOKBACK)] = np.nan
            pl.append(ic.rank_ic(s, shuffled, rng=rng)[0])
        pl = np.concatenate(pl)
        out[name]["placebo_sd"] = float(pl.std(ddof=1))
        out[name]["placebo_mean"] = float(pl.mean())
        if name == "mom12_1":
            out[name]["placebo_series"] = pl.tolist()
        print("  %-9s mean %+.4f  sd %.4f  (realised sd %.4f, iid theory %.4f)"
              % (name, pl.mean(), pl.std(ddof=1), out[name]["sd"], 1 / np.sqrt(out[name]["median_N"] - 1)))

    # pearson on raw returns, for comparison
    pv, _ = ic.rank_ic(sig["mom12_1"], fwd, rng=rng)
    raw = []
    for t in range(len(fwd)):
        m = np.isfinite(sig["mom12_1"][t]) & np.isfinite(fwd[t])
        if m.sum() >= ic.MIN_N:
            raw.append(np.corrcoef(sig["mom12_1"][t][m], fwd[t][m])[0, 1])
    raw = np.array(raw)
    out["pearson_raw"] = dict(mean=float(raw.mean()), sd=float(raw.std(ddof=1)))
    print("\nmom12_1 Spearman sd %.4f vs Pearson-on-raw-returns sd %.4f (max monthly return %.0f%%)"
          % (pv.std(ddof=1), raw.std(ddof=1), 100 * np.nanmax(rets)))

    # first-month cohort
    cohort = np.isfinite(rets[0])
    v2, _ = ic.rank_ic(sig["mom12_1"][:, cohort], fwd[:, cohort], rng=rng)
    out["cohort"] = dict(n=int(cohort.sum()), mean=float(v2.mean()), sd=float(v2.std(ddof=1)))
    print("cohort present from %s (%d tickers): mean %+.4f sd %.4f  vs full-panel mean %+.4f sd %.4f"
          % (months[0], cohort.sum(), v2.mean(), v2.std(ddof=1),
             out["mom12_1"]["mean"], out["mom12_1"]["sd"]))

    s_ic = out["mom12_1"]["sd"]
    print("\nAt the measured sigma_IC = %.4f, to detect a true IC of 0.03 at 80%% power:" % s_ic)
    req = {}
    for label, z in HURDLES.items():
        T = ic.required_T(0.03, s_ic, z=z)
        req[label] = T
        print("   %-14s (z=%.2f):  T = %3.0f months = %.1f years" % (label, z, T, T / 12))
    mde = {yr: float(ic.min_detectable(12 * yr, s_ic)) for yr in (5, 10, 20, 25, 57)}
    print("Smallest IC detectable at 80% power in a sample of length:")
    print("   " + "   ".join("%dyr: %.4f" % (y, m) for y, m in mde.items()))
    typem = {yr: float(ic.type_m(0.03, s_ic / np.sqrt(12 * yr))) for yr in (5, 10, 20, 25, 57)}
    print("Type M (exaggeration ratio) for IC=0.03 by sample length:")
    print("   " + "   ".join("%dyr: %.2fx" % (y, m) for y, m in typem.items()))
    T_full = out["mom12_1"]["T"]
    print("Effective observations in this panel: T x N_eff = %d x %.0f = %.0f, against"
          % (T_full, out["mom12_1"]["n_eff"], T_full * out["mom12_1"]["n_eff"]))
    print("   %d x %d = %.0f stock-months claimed, and %.0f needed to detect IC = 0.03."
          % (T_full, out["mom12_1"]["median_N"], T_full * out["mom12_1"]["median_N"],
             (1.96 + 0.8416) ** 2 / 0.03 ** 2))
    out["implications"] = dict(sigma_ic=s_ic, required_T=req, mde=mde, type_m=typem,
                               eff_obs=T_full * out["mom12_1"]["n_eff"],
                               claimed_obs=T_full * out["mom12_1"]["median_N"],
                               needed_obs=(1.96 + 0.8416) ** 2 / 0.03 ** 2)
    ic.put("emp", out)
    print("\n%.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
