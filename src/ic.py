"""IC estimators, power calcs, and the results.json read/write."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, rankdata

MIN_N = 200
RESULTS = Path(__file__).resolve().parents[1] / "data" / "results.json"


def put(section, payload):
    all_ = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    all_[section] = payload
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(all_, indent=1, sort_keys=True, default=float))


def get():
    return json.loads(RESULTS.read_text())


def rank_ic(sig, fwd, min_n=MIN_N, sub=None, randomize=False, rng=None, pool_min=None):
    """Spearman IC of sig[t] vs fwd[t] for each t.

    sub: subsample this many names. pool_min: skip months with fewer names (pass the
    largest N so every N uses the same months). randomize: permute the signal.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    ics, ns = [], []
    for t in range(sig.shape[0]):
        ok = np.where(np.isfinite(sig[t]) & np.isfinite(fwd[t]))[0]
        if len(ok) < max(min_n, pool_min or 0):
            continue
        if sub is not None:
            ok = rng.choice(ok, sub, replace=False)
        a = rng.permutation(len(ok)).astype(float) if randomize else rankdata(sig[t][ok])
        ics.append(np.corrcoef(a, rankdata(fwd[t][ok]))[0, 1])
        ns.append(len(ok))
    return np.array(ics), np.array(ns)


def signals(rets):
    """sig[t] known at end of month t, fwd[t] = return over t+1."""
    logr = np.log1p(rets)
    mom = np.full_like(rets, np.nan)
    for t in range(12, rets.shape[0]):
        w = logr[t - 11:t]  # skip the latest month
        mom[t] = np.where(np.isfinite(w).all(0), w.sum(0), np.nan)
    return {"mom12_1": mom[:-1], "rev1": -rets[:-1]}, rets[1:]


def fit_floor(sizes, variances):
    """OLS of Var(IC_t) on [1, 1/N] -> (sigma_F^2, c)"""
    X = np.column_stack([np.ones(len(sizes)), 1.0 / np.asarray(sizes, float)])
    (floor_var, c), *_ = np.linalg.lstsq(X, np.asarray(variances), rcond=None)
    return floor_var, c


def n_eff(var_ic):
    return 1.0 / var_ic


def power(rho, se, z=1.96):
    d = rho / se
    return norm.sf(z - d) + norm.cdf(-z - d)


def required_T(rho, sigma_ic, z=1.96, target=0.80):
    return (z + norm.ppf(target)) ** 2 * sigma_ic ** 2 / rho ** 2


def min_detectable(T, sigma_ic, z=1.96, target=0.80):
    return (z + norm.ppf(target)) * sigma_ic / np.sqrt(T)


def type_m(rho, se, z=1.96):
    """Gelman & Carlin (2014) exaggeration ratio"""
    d = rho / se
    p = norm.sf(z - d) + norm.cdf(-z - d)
    tail = d * norm.sf(z - d) + norm.pdf(z - d) - d * norm.cdf(-z - d) + norm.pdf(z + d)
    return tail / (d * p)


def nw_se(x, lags=6):
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    s = x @ x / T
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / T
    return np.sqrt(s / T)


def ic_moments_exact(q, lam, sd_eps, var_beta, sd_f):
    """(mean, floor, kappa) with Var(IC_t) ~ floor + kappa/N.

    floor = Var_f[g], kappa = E_f[(1-g^2)^2] - Cov_f(g, g(1-g^2)). See appendix.
    """
    g = lambda f: (lam + q * f) / np.sqrt(sd_eps ** 2 + var_beta * f ** 2 + lam ** 2 + 2 * lam * q * f)
    lim = 10 * sd_f
    E = lambda h: quad(lambda f: h(f) * norm.pdf(f, 0.0, sd_f), -lim, lim, limit=200)[0]
    m1, m2 = E(g), E(lambda f: g(f) ** 2)
    h = lambda f: g(f) * (1 - g(f) ** 2)
    kappa = E(lambda f: (1 - g(f) ** 2) ** 2) - (E(lambda f: g(f) * h(f)) - m1 * E(h))
    return m1, m2 - m1 ** 2, kappa


def kappa_u_only(q, lam, sd_eps, var_beta, sd_f):
    """kappa from the first, incomplete derivation (only cov_i(s, eps)). kept for the comparison"""
    d2 = lambda f: sd_eps ** 2 + var_beta * f ** 2 + lam ** 2 + 2 * lam * q * f
    lim = 10 * sd_f
    return quad(lambda f: sd_eps ** 2 / d2(f) * norm.pdf(f, 0.0, sd_f), -lim, lim, limit=200)[0]


def floor_closed(q, sd_eps, var_beta, sd_f):
    """closed-form approx, ~15% too high. kept for the comparison"""
    return q ** 2 * sd_f ** 2 / (sd_eps ** 2 + var_beta * sd_f ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, z = 1.2, 1.96
    x = rng.standard_normal(4_000_000) + d
    sig = np.abs(x) > z
    assert abs(type_m(d, 1.0, z) / (np.abs(x[sig]).mean() / d) - 1) < 0.01, "type_m"
    assert abs(power(d, 1.0, z) / sig.mean() - 1) < 0.01, "power"
    T = required_T(0.03, 0.14)
    assert abs(min_detectable(T, 0.14) - 0.03) < 1e-12, "T* / MDE inconsistent"
    sizes = np.array([50, 200, 800, 3200])
    fv, c = fit_floor(sizes, 0.0185 + 1.0 / sizes)
    assert abs(fv - 0.0185) < 1e-9 and abs(c - 1.0) < 1e-9, "fit_floor"
    print("ic.py ok")
