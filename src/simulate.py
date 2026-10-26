"""Monte Carlo checks of the derivation.

A: Var(IC_t) vs N, B: floor approximations, C: CI coverage, D: power.
"""
import time

import numpy as np

import ic

# lam -> mean IC ~0.03
PAR = dict(gamma=0.35, sd_beta=0.5, sd_f=0.06, sd_eps=0.10, lam=0.0031)
# fewer reps at big N, too slow otherwise
REPS = {50: 20000, 100: 20000, 200: 20000, 400: 20000, 800: 20000,
        1600: 12000, 3200: 12000, 12800: 6000}
SIZES = list(REPS)


def draw_panel(T, N, rng, gamma, sd_beta, sd_f, sd_eps, lam):
    """r = lam*s + beta*f + eps, corr(beta, s) = gamma. beta redrawn each period."""
    s = rng.standard_normal((T, N))
    w = rng.standard_normal((T, N))
    beta = 1.0 + sd_beta * (gamma * s + np.sqrt(1 - gamma ** 2) * w)
    f = sd_f * rng.standard_normal((T, 1))
    r = lam * s + beta * f + sd_eps * rng.standard_normal((T, N))
    return s, r


def rowwise_corr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def ic_draws(n, N, rng, chunk=2_000_000, **par):
    """n draws of IC_t at N (chunked to save memory)"""
    per = max(1, chunk // N)
    out, done = [], 0
    while done < n:
        k = min(per, n - done)
        out.append(rowwise_corr(*draw_panel(k, N, rng, **par)))
        done += k
    return np.concatenate(out)


def main():
    t0 = time.time()
    rng = np.random.default_rng(20260810)
    q = PAR["gamma"] * PAR["sd_beta"]  # cov(s, beta)
    mu, floor, kappa = ic.ic_moments_exact(q, PAR["lam"], PAR["sd_eps"], PAR["sd_beta"] ** 2, PAR["sd_f"])
    k_u = ic.kappa_u_only(q, PAR["lam"], PAR["sd_eps"], PAR["sd_beta"] ** 2, PAR["sd_f"])
    closed = ic.floor_closed(q, PAR["sd_eps"], PAR["sd_beta"] ** 2, PAR["sd_f"])
    naive = q ** 2 * PAR["sd_f"] ** 2 / PAR["sd_eps"] ** 2

    # A
    print("A. Var(IC_t) vs breadth      [floor = %.6f, sigma_F = %.4f]" % (floor, np.sqrt(floor)))
    print("      N    simulated    predicted    err")
    sim_var = []
    for N in SIZES:
        v = ic_draws(REPS[N], N, rng, **PAR).var(ddof=1)
        pred = floor + kappa / N
        sim_var.append(v)
        print("  %6d   %.6f     %.6f   %+5.1f%%" % (N, v, pred, 100 * (pred / v - 1)))
    fv, c = ic.fit_floor(SIZES, sim_var)

    # MC SE of the fit, using SE(var) ~ var * sqrt(2/n)
    X = np.column_stack([np.ones(len(SIZES)), 1.0 / np.array(SIZES, float)])
    w = np.linalg.pinv(X.T @ X) @ X.T
    se_fv, se_c = (np.sqrt(sum((w[j, i] * sim_var[i] * np.sqrt(2 / REPS[N])) ** 2
                               for i, N in enumerate(SIZES))) for j in (0, 1))
    print("  fitted Var = %.6f (+/- %.6f) + %.3f (+/- %.3f)/N   (theory: %.6f + %.3f/N)"
          % (fv, se_fv, c, se_c, floor, kappa))

    # B
    print("\nB. floor estimates vs the fitted floor %.6f +/- %.6f" % (fv, se_fv))
    for name, val in [("exact leading order", floor), ("closed form", closed), ("naive ratio", naive)]:
        print("   %-22s %.6f   %+5.1f%%   %5.1f MC sigma"
              % (name, val, 100 * (val / fv - 1), abs(val - fv) / se_fv))

    # C, D
    NC, NREP, TS = 400, 2000, (24, 60, 120, 300)
    pool = ic_draws(max(TS) * NREP, NC, rng, **PAR)
    sd1 = np.sqrt(floor + kappa / NC)
    print("\nC/D. aggregation over T at N = %d  (true mean IC %.4f, per-period sd %.4f)"
          % (NC, mu, sd1))
    print("       T   coverage   power sim   power analytic")
    cov, pw = {}, {}
    for T in TS:
        p = pool[: T * NREP].reshape(NREP, T)
        m, sd = p.mean(1), p.std(1, ddof=1)
        cov[T] = float((np.abs(m - mu) < 1.96 * sd / np.sqrt(T)).mean())
        pw[T] = (float((np.abs(m) > 1.96 * sd / np.sqrt(T)).mean()),
                 float(ic.power(mu, sd1 / np.sqrt(T))))
        print("   %5d      %.3f       %.3f           %.3f" % (T, cov[T], pw[T][0], pw[T][1]))

    ic.put("sim", dict(par=PAR, sizes=SIZES, var=sim_var, floor_exact=floor, floor_closed=closed,
                       floor_naive=naive, floor_fit_se=se_fv, mean_ic=mu, kappa=kappa, kappa_u_only=k_u, fit_floor=fv, fit_c=c, fit_c_se=se_c,
                       coverage=cov, power=pw))
    print("\n%.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
