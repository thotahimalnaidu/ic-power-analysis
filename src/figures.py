"""Figures for the paper, from data/results.json -> paper/figures/*.pdf"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import ic

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
NODATE = {"CreationDate": None}
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8a85"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.6,
})


def finish(fig, ax, name, xlabel, ylabel, title, ncol=0):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10, pad=22 if ncol else 6)
    if ncol:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, fontsize=8,
                  columnspacing=1.3, handlelength=1.7)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", metadata=NODATE)
    plt.close(fig)
    print("  ", name)


def fig1_simulation(s):
    N = np.array(s["sizes"], float)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    grid = np.logspace(np.log10(40), np.log10(2e4), 200)
    ax.plot(grid, s["floor_exact"] + s["kappa"] / grid, color=INK, lw=1.4,
            label=r"theory: $\sigma_F^2 + \kappa/N$")
    ax.axhline(s["floor_exact"], color=MUTED, ls="--", lw=1.2)
    ax.plot(N, s["var"], "o", color=BLUE, ms=6, mec="white", mew=0.8, label="simulated", zorder=3)
    ax.text(0.98, s["floor_exact"] * 1.07, r"floor $\sigma_F^2 = %.4f$" % s["floor_exact"],
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    finish(fig, ax, "fig1_simulation.pdf", "cross-section size $N$ (log)",
           r"$\mathrm{Var}(\widehat{IC}_t)$ (log)",
           "Simulation: IC variance levels off at the floor $\\sigma_F^2$", ncol=2)


def fig2_breadth(e):
    c, N = e["curves"], np.array(e["curves"]["sizes"], float)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for key, col, lab in [("mom12_1", BLUE, "12$-$1 momentum"), ("rev1", ORANGE, "1-month reversal"),
                          ("random", AQUA, "random signal (control)")]:
        ax.plot(N, c[key], "o-", color=col, ms=5, mec="white", mew=0.8, label=lab)
    ax.plot(N, c["iid"], ls="--", color=INK, lw=1.3, label=r"classical $1/\sqrt{N-1}$")
    for key, col, lab, i, dy in [("mom12_1", BLUE, "12$-$1 momentum", -1, 8),
                                 ("rev1", ORANGE, "1-month reversal", -1, -14),
                                 ("random", AQUA, "random signal (control)", 2, 9),
                                 ("iid", INK, r"classical $1/\sqrt{N-1}$", 3, -20)]:
        ax.annotate(lab, (N[i], c[key][i]), xytext=(0, dy), textcoords="offset points",
                    color=col, fontsize=8, ha="right" if i in (-1, 3) else "center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(45, 1450)
    finish(fig, ax, "fig2_breadth.pdf", "cross-section size $N$ (log)",
           r"sd$(\widehat{IC}_t)$ (log)",
           "Real data: IC volatility against cross-section size")


def fig3_nulls(e):
    m = e["mom12_1"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    grid = np.linspace(-0.55, 0.55, 400)
    iid_sd = 1 / np.sqrt(m["median_N"] - 1)
    ax.hist(m["series"], bins=34, density=True, color=BLUE, alpha=0.5,
            label="realised IC (sd %.3f)" % m["sd"])
    ax.hist(m["placebo_series"], bins=60, density=True, histtype="step", color=ORANGE, lw=1.6,
            label="misaligned-month placebo (sd %.3f)" % m["placebo_sd"])
    ax.plot(grid, norm.pdf(grid, 0, iid_sd), color=INK, lw=1.5,
            label=r"classical null, $N=%d$ (sd %.3f)" % (m["median_N"], iid_sd))
    ax.set_xlim(-0.55, 0.55)
    ax.legend(loc="upper right", fontsize=8)
    finish(fig, ax, "fig3_nulls.pdf", r"$\widehat{IC}_t$", "density",
           "12$-$1 momentum: realised IC, placebo and classical null")


def fig4_series(e):
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 4.4), sharex=True)
    for ax, (key, col, lab) in zip(axes, [("mom12_1", BLUE, "12$-$1 momentum"),
                                          ("rev1", ORANGE, "1-month reversal")]):
        v = np.array(e[key]["series"])
        x = np.array(e[key]["t_index"])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.bar(x, v, width=1.0, color=col, alpha=0.45, linewidth=0)
        roll = np.convolve(v, np.ones(24) / 24, mode="valid")
        ax.plot(x[23:], roll, color=col, lw=1.8)
        ax.axhline(v.mean(), color=INK, ls="--", lw=1.1)
        ax.set_title("%s:  mean %.4f, sd %.4f, t = %.2f" % (lab, e[key]["mean"], e[key]["sd"],
                                                            e[key]["t_iid"]), loc="left", fontsize=9)
        ax.set_ylabel(r"$\widehat{IC}_t$")
    yr0 = int(e["months"][0][:4])
    n = len(e["rev1"]["t_index"])
    axes[1].set_xticks(np.arange(0, n, 60))
    axes[1].set_xticklabels([str(yr0 + i * 5) for i in range(len(np.arange(0, n, 60)))])
    axes[1].set_xlabel("")
    fig.suptitle("Monthly rank IC, %s to %s (24-month rolling mean in bold)"
                 % (e["months"][0], e["months"][1]), x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig4_series.pdf", bbox_inches="tight", metadata=NODATE)
    plt.close(fig)
    print("   fig4_series.pdf")


def fig5_power(e):
    s_ic = e["implications"]["sigma_ic"]
    T = np.arange(12, 721)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for rho, col in [(0.05, AQUA), (0.03, BLUE), (0.02, ORANGE)]:
        p = ic.power(rho, s_ic / np.sqrt(T))
        ax.plot(T / 12, p, color=col, label="true IC = %.2f" % rho)
    ax.axhline(0.8, color=INK, ls="--", lw=1.1)
    ax.annotate("80% power", (44, 0.75), color=INK, fontsize=8)
    for xv, lab in [(25, "this panel (25 yr)"), (57, "post-1963 CRSP (57 yr)")]:
        ax.axvline(xv, color=MUTED, lw=1.0, ls=":")
        ax.text(xv - 0.9, 0.35, lab, rotation=90, va="center", ha="right", color=MUTED, fontsize=7.5)
    ax.set_xlim(1, 63)
    ax.set_ylim(0, 1.02)
    finish(fig, ax, "fig5_power.pdf", "sample length (years of monthly data)",
           "power to reject IC = 0 at 5%",
           r"Power at the measured $\sigma_{IC} = %.3f$" % s_ic, ncol=3)


def fig6_typem(e):
    s_ic = e["implications"]["sigma_ic"]
    T = np.arange(12, 721)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for z, col, lab in [(1.96, BLUE, "single test, $t>1.96$"), (3.00, ORANGE, "$t>3.00$ hurdle")]:
        ex = np.array([ic.type_m(0.03, s_ic / np.sqrt(t), z=z) for t in T])
        ax.plot(T / 12, ex, color=col, label=lab)
        j = int(np.argmax(T / 12 > 6))
        ax.annotate(lab, (T[j] / 12, ex[j]), xytext=(7, 3), textcoords="offset points",
                    color=col, fontsize=8, va="bottom")
    ax.axhline(1.0, color=INK, ls="--", lw=1.1)
    ax.annotate("no exaggeration", (40, 0.95), color=INK, fontsize=8, va="top")
    ax.set_xlim(1, 60)
    ax.set_ylim(bottom=0.7)
    finish(fig, ax, "fig6_typem.pdf", "sample length (years of monthly data)",
           r"$E[\,|\widehat{IC}|\ \mid\ \mathrm{significant}\,]\ /\ IC$",
           "Winner's curse: a true IC of 0.03, published only when significant", ncol=2)


if __name__ == "__main__":
    r = ic.get()
    print("figures ->", OUT)
    fig1_simulation(r["sim"])
    for f in (fig2_breadth, fig3_nulls, fig4_series, fig5_power, fig6_typem):
        f(r["emp"])
