"""data/results.json -> paper/numbers.tex (LaTeX macros)"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def macros(r):
    e, s = r["emp"], r["sim"]
    m, v = e["mom12_1"], e["rev1"]
    im, fit = e["implications"], e["fit"]
    out = {
        "panelStart": e["months"][0], "panelEnd": e["months"][1],
        "nTickers": f"{e['n_tickers']:,}", "panelObs": f"{im['claimed_obs']:,.0f}",
        # realised IC series
        "momT": m["T"], "momMean": f"{m['mean']:.4f}", "momSd": f"{m['sd']:.4f}",
        "momTstat": f"{m['t_iid']:.2f}", "momTnw": f"{m['t_nw']:.2f}",
        "momAR": f"{m['ar1']:.2f}", "momKurt": f"{m['kurtosis']:.2f}",
        "momN": f"{m['median_N']:,.0f}", "momNeff": f"{m['n_eff']:.0f}",
        "momPlacebo": f"{m['placebo_sd']:.4f}",
        "momPlaceboRatio": f"{m['placebo_sd']*(m['median_N']-1)**0.5:.1f}",
        "revT": v["T"], "revMean": f"{v['mean']:.4f}", "revSd": f"{v['sd']:.4f}",
        "revTstat": f"{v['t_iid']:.2f}", "revTnw": f"{v['t_nw']:.2f}",
        "revAR": f"{v['ar1']:.2f}", "revNeff": f"{v['n_eff']:.0f}",
        "revN": f"{v['median_N']:,.0f}",
        "revPlacebo": f"{v['placebo_sd']:.4f}",
        # the breadth-curve fits
        "momFloor": f"{fit['mom12_1']['sigma_F']:.4f}", "momC": f"{fit['mom12_1']['c']:.3f}",
        "momCeil": f"{1/fit['mom12_1']['floor_var']:.0f}",
        "revFloor": f"{fit['rev1']['sigma_F']:.4f}", "revC": f"{fit['rev1']['c']:.3f}",
        "revCeil": f"{1/fit['rev1']['floor_var']:.0f}",
        "randFloorVar": f"{fit['random']['floor_var']:.6f}", "randC": f"{fit['random']['c']:.3f}",
        "momFloorVar": f"{fit['mom12_1']['floor_var']:.6f}",
        "momOvershoot": f"{100*(fit['mom12_1']['c']-1):.0f}",
        "neffOneK": f"{1/(fit['mom12_1']['floor_var']+1e-3):.1f}",
        "neffTwoK": f"{1/(fit['mom12_1']['floor_var']+5e-4):.1f}",
        "randMaxDev": f"{max(abs(a/b-1) for a, b in zip(e['curves']['random'], e['curves']['iid']))*100:.0f}",
        "iidNull": f"{(m['median_N']-1)**-0.5:.4f}",
        # robustness
        "cohortN": f"{e['cohort']['n']:,}", "cohortMean": f"{e['cohort']['mean']:.4f}",
        "cohortSd": f"{e['cohort']['sd']:.4f}",
        "cohortSdChange": f"{100*(e['cohort']['sd']/m['sd']-1):.0f}",
        "pearsonMean": f"{e['pearson_raw']['mean']:.4f}", "pearsonSd": f"{e['pearson_raw']['sd']:.4f}",
        # simulation
        "simTrueIC": f"{s['mean_ic']:.4f}", "simFloor": f"{s['floor_exact']:.6f}",
        "simFloorSd": f"{s['floor_exact']**0.5:.4f}",
        "simClosed": f"{s['floor_closed']:.6f}", "simNaive": f"{s['floor_naive']:.6f}",
        "simFit": f"{s['fit_floor']:.6f}", "simFitSe": f"{s['floor_fit_se']:.6f}",
        "simFitC": f"{s['fit_c']:.3f}", "simKappa": f"{s['kappa']:.3f}",
        "simFitCSe": f"{s['fit_c_se']:.3f}",
        "simKappaUErr": f"{100*(1-s['kappa_u_only']/s['kappa']):.0f}",
        "simClosedErr": f"{100*(s['floor_closed']/s['fit_floor']-1):.1f}",
        "simNaiveErr": f"{100*(s['floor_naive']/s['fit_floor']-1):.1f}",
        "simExactErr": f"{100*(s['floor_exact']/s['fit_floor']-1):.1f}",
        "simExactSig": f"{abs(s['floor_exact']-s['fit_floor'])/s['floor_fit_se']:.1f}",
        "simClosedSig": f"{abs(s['floor_closed']-s['fit_floor'])/s['floor_fit_se']:.1f}",
        "simNaiveSig": f"{abs(s['floor_naive']-s['fit_floor'])/s['floor_fit_se']:.1f}",
        # implications
        "sigmaIC": f"{im['sigma_ic']:.3f}",
        "effObs": f"{im['eff_obs']:,.0f}", "neededObs": f"{im['needed_obs']:,.0f}",
        "overstate": f"{im['claimed_obs']/im['eff_obs']:.0f}",
    }
    for k, v_ in im["required_T"].items():
        out[f"reqT{k}"] = f"{v_:.0f}"
        out[f"reqY{k}"] = f"{v_/12:.1f}"
    for k, v_ in im["mde"].items():
        out[f"mde{k}"] = f"{v_:.4f}"
    for k, v_ in im["type_m"].items():
        out[f"typem{k}"] = f"{v_:.2f}"
    for k, v_ in s["coverage"].items():
        out[f"cov{k}"] = f"{v_:.3f}"
    for k, v_ in s["power"].items():
        out[f"pwsim{k}"] = f"{v_[0]:.3f}"
        out[f"pwana{k}"] = f"{v_[1]:.3f}"
    return out


def texname(key):
    # macro names can't have digits
    digits = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
              "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    return "".join(digits.get(ch, ch) for ch in key if ch.isalnum())


def tables(r):
    e, s = r["emp"], r["sim"]
    sim = []
    for n, v in zip(s["sizes"], s["var"]):
        pred = s["floor_exact"] + s["kappa"] / n
        sim.append(r"%d & %.6f & %.6f & $%+.1f\%%$ \\" % (n, v, pred, 100 * (pred / v - 1)))
    c, sizes = e["curves"], e["curves"]["sizes"]
    breadth = [r"%d & %.4f & %.4f & %.4f & %.4f \\"
               % (n, c["mom12_1"][i], c["rev1"][i], c["random"][i], c["iid"][i])
               for i, n in enumerate(sizes)]
    fit = e["fit"]
    breadth.append(r"\midrule")
    breadth.append(r"fitted $\sF$ & %.4f & %.4f & $\approx 0$ & --- \\"
                   % (fit["mom12_1"]["sigma_F"], fit["rev1"]["sigma_F"]))
    breadth.append(r"fitted $c$ & %.3f & %.3f & %.3f & --- \\"
                   % (fit["mom12_1"]["c"], fit["rev1"]["c"], fit["random"]["c"]))
    return {"simrows": "\n".join(sim), "breadthrows": "\n".join(breadth)}


if __name__ == "__main__":
    r = json.loads((ROOT / "data" / "results.json").read_text())
    lines = ["% Generated by tools/make_numbers.py -- do not edit.",
             "% Every estimate in paper.tex comes from data/results.json via these macros."]
    for key, val in macros(r).items():
        # {,} so 1,552 has no extra space in math mode
        lines.append("\\newcommand{\\%s}{%s}" % (texname(key), str(val).replace(",", "{,}")))
    for key, val in tables(r).items():
        lines.append("\\newcommand{\\%s}{%%\n%s}" % (texname(key), val))
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"paper/numbers.tex: {len(lines)-2} macros")
