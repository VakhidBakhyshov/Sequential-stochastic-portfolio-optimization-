# -*- coding: utf-8 -*-
"""Score the production policy's exposure dials against the KNOWN law of a synthetic run.

Usage (from the package root):  python -m extensions.synthetic_engine.evaluate --law gbm | regime
Reads results/known_law_<law>/ (production outputs + synthetic inputs + truth) and writes
results/known_law_<law>/eval/: monthly signal frame, matched-exposure rule comparison, forecast-error
moments against the true next-month risk, detection lags and false alarms, third-signal admission
test, oracle tracking, execution identities, controller bounds, and a summary JSON.

Conventions follow the paper: monthly rows are dated by the NEXT rebalance (the month they realize);
the dial dated d1 was formed at d0 from information through d0-1. The slow dial is the centred
scenario CVaR at the reading level 0.95 normalized by its expanding median (overlay_arms.build);
the fast dial targets 12% annual volatility from the 21-day realized volatility of the target book."""
from __future__ import annotations
import argparse, json, sys, io
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import overlay_arms as oa
from extensions.synthetic_engine.synth_market import TRADING_DAYS

RF, KMIN, KMAX, TAU, BETA = 0.02, 0.30, 1.00, 0.12, 0.95
QB = norm.pdf(norm.ppf(BETA)) / (1.0 - BETA)          # Gaussian CVaR / sigma at beta = 0.95 (2.0627)
NBOOT, SEED = 5000, 42


def apply_k(k, r): return k * r + (1 - k) * RF / 12


def matched(k: pd.Series, target: float) -> pd.Series:
    k = k.clip(KMIN, KMAX); return (k * (target / k.mean())).clip(KMIN, KMAX)


def paired(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float); n = len(a); ws = []; wd = []
    for _ in range(NBOOT):
        i = rng.integers(0, n, n); ws.append(oa.ann_sharpe(b[i]) > oa.ann_sharpe(a[i])); wd.append(oa.max_dd(b[i]) > oa.max_dd(a[i]))
    return float(np.mean(ws)), float(np.mean(wd))


def load_truth(folder: Path):
    tr = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    state = pd.read_csv(folder / "regime_state.csv", index_col=0, parse_dates=True)["state"]
    sig = np.array(tr["sigma_annual"]); cc = np.array(tr["corr_calm"]); cs = np.array(tr["corr_stress"])
    vmult = pd.Series(tr["vol_mult_daily"], index=state.index)          # stress multiplier x persistent cycle, by day
    tickers = [f"SYN{i:02d}" for i in range(len(sig))]
    def cov_daily(day):
        s = sig * float(vmult.loc[day]); c = cs if int(state.loc[day]) == 1 else cc
        return np.outer(s, s) * c / TRADING_DAYS
    return state, tickers, cov_daily


def book_true_vol(w: pd.Series, tickers, cov_daily, days) -> np.ndarray:
    """Annualized true conditional volatility of book w on each day in `days`."""
    wv = w.reindex(tickers).fillna(0.0).to_numpy()
    return np.array([np.sqrt(max(wv @ cov_daily(d) @ wv, 0.0) * TRADING_DAYS) for d in days])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--law", default="regime", choices=["gbm", "regime"]); a = ap.parse_args()
    folder = ROOT / "results" / f"known_law_{a.law}"; out = folder / "eval"; out.mkdir(exist_ok=True)
    oa.DAILY = folder / "synthetic_daily_returns.csv"
    arms, ks = oa.build(folder)                                 # base/fast/slow/combined, dials indexed by d1
    r_m = arms["base"]; idx = r_m.index; dates_m = list(idx)
    fc = pd.read_csv(folder / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"]); fc = fc.set_index("date").reindex(idx)
    W = oa.load_weights(folder, "weights"); reb = list(W.keys())
    dly = oa.daily_simple(); state, tickers, cov_daily = load_truth(folder)
    pdaily = oa.book_daily(W, dly); vol21 = pdaily.rolling(21, min_periods=10).std() * np.sqrt(TRADING_DAYS)
    rows = []
    for d1 in dates_m:
        d0 = reb[reb.index(d1) - 1] if d1 in reb else max(d for d in reb if d < d1)
        w = W[d0]; days = dly.index[(dly.index > d0) & (dly.index <= d1)]
        prev_days = dly.index[dly.index < d0]; d0m1 = prev_days[-1]
        tv_next = book_true_vol(w, tickers, cov_daily, days); tv_now = book_true_vol(w, tickers, cov_daily, [d0m1])[0]
        st_month = float(state.reindex(days).mean()); st_now = int(state.loc[d0m1])
        st_prev_window = float(state.loc[(state.index >= d0m1 - pd.Timedelta(days=31)) & (state.index <= d0m1)].mean())
        rn = pdaily.reindex(days).dropna(); real_next = float(rn.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rn) > 5 else np.nan
        rows.append(dict(date=d1, d0=d0, kF=float(ks["fast"].loc[d1]), kS=float(ks["slow"].loc[d1]), kC=float(ks["combined"].loc[d1]),
                         r_base=float(r_m.loc[d1]), fast_vol=float(vol21.asof(d0m1)), slow_cvar=float(fc.loc[d1, "cvar_model"]),
                         slow_vol_equiv=float(fc.loc[d1, "cvar_model"]) / QB * np.sqrt(TRADING_DAYS), model_vol=float(fc.loc[d1, "vol_model"]) * np.sqrt(TRADING_DAYS),
                         oracle_now=tv_now, oracle_next=float(np.sqrt(np.mean(tv_next ** 2))), oracle_cvar_now=QB * tv_now / np.sqrt(TRADING_DAYS),
                         realized_next=real_next, stress_share_next=st_month, stress_now=st_now, calm_window=float(st_prev_window == 0.0 and st_month == 0.0),
                         n_held=int((w > 1e-9).sum()), beta_t=float(fc.loc[d1, "beta_t"]), lambda_t=float(fc.loc[d1, "lambda_t"]),
                         binding=bool(fc.loc[d1, "lp_binding"]), shadow=float(fc.loc[d1, "lp_shadow_price"]), alpha_exec=float(fc.loc[d1, "alpha_exec"]),
                         turn_target=float(fc.loc[d1, "turnover_target_l1"]), turn_exec=float(fc.loc[d1, "turnover_exec_l1"])))
    M = pd.DataFrame(rows).set_index("date")
    # ---- third signals: an incremental one (noisy preview of the coming month's true risk) and a redundant one (noisy copy of the fast dial)
    rng = np.random.default_rng(SEED)
    M["k_inc"] = np.clip(TAU / (M["oracle_next"] * np.exp(rng.normal(0.0, 0.15, len(M)))), KMIN, KMAX)
    M["k_red"] = np.clip(M["kF"] * np.exp(rng.normal(0.0, 0.10, len(M))), KMIN, KMAX)
    M["k3_inc"] = np.clip((M["kF"] * M["kS"] * M["k_inc"]) ** (1 / 3), KMIN, KMAX)
    M["k3_red"] = np.clip((M["kF"] * M["kS"] * M["k_red"]) ** (1 / 3), KMIN, KMAX)
    M["k_geo"] = np.clip(np.sqrt(M["kF"] * M["kS"]), KMIN, KMAX); M["k_min"] = np.minimum(M["kF"], M["kS"]); M["k_blend"] = (0.5 * M["kF"] + 0.5 * M["kS"]).clip(KMIN, KMAX)
    M.to_csv(out / "monthly_signals.csv")
    summary = {"law": a.law, "months": int(len(M)), "mean_held": float(M["n_held"].mean())}

    v = M[["slow_cvar", "oracle_cvar_now", "slow_vol_equiv", "oracle_now", "model_vol", "fast_vol", "oracle_next", "realized_next"]].dropna()
    summary["oracle"] = dict(corr_slow_oracle_now=float(v["slow_cvar"].corr(v["oracle_cvar_now"])), rel_bias_slow=float((v["slow_cvar"] / v["oracle_cvar_now"] - 1).mean()),
                             corr_slow_oracle_next=float(v["slow_vol_equiv"].corr(v["oracle_next"])), corr_fast_oracle_next=float(v["fast_vol"].corr(v["oracle_next"])),
                             corr_fast_now=float(v["fast_vol"].corr(v["oracle_now"])), corr_realized_oracle_next=float(v["realized_next"].corr(v["oracle_next"])),
                             shape_ratio_mean=float((M["slow_cvar"] / (M["model_vol"] / np.sqrt(TRADING_DAYS))).mean()), shape_ratio_sd=float((M["slow_cvar"] / (M["model_vol"] / np.sqrt(TRADING_DAYS))).std()),
                             corr_kF_kS=float(M["kF"].corr(M["kS"])))
    # ---- 2) forecast-error moments against the true next-month risk (log space = the paper's Proposition; levels as a check)
    def moments(y, F, S):
        eF, eS = y - F, y - S; vF, vS, c = float(np.mean(eF ** 2)), float(np.mean(eS ** 2)), float(np.mean(eF * eS))
        den = vF + vS - 2 * c; al = float((vS - c) / den) if den > 1e-12 else np.nan
        mse_star = float(np.mean((y - (al * F + (1 - al) * S)) ** 2)) if np.isfinite(al) else np.nan
        rho = c / np.sqrt(vF * vS); return dict(v_F=vF, v_S=vS, c=c, rho=float(rho), alpha_star_on_fast=al, gain=bool(c < min(vF, vS)),
                                                 mse_best_single=float(min(vF, vS)), mse_optimal=mse_star, bates_bound=float(min(np.sqrt(vF / vS), np.sqrt(vS / vF))))
    summary["errors_log"] = moments(np.log(v["oracle_next"]), np.log(v["fast_vol"]), np.log(v["slow_vol_equiv"]))
    summary["errors_level"] = moments(v["oracle_next"].to_numpy(), v["fast_vol"].to_numpy(), v["slow_vol_equiv"].to_numpy())
    # ---- 3) admission test (Frisch-Waugh-Lovell): incremental vs redundant third signal
    y = np.log(v["oracle_next"].to_numpy()); X = np.column_stack([np.ones(len(y)), np.log(v["fast_vol"]), np.log(v["slow_vol_equiv"])])
    def resid(z): b, *_ = np.linalg.lstsq(X, z, rcond=None); return z - X @ b
    ry = resid(y); mse_pair = float(np.mean(ry ** 2)); adm = {}
    cand = {"incremental": np.log(TAU / M.loc[v.index, "k_inc"].to_numpy()), "redundant": np.log(TAU / M.loc[v.index, "k_red"].to_numpy())}
    for name, cvec in cand.items():
        rc = resid(cvec); formula = float(np.cov(ry, rc, ddof=0)[0, 1] ** 2 / max(np.var(rc), 1e-18))
        Xc = np.column_stack([X, cvec]); b, *_ = np.linalg.lstsq(Xc, y, rcond=None); mse_full = float(np.mean((y - Xc @ b) ** 2))
        adm[name] = dict(mse_pair=mse_pair, mse_with_candidate=mse_full, measured_reduction=mse_pair - mse_full, formula_reduction=formula,
                         identity_residual=abs((mse_pair - mse_full) - formula), share_of_pair_mse=(mse_pair - mse_full) / mse_pair)
    summary["admission"] = adm
    # ---- 4) detection lags and false alarms (regime law only)
    det = []
    if a.law == "regime":
        st = state; ch = st.diff().fillna(0); entries = [d for d in st.index[ch == 1] if d > M.index[0]]; exits = list(st.index[ch == -1])
        for e in entries:
            after = M.index[M.index > e]                      # months realized after the entry
            rec = {"entry": e.strftime("%Y-%m-%d")}
            ex = [x for x in exits if x > e]; rec["exit"] = ex[0].strftime("%Y-%m-%d") if ex else "end"
            for col in ["kF", "kS", "kC", "k3_inc", "k_geo"]:
                hit = [i for i, d in enumerate(after) if M.loc[d, col] < 0.8]; rec[f"lag_{col}"] = int(hit[0]) if hit else np.nan
                rec[f"min_{col}"] = float(M.loc[after[:6], col].min()) if len(after) else np.nan
            det.append(rec)
        D = pd.DataFrame(det); D.to_csv(out / "detection.csv", index=False)
        calm = M[M["calm_window"] == 1.0]
        summary["detection"] = {"n_entries": int(len(D)), **{f"median_lag_{c}": float(D[f"lag_{c}"].median()) for c in ["kF", "kS", "kC", "k3_inc", "k_geo"]},
                                **{f"hit_rate_{c}": float(D[f"lag_{c}"].notna().mean()) for c in ["kF", "kS", "kC"]},
                                "calm_months": int(len(calm)), **{f"false_alarm_{c}": float((calm[c] < 0.8).mean()) for c in ["kF", "kS", "kC"]},
                                "stress_months": int((M["stress_share_next"] > 0.5).sum()),
                                **{f"mean_k_in_stress_{c}": float(M.loc[M["stress_share_next"] > 0.5, c].mean()) for c in ["kF", "kS", "kC"]},
                                **{f"mean_k_in_calm_{c}": float(calm[c].mean()) for c in ["kF", "kS", "kC"]}}
    # ---- 5) specificity under the constant law / exposure cost
    summary["specificity"] = {f"share_months_{c}_below_1": float((M[c] < 0.999).mean()) for c in ["kF", "kS", "kC"]}
    summary["specificity"].update({f"share_months_{c}_below_0.9": float((M[c] < 0.9).mean()) for c in ["kF", "kS", "kC"]})
    summary["specificity"].update({f"mean_{c}": float(M[c].mean()) for c in ["kF", "kS", "kC"]})
    # ---- 6) matched-exposure comparison of the fixed rules and the three-signal aggregates
    rng = np.random.default_rng(SEED); kp = M["kC"]; target = float(kp.mean()); rule_rows = []
    rules = {"product": kp, "minimum": M["k_min"], "geometric mean": M["k_geo"], "convex blend 0.5": M["k_blend"], "fast only": M["kF"], "slow only": M["kS"],
             "3-signal geometric (incremental)": M["k3_inc"], "3-signal geometric (redundant)": M["k3_red"], "3-signal product (incremental)": (kp * M["k_inc"]).clip(KMIN, KMAX)}
    r_prod = apply_k(matched(kp, target), M["r_base"])
    for name, k in rules.items():
        for mode, kk in [("matched", matched(k, target)), ("raw", k.clip(KMIN, KMAX))]:
            r = apply_k(kk, M["r_base"]); m = oa.metrics(r.values); pS, pD = paired(r_prod.values, r.values, rng) if name != "product" or mode == "raw" else (np.nan, np.nan)
            rule_rows.append({"rule": name, "mode": mode, "mean_k": float(kk.mean()), **{k_: round(v_, 4) for k_, v_ in m.items()}, "P_sharpe_vs_product": pS, "P_dd_vs_product": pD})
    base_m = oa.metrics(M["r_base"].values); rule_rows.append({"rule": "unmanaged base", "mode": "raw", "mean_k": 1.0, **{k_: round(v_, 4) for k_, v_ in base_m.items()}})
    R = pd.DataFrame(rule_rows); R.to_csv(out / "rules_matched_exposure.csv", index=False)
    # ---- 7) execution identities, controller bounds, LP diagnostics
    resid_exec = (M["turn_exec"] - M["alpha_exec"] * M["turn_target"]).abs().max()
    dh = pd.read_csv(folder / "dynamic_parameter_history.csv")
    summary["identities"] = dict(max_execution_identity_residual=float(resid_exec), beta_min=float(dh["confidence_level"].min()), beta_max=float(dh["confidence_level"].max()),
                                 lambda_min=float(dh["turnover_penalty"].min()), lambda_max=float(dh["turnover_penalty"].max()), binding_share=float(M["binding"].mean()),
                                 median_shadow_when_binding=float(M.loc[M["binding"], "shadow"].median()) if M["binding"].any() else np.nan, mean_alpha_exec=float(M["alpha_exec"].mean()))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float)); print(R.round(3).to_string())


if __name__ == "__main__":
    main()
