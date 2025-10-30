
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replicate (statsmodels fetch): AR(1)-mixture sieve–Whittle + closed-form EM on U.S. unemployment
------------------------------------------------------------------------------------------------

This script fetches the U.S. macrodata bundle from statsmodels and reproduces the two real-data
applications:
  (A) ΔUNEMP (quarterly change in unemployment) – preferred for forecasting;
  (B) UNEMP level (for completeness).

Dependencies:
  numpy, pandas, matplotlib, statsmodels
  + your two modules in the same folder:
      ar1_mixture_sieve_whittle_em.py   (sieve–Whittle + spectral atoms & periodogram)
      mixture_ar1_sim.py                (Kalman filter + EM closed-form M-step)

Outputs (created under --outdir, default current folder):
  - <series>_series_split.png
  - <series>_mixing_weights_grid.png
  - <series>_spectra_compare.png
  - <series>_em_loglik_path.png
  - <series>_results_summary.csv
  - <series>_mixture_params.json

Example:
  python replicate_mixture_ar1_statsmodels.py --series dunemp --outdir results
"""
import os, math, json, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
import importlib, sys, statsmodels.api as sm

# Import local helper modules shipped with the paper
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
impl = importlib.import_module("ar1_mixture_sieve_whittle_em")
mixsim = importlib.import_module("mixture_ar1_sim")

# ------------------------------- Data loading --------------------------------

def load_series_from_statsmodels(series: str):
    assert series in {"dunemp", "unemp"}
    df = sm.datasets.macrodata.load_pandas().data.copy()
    # Construct ΔUNEMP and a date label
    df["dunemp"] = df["unemp"].astype(float).diff()
    df["date"] = df["year"].astype(int).astype(str) + "Q" + df["quarter"].astype(int).astype(str)

    if series == "dunemp":
        s = df[["date", "dunemp"]].dropna().reset_index(drop=True)
        y = s["dunemp"].values.astype(float)
    else:
        s = df[["date", "unemp"]].reset_index(drop=True)
        y = s["unemp"].values.astype(float)

    # 80/20 split (same choice as in the paper’s demo)
    split = int(round(0.8 * len(y)))
    y_train_raw, y_test_raw = y[:split], y[split:]

    # Standardize using TRAIN statistics (unit-variance normalization used by the sieve)
    mu = y_train_raw.mean()
    sd = y_train_raw.std(ddof=0)
    if sd <= 0:
        raise RuntimeError("Zero variance on training set; check the data.")
    y_train = (y_train_raw - mu) / sd
    y_test  = (y_test_raw  - mu) / sd

    return {
        "dates": s["date"].values,
        "split": split,
        "y_train": y_train,
        "y_test": y_test,
        "train_mean": float(mu),
        "train_std": float(sd)
    }

# ------------------------------ Benchmarks -----------------------------------

def toeplitz(c):
    c = np.asarray(c, dtype=float); n = len(c)
    T = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            T[i, j] = c[abs(i - j)]
    return T

def acovf(x, nlags):
    x = np.asarray(x, dtype=float); x = x - x.mean(); n = len(x)
    return np.array([np.dot(x[:n-k], x[k:]) / n for k in range(nlags+1)], dtype=float)

def yule_walker_bic(y, pmax=12):
    y = np.asarray(y, dtype=float); best=None; r=acovf(y, pmax)
    for p in range(1, pmax+1):
        R = toeplitz(r[:p]); phi = np.linalg.solve(R, r[1:p+1])
        # Residuals (one-step in-sample to get variance proxy for predictive score)
        resid = []
        for t in range(p, len(y)):
            resid.append(y[t] - np.dot(phi, y[t-p:t][::-1]))
        resid = np.array(resid, dtype=float)
        s2hat = float(np.mean(resid**2))
        bic = len(resid) * np.log(max(s2hat, 1e-12)) + p * np.log(len(resid))
        if (best is None) or (bic < best["bic"]):
            best = {"p": p, "phi": phi, "sigma2": float(max(s2hat, 1e-12)), "bic": float(bic)}
    return best

def ar_forecast_errors(y_train, y_test, phi):
    p = len(phi); y_full = np.concatenate([y_train, y_test]); start = len(y_train)
    errs = []
    for t in range(start, len(y_full)):
        prev = y_full[t-p:t][::-1]
        if len(prev) < p: prev = np.pad(prev, (0, p-len(prev)))
        yhat = float(np.dot(phi, prev)); errs.append(y_full[t] - yhat)
    return np.array(errs, dtype=float)

def ar_spectrum_2pi(phi, sigma2, omegas):
    p = len(phi)
    re = 1.0 - np.sum([phi[i]*np.cos((i+1)*omegas) for i in range(p)], axis=0)
    im = - np.sum([phi[i]*np.sin((i+1)*omegas) for i in range(p)], axis=0)
    denom = re**2 + im**2
    return sigma2/np.maximum(denom, 1e-12)

def daniell_smoother(x, m=5):
    k = np.ones(2*m+1, dtype=float)/(2*m+1)
    return np.convolve(x, k, mode="same")

def dm_test_sqerr(e1, e2):
    # Diebold–Mariano test on squared-error loss
    d = e1**2 - e2**2
    T = len(d); dbar = float(np.mean(d))
    bw = max(1, int(4 * (T/100.0)**(2.0/9.0)))  # small NW bandwidth
    gamma0 = float(np.dot(d - dbar, d - dbar) / T)
    varNW = gamma0
    for l in range(1, bw+1):
        cov = float(np.dot(d[l:] - dbar, d[:-l] - dbar) / T)
        weight = 1.0 - l/(bw+1.0)
        varNW += 2.0 * weight * cov
    varNW = varNW / T
    from math import erf, sqrt
    DM = dbar / max(np.sqrt(varNW), 1e-16)
    p = 2.0 * (1.0 - 0.5*(1.0 + erf(abs(DM)/sqrt(2.0))))
    return DM, p, dbar

# ------------------------------ Pipeline -------------------------------------

def run_series(series: str, outdir: str):
    assert series in {"dunemp", "unemp"}
    meta = load_series_from_statsmodels(series)
    y_train, y_test = meta["y_train"], meta["y_test"]

    # Sieve–Whittle fit on training
    omegas, I = impl.periodogram(y_train)
    rho_max = 0.98
    grid = np.linspace(0.01, rho_max, int((rho_max-0.01)/0.005)+1)
    atoms = impl.spectral_atoms(grid, omegas, include_white=True)
    p_hat, Lval = impl.minimize_whittle_on_simplex(atoms, I, max_iter=800, step0=0.6, tol=1e-10)

    # Support localization (peak picking + barycentric poles)
    p_grid = p_hat[1:]
    Ksel = 2 if series == "dunemp" else 1
    min_sep = 0.02; eps = 0.04
    idx_sorted = np.argsort(-p_grid)
    chosen = []
    for j in idx_sorted:
        if all(abs(grid[j]-grid[k]) >= min_sep for k in chosen):
            chosen.append(j)
        if len(chosen) >= Ksel:
            break

    def barycentric_given_peaks(grid, p_grid, peak_idx, eps):
        rhos_hat = []; thetas_hat = []; used = np.zeros_like(p_grid, dtype=bool)
        for j in peak_idx:
            mask = np.abs(grid - grid[j]) <= eps
            mass = float(p_grid[mask].sum())
            if mass > 0:
                rhat = float(np.sum(p_grid[mask]*grid[mask]) / mass)
            else:
                rhat = float(grid[j])
            rhos_hat.append(rhat); thetas_hat.append(mass); used |= mask
        sigma2_init = float(p_hat[0] + p_grid[~used].sum())
        return np.array(rhos_hat), np.array(thetas_hat), sigma2_init

    rho_init, w_init_mass, sigma2_init = barycentric_given_peaks(grid, p_grid, chosen, eps=eps)
    scale = (w_init_mass.sum() + sigma2_init)
    w_init = w_init_mass/scale; sigma2_init = sigma2_init/scale

    # EM with closed-form M-step (sum constraint)
    def run_em(y, rhos, w0, s20, iters=600, tol=1e-12):
        w = np.array(w0, dtype=float).copy(); s2 = float(s20)
        ll_path = [mixsim.observed_loglik(y, rhos, w, s2)]
        for _ in range(iters):
            A, B, _, _ = mixsim.em_A_B_from_smoother(y, rhos, w, s2)
            w, s2 = mixsim.em_mstep_unit_sum(A, B, len(y), S=1.0)
            ll_path.append(mixsim.observed_loglik(y, rhos, w, s2))
            if abs(ll_path[-1] - ll_path[-2]) < tol * (1.0 + abs(ll_path[-2])):
                break
        return w, s2, np.array(ll_path)

    w_em, s2_em, ll_path = run_em(y_train, rho_init, w_init, sigma2_init)

    # AR(p) benchmark (Yule–Walker + BIC)
    best_ar = yule_walker_bic(y_train, pmax=12)
    errs_ar = ar_forecast_errors(y_train, y_test, best_ar["phi"])
    # Mixture predictive errors (KF innovations on test)
    # Condition the test predictions on the full training history
    y_full = np.concatenate([y_train, y_test])
    kf_full = mixsim.kalman_filter(y_full, rho_init, w_em, s2_em)
    errs_mix = kf_full.v[meta["split"]:]     # one‑step innovations on test, conditioned on training

    # Sum of one‑step predictive log densities on test:
    # log p(y_test | y_train) = loglik(full) − loglik(train)
    logscore_mix = (mixsim.observed_loglik(y_full, rho_init, w_em, s2_em)
                    - mixsim.observed_loglik(y_train, rho_init, w_em, s2_em))
    logscore_ar  = float(np.sum(-0.5*(np.log(2*np.pi*best_ar["sigma2"]) + (errs_ar**2)/best_ar["sigma2"])))

    # MSE
    mse_mix = float(np.mean(errs_mix**2))
    mse_ar  = float(np.mean(errs_ar**2))

    # DM test
    DM, pDM, dbar = dm_test_sqerr(errs_mix, errs_ar)

    # Spectral comparison (training)
    s_hat_tr = p_hat @ atoms
    s_daniell = 2*np.pi * daniell_smoother(I, m=5)
    s_ar_tr = ar_spectrum_2pi(best_ar["phi"], best_ar["sigma2"], omegas)

    # ----------------------------- Save outputs -----------------------------
    os.makedirs(outdir, exist_ok=True)

    # (1) series split plot from raw series (for date ticks, use statsmodels dates)
    raw_df = sm.datasets.macrodata.load_pandas().data.copy()
    if series == "dunemp":
        raw = raw_df["unemp"].astype(float).diff().dropna().values
    else:
        raw = raw_df["unemp"].astype(float).values
    split_plot = int(round(0.8 * len(raw)))
    plt.figure(figsize=(10,3))
    plt.plot(raw); plt.axvline(split_plot, linestyle="--")
    title = "ΔUNEMP (quarterly change)" if series=="dunemp" else "UNEMP (quarterly)"
    ylabel = "ΔUNEMP (pp)" if series=="dunemp" else "UNEMP (%)"
    plt.title(f"{title} – train/test split"); plt.xlabel("t"); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{series}_series_split.png"), dpi=150); plt.close()

    # (2) weights on grid + selected poles
    plt.figure(figsize=(10,3))
    plt.stem(np.r_[0.0, grid], p_hat, basefmt=" ")
    for j in chosen: plt.axvline(grid[j], linestyle="--")
    plt.title(f"{series.upper()}: estimated mixing measure on {{white}}∪grid (vertical = selected peaks)")
    plt.xlabel("ρ"); plt.ylabel("weight")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{series}_mixing_weights_grid.png"), dpi=150); plt.close()

    # (3) spectra comparison
    plt.figure(figsize=(10,4))
    plt.plot(omegas, s_hat_tr/(2*np.pi), label="Mixture f̂(ω)")
    plt.plot(omegas, s_ar_tr/(2*np.pi), label=f"AR({best_ar['p']}) f̂(ω)")
    plt.plot(omegas, s_daniell/(2*np.pi), label="Smoothed periodogram")
    plt.title(f"{series.upper()}: spectral density (training)")
    plt.xlabel("ω"); plt.ylabel("f(ω)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{series}_spectra_compare.png"), dpi=150); plt.close()

    # (4) EM log-likelihood path
    plt.figure(figsize=(8,3))
    plt.plot(np.arange(len(ll_path)), ll_path, marker="o")
    plt.title(f"{series.upper()}: EM with closed-form M-step (monotone)")
    plt.xlabel("EM iteration"); plt.ylabel("log-likelihood")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{series}_em_loglik_path.png"), dpi=150); plt.close()

    # CSV summary
    summary = pd.DataFrame({
        "Model": ["Mixture (AR(1) sum)", f"AR({best_ar['p']}) YW-BIC"],
        "Test log score (sum 1-step)": [logscore_mix, logscore_ar],
        "Test MSE (1-step)": [mse_mix, mse_ar],
        "K or p": [len(rho_init), int(best_ar["p"])],
        "DM_stat (sqerr, mix-AR)": [DM, np.nan],
        "DM_pvalue": [pDM, np.nan],
        "avg_loss_diff dbar (mix-AR)": [dbar, np.nan],
    })
    summary.to_csv(os.path.join(outdir, f"{series}_results_summary.csv"), index=False)

    # Params JSON
    params = {
        "rho_hat": [float(x) for x in rho_init],
        "w_em": [float(x) for x in w_em],
        "sigma2_em": float(s2_em),
        "train_ll_increase": float(ll_path[-1] - ll_path[0]),
        "selected_peak_positions_on_grid": [float(grid[j]) for j in chosen],
        "eps_localization": float(eps),
        "grid_rho_max": float(rho_max),
        "grid_step": float(grid[1] - grid[0]) if len(grid)>1 else None,
        "train_mean": meta["train_mean"],
        "train_std": meta["train_std"],
        "split_index": int(meta["split"])
    }
    with open(os.path.join(outdir, f"{series}_mixture_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Print a concise summary to stdout
    print(f"\n[{series}] EM Δ loglik (train): {params['train_ll_increase']:.6f}")
    print(f"[{series}] Test MSE: mixture={mse_mix:.6f}, AR({best_ar['p']})={mse_ar:.6f}")
    print(f"[{series}] Test logscore: mixture={logscore_mix:.6f}, AR({best_ar['p']})={logscore_ar:.6f}")
    print(f"[{series}] DM test on sq. error: DM={DM:.3f}, p={pDM:.3f} (negative favors mixture)")
    print(f"[{series}] Poles ρ̂: {rho_init}  weights: {w_em}  σ²: {s2_em:.6f}")

# --------------------------------- CLI ---------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, default="dunemp", choices=["dunemp", "unemp", "both"],
                        help="Which series to run")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for figures/CSVs")
    args = parser.parse_args()

    if args.series in ("dunemp", "both"):
        run_series("dunemp", args.outdir)
    if args.series in ("unemp", "both"):
        run_series("unemp", args.outdir)

if __name__ == "__main__":
    main()
