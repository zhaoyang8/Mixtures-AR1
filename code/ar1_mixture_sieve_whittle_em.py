
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AR(1) Mixture: Sieve–Whittle Estimation, Support Localization,
and Closed-Form EM Weight Update under a Unit-Variance Sum Constraint

This script provides a reference implementation to *empirically* verify the
paper's main claims via simulation:

  (i) Uniform Whittle LLN over the AR(1)+white sieve (fixed and shrinking boundary);
 (ii) Support localization of the mixing measure and *barycentric* pole estimation;
(iii) Closed-form EM M-step for (weights, white-noise variance) under a sum constraint,
      using a single dual scalar, with EM monotonicity.

It can be run as a module or directly as a script.
Dependencies: numpy, matplotlib (optional for plots), pandas (optional for tables).

Author: (you)
Date: October 2025

USAGE (examples):
  python ar1_mixture_sieve_whittle_em.py --run all --outdir results
  python ar1_mixture_sieve_whittle_em.py --run lln_fixed --outdir results
  python ar1_mixture_sieve_whittle_em.py --run lln_shrink --outdir results
  python ar1_mixture_sieve_whittle_em.py --run localization --outdir results

The script saves summary CSVs and (optionally) PNG plots in the working directory.

"""

import numpy as np
import math
import sys
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import csv
from pathlib import Path

def save_lln_results_csv(results, boundary: str, out_path: str):
    """
    Save a list[ExperimentResult] to CSV with one row per T.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["boundary","T","sup_abs_diff","mean_abs_diff","median_abs_diff","q90_abs_diff","q99_abs_diff"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "boundary": boundary,
                "T": r.T,
                "sup_abs_diff": r.sup_abs_diff,
                "mean_abs_diff": r.mean_abs_diff,
                "median_abs_diff": r.median_abs_diff,
                "q90_abs_diff": r.q90_abs_diff,
                "q99_abs_diff": r.q99_abs_diff,
            })

def save_localization_csv(out: dict, true_rhos: np.ndarray, true_w: np.ndarray, sigma2: float, prefix: str):
    """
    Save (i) a summary CSV of barycentric estimates and (ii) a CSV of sieve weights p_hat over the grid.
    Produces files: <prefix>.summary.csv and <prefix>.p_hat.csv
    """
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Summary (barycentric + white weight)
    with open(prefix_path.with_suffix(".summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T","boundary","Lval"])
        w.writerow([out["T"], out["boundary"], out["Lval"]])
        w.writerow([])
        w.writerow(["k","rho_true","w_true","rho_hat","theta_hat"])
        for k in range(len(true_rhos)):
            w.writerow([k, float(true_rhos[k]), float(true_w[k]), float(out["rho_hat"][k]), float(out["theta_hat"][k])])
        w.writerow([])
        w.writerow(["white_weight_hat","white_weight_true(sigma2)"])
        w.writerow([float(out["p_hat"][0]), float(sigma2)])

    # 2) Full sieve weights (white + each grid atom)
    with open(prefix_path.with_suffix(".p_hat.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["j","atom","rho","weight"])
        w.writerow([0, "white", "", float(out["p_hat"][0])])
        for j, rho in enumerate(out["grid_rhos"], start=1):
            w.writerow([j, "ar1", float(rho), float(out["p_hat"][j])])


# ------------------------- Utilities -------------------------

def set_seed(seed: int = 1234):
    np.random.seed(seed)

def unit_simplex_random(n: int) -> np.ndarray:
    """Draw a random point on the n-simplex (sum=1, all positive)."""
    x = np.random.exponential(scale=1.0, size=n)
    return x / x.sum()

# ------------------------- Spectral primitives -------------------------

def phi_ar1(rho: float, omega: np.ndarray) -> np.ndarray:
    """
    AR(1) spectral atom (scaled as s(omega) = sum_j p_j * phi_j):
      phi(rho, omega) = (1 - rho^2) / (1 + rho^2 - 2*rho*cos(omega))
    The "white" atom is phi_0(omega) = 1.
    """
    return (1.0 - rho**2) / (1.0 + rho**2 - 2.0 * rho * np.cos(omega))

def spectral_atoms(grid_rhos: np.ndarray, omegas: np.ndarray, include_white: bool = True) -> np.ndarray:
    """
    Return matrix A of shape (J, M) where:
      J = number of atoms (K_hat + 1 if include_white), M = number of frequencies
      rows: atoms j, columns: omega_m
    Ordering: j = 0 is the white-noise atom (all ones) if include_white.
    """
    M = len(omegas)
    if include_white:
        J = len(grid_rhos) + 1
        A = np.empty((J, M), dtype=float)
        A[0, :] = 1.0
        for j, rho in enumerate(grid_rhos, start=1):
            A[j, :] = phi_ar1(rho, omegas)
        return A
    else:
        J = len(grid_rhos)
        A = np.empty((J, M), dtype=float)
        for j, rho in enumerate(grid_rhos):
            A[j, :] = phi_ar1(rho, omegas)
        return A

def true_spectrum_s(omegas: np.ndarray, rhos: np.ndarray, weights: np.ndarray, sigma2: float) -> np.ndarray:
    """
    s*(omega) = 2*pi*f*(omega) in the paper's notation, under unit-variance scaling.
    Here we build it directly as a convex combination:
         s*(omega) = sum_k w_k * phi(rho_k, omega) + sigma2 * 1
    with sum(w) + sigma2 = 1.
    """
    s = np.full_like(omegas, fill_value=sigma2, dtype=float)
    for rho, w in zip(rhos, weights):
        s += w * phi_ar1(rho, omegas)
    return s

# ------------------------- Time-series simulation -------------------------

def simulate_ar1_mixture(T: int, rhos: np.ndarray, weights: np.ndarray, sigma2: float, burnin: int = 1000) -> np.ndarray:
    """
    Simulate y_t = sum_k alpha_{k,t} + u_t with
       alpha_{k,t+1} = rho_k * alpha_{k,t} + eta_{k,t},   eta_{k,t} ~ N(0, (1 - rho_k^2)*w_k)
       u_t ~ N(0, sigma2)
    All components independent. Stationary initialization for alpha_{k,1}.
    The variance of alpha_{k,t} is w_k by construction.
    """
    K = len(rhos)
    assert np.all(weights > 0) and sigma2 >= 0 and abs(weights.sum() + sigma2 - 1.0) < 1e-8
    # Simulate states with a burn-in for safety; stationary innovations
    total_T = T + burnin
    alphas = np.zeros((K, total_T), dtype=float)
    # Initialize alpha_1 ~ N(0, w_k)
    alphas[:, 0] = np.random.normal(loc=0.0, scale=np.sqrt(weights), size=K)
    for t in range(1, total_T):
        eps = np.random.normal(loc=0.0, scale=np.sqrt((1.0 - rhos**2) * weights))
        alphas[:, t] = rhos * alphas[:, t - 1] + eps
    y = alphas[:, burnin:].sum(axis=0)
    if sigma2 > 0:
        y += np.random.normal(loc=0.0, scale=np.sqrt(sigma2), size=T)
    return y

# ------------------------- Periodogram & Whittle objective -------------------------

def periodogram(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical (raw) periodogram at ω_j = 2π j / T, j = 1..M,
    with I_T(ω) = (1/(2πT)) |∑ y_t e^{-i ω t}|^2.

    Conventions (aligned with mixture_ar1_sim.rfft_periodogram_2pi):
      • Center y before FFT.
      • Drop DC (j=0) always.
      • If T is even, also drop the Nyquist term (j=T/2).
    """
    y = np.asarray(y, dtype=float)
    T = len(y)

    # NEW: center y to match mixture_ar1_sim
    y = y - y.mean()

    Y = np.fft.rfft(y)                   # includes 0 and (if T even) Nyquist
    freqs = np.fft.rfftfreq(T, d=1.0)    # cycles/sample
    omegas_full = 2.0 * np.pi * freqs

    # NEW: drop DC; drop Nyquist if T even
    if T % 2 == 0:
        idx = np.arange(1, len(Y) - 1)   # exclude 0 and Nyquist
    else:
        idx = np.arange(1, len(Y))       # exclude 0 only

    Ysel = Y[idx]
    omegas = omegas_full[idx]
    I = (1.0 / (2.0 * np.pi * T)) * (np.abs(Ysel) ** 2)
    return omegas, I


def whittle_objective_from_atoms(p: np.ndarray, atoms: np.ndarray, I: np.ndarray) -> float:
    """
    Discrete Whittle objective (normalized) using s(omega) = sum_j p_j * phi_j(omega).
    Here 'atoms' has shape (J, M), I has shape (M,).
    Value: L_T(p) = (1/M) sum_m [ log s_m + (2*pi)*I_m / s_m ].
    """
    s = p @ atoms  # shape (M,)
    # numerical guard
    s = np.maximum(s, 1e-12)
    val = np.mean(np.log(s) + (2.0 * np.pi) * I / s)
    return float(val)

def whittle_gradient_from_atoms(p: np.ndarray, atoms: np.ndarray, I: np.ndarray) -> np.ndarray:
    s = p @ atoms
    s = np.maximum(s, 1e-12)
    # grad_k = (1/M) Σ_m φ_{k,m}/s_m * (1 − (2π) I_m / s_m)
    temp = (atoms / s) * (1.0 - (2.0 * np.pi) * I / s)
    return temp.mean(axis=1)

def j_population_from_atoms(p: np.ndarray, atoms: np.ndarray, s_true: np.ndarray) -> float:
    """
    Discrete approximation to J(p) = (1/2π) ∫ [log f + f*/f] dω, expressed with s=2π f:
    J_d(p) ≈ (1/M) sum_m [ log s_m + s*_m / s_m ].
    """
    s = p @ atoms
    s = np.maximum(s, 1e-12)
    val = np.mean(np.log(s) + s_true / s)
    return float(val)

# ------------------------- Mirror descent on simplex -------------------------

def project_simplex_eg(p: np.ndarray) -> np.ndarray:
    """Normalize to the simplex (positive, sum to one)."""
    p = np.maximum(p, 1e-18)
    return p / p.sum()

def minimize_whittle_on_simplex(atoms: np.ndarray, I: np.ndarray, max_iter: int = 500,
                                step0: float = 0.5, verbose: bool = False, tol: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    Exponentiated Gradient (mirror descent) to minimize Whittle objective over p in the simplex.
    Returns (p_hat, L_T(p_hat)).
    """
    J, M = atoms.shape
    p = np.ones(J) / J  # uniform init
    prev = np.inf
    for t in range(1, max_iter + 1):
        g = whittle_gradient_from_atoms(p, atoms, I)
        eta = step0 / math.sqrt(t)  # diminishing stepsize
        # EG update
        p = p * np.exp(-eta * g)
        p = project_simplex_eg(p)
        if (t % 25 == 0) or (t == max_iter):
            val = whittle_objective_from_atoms(p, atoms, I)
            if verbose:
                print(f"  iter {t:4d}  L={val:.6f}")
            if abs(prev - val) < tol * (1.0 + abs(val)):
                break
            prev = val
    val = whittle_objective_from_atoms(p, atoms, I)
    return p, val

# ------------------------- Uniform LLN experiment helpers -------------------------

def uniform_lln_sup_deviation(omegas: np.ndarray, I: np.ndarray, atoms: np.ndarray,
                              s_true: np.ndarray, n_draws: int = 200,
                              include_minimizer: bool = True) -> Tuple[float, Dict[str, float]]:
    """
    Approximate sup_{p in sieve} | L_T(p) - J(p) | by:
      - drawing random p's on the simplex, evaluating |L - J|
      - including the sieve Whittle minimizer for additional stress
    Returns (sup_abs_diff, summary) where summary has mean/median etc.
    """
    diffs = []
    # Random draw evaluation
    for _ in range(n_draws):
        p = unit_simplex_random(atoms.shape[0])
        L = whittle_objective_from_atoms(p, atoms, I)
        Jd = j_population_from_atoms(p, atoms, s_true)
        diffs.append(abs(L - Jd))
    # Include minimizer point
    if include_minimizer:
        p_hat, _ = minimize_whittle_on_simplex(atoms, I, max_iter=300)
        L = whittle_objective_from_atoms(p_hat, atoms, I)
        Jd = j_population_from_atoms(p_hat, atoms, s_true)
        diffs.append(abs(L - Jd))
    diffs = np.array(diffs)
    summary = {
        "mean_abs": float(np.mean(diffs)),
        "median_abs": float(np.median(diffs)),
        "q90_abs": float(np.quantile(diffs, 0.9)),
        "q99_abs": float(np.quantile(diffs, 0.99)),
    }
    return float(np.max(diffs)), summary

# ------------------------- Support localization and barycentric -------------------------

def barycentric_localization(grid_rhos: np.ndarray, p_hat: np.ndarray, true_rhos: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each true pole rho_k*, collect sieve mass within |rho - rho_k*| <= eps
    and compute the barycentric estimate (weighted average) using that local mass.
    Returns (rho_hat_k, theta_hat_k) for each k.
    """
    K = len(true_rhos)
    # p_hat includes white atom at index 0
    rho_hat = np.zeros(K, dtype=float)
    theta_hat = np.zeros(K, dtype=float)
    for k, rstar in enumerate(true_rhos):
        idx = np.where(np.abs(grid_rhos - rstar) <= eps)[0]
        # shift by +1 for the white atom at 0
        weights_local = p_hat[idx + 1]
        mass = weights_local.sum()
        theta_hat[k] = mass
        if mass > 0:
            rho_hat[k] = np.sum(weights_local * grid_rhos[idx]) / mass
        else:
            # no mass captured -> return nearest grid point as fallback
            j = int(np.argmin(np.abs(grid_rhos - rstar)))
            rho_hat[k] = grid_rhos[j]
    return rho_hat, theta_hat

# ------------------------- State-space likelihood & EM -------------------------

@dataclass
class StateSpaceSpec:
    rhos: np.ndarray      # (K,)
    w: np.ndarray         # (K,)
    sigma2: float         # scalar
    # by construction sum(w)+sigma2=1

def kalman_filter_loglik(y: np.ndarray, rhos: np.ndarray, w: np.ndarray, sigma2: float) -> Tuple[float, Dict]:
    """
    Gaussian log-likelihood via standard Kalman filter for the system:
       alpha_{t+1} = T alpha_t + eta_t,  T=diag(rhos),  eta_t ~ N(0, Q) with Q=diag((1-rho^2)*w)
       y_t = Z alpha_t + eps_t,  Z = 1_K',  eps_t ~ N(0, H=sigma2)
    alpha_1 ~ N(0, P1=diag(w)) independent of disturbances.
    Returns (loglik, cache) where cache has arrays needed for smoothing.
    """
    Tlen = len(y)
    K = len(rhos)
    Tmat = np.diag(rhos)
    Q = np.diag((1.0 - rhos**2) * w)
    Z = np.ones((1, K), dtype=float)
    H = np.array([[sigma2]], dtype=float)

    # Initial state
    a = np.zeros((K,))  # mean
    P = np.diag(w).copy()

    v_list = []
    F_list = []
    a_list = [a.copy()]
    P_list = [P.copy()]
    Kt_list = []
    Lt_list = []

    loglik = 0.0
    for t in range(Tlen):
        # Predict observation
        yhat = float(Z @ a)
        v = y[t] - yhat
        F = float(Z @ P @ Z.T + H)  # scalar
        # Kalman gain
        Kt = (Tmat @ P @ Z.T) / F  # shape (K,1)
        # Update and advance
        a_next = Tmat @ a + (Kt.flatten()) * v
        L = Tmat - Kt @ Z  # KxK

        # Save
        v_list.append(v)
        F_list.append(F)
        a_list.append(a_next.copy())
        P_list.append(Tmat @ P @ L.T + Q)  # Joseph not required; standard recursion
        Kt_list.append(Kt.copy())
        Lt_list.append(L.copy())

        # Lik contribution
        loglik += -0.5 * (math.log(2.0 * math.pi) + math.log(F) + (v**2) / F)

        # Prepare next step
        a = a_next
        P = P_list[-1]

    cache = {
        "v": np.array(v_list),
        "F": np.array(F_list),
        "a": np.array(a_list),   # length T+1
        "P": np.array(P_list),   # length T+1
        "K": np.array(Kt_list),
        "L": np.array(Lt_list),
        "T": Tmat,
        "Q": Q,
        "Z": Z,
        "H": H,
    }
    return loglik, cache

def disturbance_smoother(y: np.ndarray, cache: Dict) -> Dict:
    """
    Durbin–Koopman disturbance smoother (scalar observation). Returns r,N arrays and
    smoothed disturbances’ expectations and variances.
    """
    v = cache["v"]
    F = cache["F"]
    a_list = cache["a"]
    P_list = cache["P"]
    Kt_list = cache["K"]
    Lt_list = cache["L"]
    Tmat = cache["T"]
    Q = cache["Q"]
    Z = cache["Z"]
    H = cache["H"]

    Tlen = len(v)
    K = Q.shape[0]

    r = np.zeros((Tlen + 1, K))   # r_T = 0
    N = np.zeros((Tlen + 1, K, K))

    # Backward pass
    for t in range(Tlen, 0, -1):
        # indices: in arrays, forward step appended element t corresponds to time t
        Zt = Z
        Ft = F[t-1]
        vt = v[t-1]
        Pt = P_list[t-1]
        Lt = Lt_list[t-1]

        r[t-1] = (Zt.T.flatten() * (vt / Ft)).flatten() + Lt.T @ r[t]
        N[t-1] = (Zt.T @ Zt) / Ft + Lt.T @ N[t] @ Lt

    # Smoothed disturbances
    eta_hat = np.zeros((Tlen, K))
    Var_eta = np.zeros((Tlen, K, K))
    for t in range(Tlen):
        eta_hat[t] = Q @ r[t+1]
        Var_eta[t] = Q - Q @ N[t+1] @ Q

    # Smoothed initial state alpha_1
    P1 = P_list[0]
    r0 = r[0]
    N0 = N[0]
    E_alpha1 = P1 @ r0    # mean is 0; smoothed mean contribution P1 r0
    Var_alpha1 = P1 - P1 @ N0 @ P1
    # Return also epsilon disturbances if H>0
    eps_hat = None
    Var_eps = None
    if float(H) > 0:
        eps_hat = np.zeros(Tlen)
        Var_eps = np.zeros(Tlen)
        for t in range(Tlen):
            Pt = P_list[t]
            Ft = F[t]
            vt = v[t]
            Lt = Lt_list[t]
            # scalar formulas (Durbin-Koopman)
            eps_hat[t] = float(H * (vt / Ft - (Z @ Pt @ r[t+1])))
            Var_eps[t] = float(H - (H**2) / Ft + H * (Z @ Pt @ N[t+1] @ Pt @ Z.T) * H)

    return {
        "r": r, "N": N,
        "eta_hat": eta_hat, "Var_eta": Var_eta,
        "E_alpha1": E_alpha1, "Var_alpha1": Var_alpha1,
        "eps_hat": eps_hat, "Var_eps": Var_eps
    }



# ------------------------- Experiments -------------------------

@dataclass
class ExperimentResult:
    T: int
    sup_abs_diff: float
    mean_abs_diff: float
    median_abs_diff: float
    q90_abs_diff: float
    q99_abs_diff: float

def run_uniform_lln_experiment(T_list: List[int], true_rhos: np.ndarray, true_w: np.ndarray, sigma2: float,
                               boundary: str = "fixed", delta_fixed: float = 0.02, draws: int = 200,
                               seed: int = 123, verbose: bool = False) -> List[ExperimentResult]:
    """
    For each T in T_list:
      - simulate y
      - build sieve grid according to 'boundary' (fixed or shrinking)
      - compute atoms, periodogram, true s*
      - approximate sup_p |L - J| over random p (and include minimizer)
    """
    set_seed(seed)
    results = []
    for T in T_list:
        y = simulate_ar1_mixture(T, true_rhos, true_w, sigma2, burnin=1000)
        omegas, I = periodogram(y)

        # Build sieve grid
        if boundary == "fixed":
            rho_max = 1.0 - delta_fixed
            grid = np.linspace(0.02, rho_max, int((rho_max-0.02)/0.005)+1)
        elif boundary == "shrinking":
            # Use delta_T that shrinks slowly: 1 / (1 + ln T)
            delta_T = 1.0 / (1.0 + math.log(T))
            rho_max = 1.0 - delta_T
            # Make Δ * δ_T^{-2} → 0 as T→∞ (Assumption 4(iv)).
            # Example: Δ = δ_T^2 / (1 + log T) ⇒ Δ * δ_T^{-2} = 1 / (1 + log T) → 0.
            mesh = (delta_T**2) / (1.0 + math.log(T))
            npts = max(50, int((rho_max - 0.02)/mesh) + 1)
            grid = np.linspace(0.02, rho_max, npts)
        else:
            raise ValueError("boundary must be 'fixed' or 'shrinking'")

        atoms = spectral_atoms(grid, omegas, include_white=True)
        s_true = true_spectrum_s(omegas, true_rhos, true_w, sigma2)

        sup_diff, summ = uniform_lln_sup_deviation(omegas, I, atoms, s_true, n_draws=draws, include_minimizer=True)
        res = ExperimentResult(T=T,
                               sup_abs_diff=sup_diff,
                               mean_abs_diff=summ["mean_abs"],
                               median_abs_diff=summ["median_abs"],
                               q90_abs_diff=summ["q90_abs"],
                               q99_abs_diff=summ["q99_abs"])
        if verbose:
            print(f"[{boundary}] T={T:6d}  sup|L-J|={sup_diff:.4g}  mean={summ['mean_abs']:.4g}  median={summ['median_abs']:.4g}")
        results.append(res)
    return results

def run_localization_experiment(T: int, true_rhos: np.ndarray, true_w: np.ndarray, sigma2: float,
                                eps: float, boundary: str = "fixed", delta_fixed: float = 0.02,
                                seed: int = 1234) -> Dict:
    """
    Single-run localization & barycentric estimate.
    Returns dict with p_hat, grid_rhos, and barycentric outputs.
    """
    set_seed(seed)
    y = simulate_ar1_mixture(T, true_rhos, true_w, sigma2, burnin=1000)
    omegas, I = periodogram(y)

    if boundary == "fixed":
        rho_max = 1.0 - delta_fixed
        grid = np.linspace(0.02, rho_max, int((rho_max-0.02)/0.005)+1)
    else:
        delta_T = 1.0 / (1.0 + math.log(T))
        rho_max = 1.0 - delta_T
        mesh = 0.5 * (delta_T**2)
        npts = max(50, int((rho_max - 0.02)/mesh) + 1)
        grid = np.linspace(0.02, rho_max, npts)

    atoms = spectral_atoms(grid, omegas, include_white=True)
    p_hat, Lval = minimize_whittle_on_simplex(atoms, I, max_iter=400)

    rho_hat, theta_hat = barycentric_localization(grid, p_hat, true_rhos, eps=eps)

    return {
        "T": T,
        "boundary": boundary,
        "grid_rhos": grid,
        "p_hat": p_hat,
        "rho_hat": rho_hat,
        "theta_hat": theta_hat,
        "Lval": Lval
    }


# ------------------------- CLI -------------------------

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="all",
                        help="Which experiment to run: all | lln_fixed | lln_shrink | localization")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save CSV outputs.")

    args = parser.parse_args()

    # True models for the two boundary regimes
    # (A) Fixed boundary: allow near-unit root but strictly inside 1 - delta_fixed
    true_rhos_fixed = np.array([0.20, 0.60, 0.95])
    true_w_fixed = np.array([0.45, 0.35, 0.10])
    sigma2_fixed = 0.10  # sum=1

    # (B) Shrinking boundary: pick rhos that remain <= 1 - delta_T for the T we consider
    # We use moderately persistent components but below typical delta_T for T up to ~1e5
    true_rhos_shrink = np.array([0.45, 0.60, 0.75])
    true_w_shrink = np.array([0.50, 0.30, 0.15])
    sigma2_shrink = 0.05  # sum=1

    if args.run in ("all", "lln_fixed"):
        T_list = [2048, 8192, 32768]
        res = run_uniform_lln_experiment(T_list, true_rhos_fixed, true_w_fixed, sigma2_fixed,
                                         boundary="fixed", delta_fixed=0.02, draws=250, seed=args.seed, verbose=True)
        save_lln_results_csv(res, "fixed", str(Path(args.outdir) / "lln_fixed.csv"))
    if args.run in ("all", "lln_shrink"):
        T_list = [2048, 8192, 32768]
        res = run_uniform_lln_experiment(T_list, true_rhos_shrink, true_w_shrink, sigma2_shrink,
                                         boundary="shrinking", draws=250, seed=args.seed, verbose=True)
        save_lln_results_csv(res, "shrinking", str(Path(args.outdir) / "lln_shrinking.csv"))
    if args.run in ("all", "localization"):
        # pick fixed-boundary case with wellseparated poles
        T = 16000
        eps = 0.05
        out = run_localization_experiment(T, true_rhos_fixed, true_w_fixed, sigma2_fixed,
                                          eps=eps, boundary="fixed", delta_fixed=0.02, seed=args.seed)
        print("\nLocalization (fixed boundary):")
        print(f"  barycentric rho_hat: {np.round(out['rho_hat'], 4)} vs true {np.round(true_rhos_fixed, 4)}")
        print(f"  local masses theta_hat: {np.round(out['theta_hat'], 4)} vs true {np.round(true_w_fixed, 4)}")
        print("  white-noise weight (sieve): {:.4f} (true {:.4f})".format(out["p_hat"][0], 0.10))
        save_localization_csv(out, true_rhos_fixed, true_w_fixed, sigma2_fixed, str(Path(args.outdir) / "localization_fixed"))


if __name__ == "__main__":
    cli()
