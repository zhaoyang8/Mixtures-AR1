#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixtures of AR(1) Components: Sieve–Whittle and Closed–Form EM M–Step (Simulation)

This script:
  1) Simulates a unit-variance AR(1) mixture with optional white noise;
  2) Fits a sieve–Whittle estimator on an AR(1)+white dictionary (with the correct 2π scaling);
  3) Shows localization via barycentric pole estimates;
  4) Runs EM for (w, σ^2) with the *closed-form* sum-constrained M-step and verifies
     monotone increase of the observed Gaussian log-likelihood;
  5) Plots spectra, mixing weights on the grid, and EM log-likelihood path.

No external dependencies beyond numpy and matplotlib.

Usage:
  python mixture_ar1_sim.py

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

# ---------------------- Core utilities ----------------------

def simulate_ar1_mixture(T: int,
                         rhos: np.ndarray,
                         weights: np.ndarray,
                         sigma2: float,
                         seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    K = len(rhos)
    alphas = np.zeros((K, T))
    # stationary init for each state: Var(alpha_k) = w_k
    for k in range(K):
        rho = rhos[k]
        w = weights[k]
        alphas[k, 0] = rng.normal(0.0, np.sqrt(w))
        for t in range(T - 1):
            eta = rng.normal(0.0, np.sqrt((1.0 - rho**2) * w))
            alphas[k, t+1] = rho * alphas[k, t] + eta
    y = np.sum(alphas, axis=0) + rng.normal(0.0, np.sqrt(sigma2), size=T)
    return y


def rfft_periodogram_2pi(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ω_j (j=1..M) and yscaled_j := 2π I_T(ω_j) = |DFT_j|^2 / T
    with I_T(ω_j) = (1/(2π T)) |Sum y_t e^{-i ω_j t}|^2, ω_j=2π j/T.
    """
    T = len(y)
    y = y - np.mean(y)
    Y = np.fft.rfft(y)
    nfreqs = len(Y)
    if T % 2 == 0:
        js = np.arange(1, nfreqs - 1)  # exclude 0 and Nyquist
    else:
        js = np.arange(1, nfreqs)
    w = 2.0 * np.pi * js / T
    yscaled = (np.abs(Y[js]) ** 2) / T   # equals 2π I_T(ω_j)
    return w, yscaled


def ar1_phi(rho: np.ndarray, wgrid: np.ndarray) -> np.ndarray:
    c = np.cos(wgrid)[:, None]
    rho = rho[None, :]
    num = 1.0 - rho**2
    den = 1.0 + rho**2 - 2.0 * rho * c
    return num / den


@dataclass
class SieveWhittleResult:
    p: np.ndarray
    grid: np.ndarray
    obj_path: List[float]
    converged: bool


def fit_sieve_whittle(y: np.ndarray,
                      rho_grid: np.ndarray,
                      max_iter: int = 300,
                      tol: float = 1e-9,
                      stepsize: float = 0.2,
                      backtrack: float = 0.5,
                      max_backtrack: int = 20) -> SieveWhittleResult:
    wgrid, yscaled = rfft_periodogram_2pi(y)
    M = len(wgrid)
    Phi = np.concatenate([np.ones((M, 1)), ar1_phi(rho_grid, wgrid)], axis=1)  # (M, J+1)
    J = Phi.shape[1]
    p = np.ones(J) / J
    obj_path = []

    def objective(pv: np.ndarray) -> float:
        s = Phi @ pv
        return float(np.mean(np.log(s) + yscaled / s))

    def grad(pv: np.ndarray) -> np.ndarray:
        s = Phi @ pv
        g_s = (1.0 / s) - (yscaled / (s * s))
        return np.mean(Phi * g_s[:, None], axis=0)

    last_obj = objective(p)
    obj_path.append(last_obj)
    converged = False

    for _ in range(max_iter):
        g = grad(p)
        step = stepsize
        accepted = False
        for _ in range(max_backtrack):
            p_trial = p * np.exp(-step * g)
            s = np.sum(p_trial)
            if s <= 0 or not np.isfinite(s):
                step *= backtrack
                continue
            p_trial /= s
            obj_trial = objective(p_trial)
            if obj_trial <= last_obj:
                accepted = True
                break
            step *= backtrack
        if not accepted:
            break
        p = p_trial
        obj_path.append(obj_trial)
        if abs(last_obj - obj_trial) < tol * (1.0 + abs(last_obj)):
            converged = True
            last_obj = obj_trial
            break
        last_obj = obj_trial

    return SieveWhittleResult(p=p, grid=rho_grid, obj_path=obj_path, converged=converged)


def spectral_from_mixture(p: np.ndarray, rho_grid: np.ndarray, wgrid: np.ndarray) -> np.ndarray:
    Phi = np.concatenate([np.ones((len(wgrid), 1)), ar1_phi(rho_grid, wgrid)], axis=1)
    return Phi @ p


def true_spectrum_2pi(rhos: np.ndarray, w: np.ndarray, sigma2: float, wgrid: np.ndarray) -> np.ndarray:
    if len(rhos) > 0:
        Phi = ar1_phi(rhos, wgrid)
        s = Phi @ w + sigma2 * np.ones(len(wgrid))
    else:
        s = sigma2 * np.ones(len(wgrid))
    return s


def barycentric_poles(p: np.ndarray, rho_grid: np.ndarray, true_rhos: np.ndarray, eps: float = 0.05):
    p_white = p[0]
    p_grid = p[1:]
    estimates = []
    used_mask = np.zeros_like(p_grid, dtype=bool)
    for rstar in true_rhos:
        idx = np.where(np.abs(rho_grid - rstar) <= eps)[0]
        mass = p_grid[idx].sum()
        if mass > 1e-12:
            rhat = (p_grid[idx] * rho_grid[idx]).sum() / mass
        else:
            rhat = np.nan
        estimates.append((float(rhat), float(mass)))
        used_mask[idx] = True
    outside_mass = float(p_grid[~used_mask].sum())
    return estimates, p_white, outside_mass


# ---------------------- Kalman filter / smoother and EM ----------------------

@dataclass
class KFState:
    v: np.ndarray
    F: np.ndarray
    K: np.ndarray
    a: np.ndarray
    P: np.ndarray
    a_next: np.ndarray
    P_next: np.ndarray
    L: np.ndarray
    loglik: float


def kalman_filter(y: np.ndarray,
                  rhos: np.ndarray,
                  w: np.ndarray,
                  sigma2: float) -> KFState:
    Tmat = np.diag(rhos)
    Z = np.ones((1, len(rhos)))
    Q = np.diag((1.0 - rhos**2) * w)
    H = np.array([[sigma2]])

    Kdim = len(rhos)
    Tn = len(y)

    a = np.zeros((Tn + 1, Kdim))
    P = np.zeros((Tn + 1, Kdim, Kdim))
    v = np.zeros(Tn)
    F = np.zeros(Tn)
    Karr = np.zeros((Tn, Kdim))
    L = np.zeros((Tn, Kdim, Kdim))

    # stationary prior
    a[0] = np.zeros(Kdim)
    P[0] = np.diag(w)

    loglik = 0.0
    for t in range(Tn):
        v[t] = float(y[t] - (Z @ a[t])[0])
        F[t] = float((Z @ P[t] @ Z.T + H)[0, 0])
        if F[t] <= 1e-12:
            F[t] = 1e-12
        loglik += -0.5 * (math.log(2.0 * math.pi) + math.log(F[t]) + (v[t] ** 2) / F[t])

        K_t = (Tmat @ P[t] @ Z.T / F[t]).reshape(-1)
        Karr[t] = K_t
        a[t + 1] = (Tmat @ a[t]).reshape(-1) + K_t * v[t]

        L[t] = Tmat - np.outer(K_t, Z)

        P[t + 1] = Tmat @ P[t] @ L[t].T + Q
        P[t + 1] = 0.5 * (P[t + 1] + P[t + 1].T)

    return KFState(v=v, F=F, K=Karr, a=a[:-1], P=P[:-1], a_next=a[1:], P_next=P[1:], L=L, loglik=loglik)


@dataclass
class SmoothState:
    ahat: np.ndarray
    V: np.ndarray
    uhat: np.ndarray
    Uvar: np.ndarray
    etahat: np.ndarray
    Evar: np.ndarray
    r: np.ndarray
    N: np.ndarray


def disturbance_smoother(y: np.ndarray,
                         rhos: np.ndarray,
                         w: np.ndarray,
                         sigma2: float,
                         kf: KFState = None) -> SmoothState:
    if kf is None:
        kf = kalman_filter(y, rhos, w, sigma2)

    Tn = len(y)
    Kdim = len(rhos)

    Tmat = np.diag(rhos)
    Z = np.ones((1, Kdim))
    Q = np.diag((1.0 - rhos**2) * w)

    r = np.zeros((Tn + 1, Kdim))
    N = np.zeros((Tn + 1, Kdim, Kdim))

    for t in range(Tn, 0, -1):
        i = t - 1
        F_inv = 1.0 / kf.F[i]
        r[i] = (Z.T * (F_inv * kf.v[i])).reshape(-1) + kf.L[i].T @ r[i + 1]
        N[i] = (Z.T @ Z) * F_inv + kf.L[i].T @ N[i + 1] @ kf.L[i]

    ahat = kf.a + np.einsum('tij,tj->ti', kf.P, r[:-1])
    V = kf.P - np.einsum('tij,tjk,tkm->tim', kf.P, N[:-1], kf.P)

    # observation disturbance
    uhat = np.zeros(Tn)
    Uvar = np.zeros(Tn)
    for t in range(Tn):
        F_inv = 1.0 / kf.F[t]
        Kt = kf.K[t][:, None]
        uhat[t] = float(sigma2 * (F_inv * kf.v[t] - (Kt.T @ r[t + 1: t + 2].T)[0, 0]))
        Uvar[t] = float(sigma2 - sigma2 * (F_inv + (Kt.T @ N[t + 1] @ Kt)[0, 0]) * sigma2)

    etahat = np.zeros((Tn, Kdim))
    Evar = np.zeros((Tn, Kdim, Kdim))
    for t in range(Tn):
        etahat[t] = (Q @ r[t + 1]).reshape(-1)
        Evar[t] = Q - Q @ N[t + 1] @ Q

    return SmoothState(ahat=ahat, V=V, uhat=uhat, Uvar=Uvar, etahat=etahat, Evar=Evar, r=r, N=N)


def em_A_B_from_smoother(y: np.ndarray,
                         rhos: np.ndarray,
                         w: np.ndarray,
                         sigma2: float) -> Tuple[np.ndarray, float, KFState, SmoothState]:
    kf = kalman_filter(y, rhos, w, sigma2)
    sm = disturbance_smoother(y, rhos, w, sigma2, kf=kf)

    Kdim = len(rhos)
    Tn = len(y)

    A = np.zeros(Kdim)
    for k in range(Kdim):
        E_zeta2 = sm.V[0, k, k] + sm.ahat[0, k] ** 2
        E_eta2_sum = 0.0
        for t in range(Tn-1):
            E_eta2_sum += sm.Evar[t, k, k] + sm.etahat[t, k] ** 2
        A[k] = E_zeta2 + (1.0 / (1.0 - rhos[k] ** 2)) * E_eta2_sum

    B = float(np.sum(sm.Uvar + sm.uhat ** 2))
    return A, B, kf, sm


def em_mstep_unit_sum(A: np.ndarray, B: float, Tn: int, S: float = 1.0):
    C = list(A) + [B]
    C = np.array(C, dtype=float)
    Cmax = np.max(C)
    lam_min = - (Tn ** 2) / (8.0 * Cmax) + 1e-12

    def F_of_lambda(lam: float) -> float:
        vals = 2.0 * C / (Tn + np.sqrt(Tn ** 2 + 8.0 * lam * C))
        return float(np.sum(vals))

    def solve_lambda(S_target: float) -> float:
        F0 = F_of_lambda(0.0)
        if S_target <= F0:
            lo, hi = 0.0, 1.0
            while F_of_lambda(hi) > S_target:
                hi *= 2.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if F_of_lambda(mid) > S_target:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        else:
            lo, hi = lam_min, 0.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if F_of_lambda(mid) < S_target:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)

    lam_star = solve_lambda(S)
    xs = 2.0 * C / (Tn + np.sqrt(Tn ** 2 + 8.0 * lam_star * C))
    w_new = xs[:-1]
    sigma2_new = xs[-1]
    return w_new, sigma2_new


def observed_loglik(y: np.ndarray, rhos: np.ndarray, w: np.ndarray, sigma2: float) -> float:
    return float(kalman_filter(y, rhos, w, sigma2).loglik)


# ---------------------- Demo / main ----------------------

def main():
    # True model (unit variance): 3 AR(1) components + white noise
    rhos_true = np.array([0.2, 0.6, 0.85])
    weights_true = np.array([0.25, 0.5, 0.2])
    sigma2_true = 1.0 - np.sum(weights_true)
    Tn = 4096
    seed = 13

    y = simulate_ar1_mixture(Tn, rhos_true, weights_true, sigma2_true, seed=seed)

    # Sieve grid up to a fixed boundary 1-δ
    delta = 0.08
    rho_max = 1.0 - delta
    J = 150
    rho_grid = np.linspace(0.01, rho_max, J)

    fit = fit_sieve_whittle(y, rho_grid, max_iter=300, tol=1e-9, stepsize=0.2)

    wgrid, yscaled = rfft_periodogram_2pi(y)
    s_true = true_spectrum_2pi(rhos_true, weights_true, sigma2_true, wgrid)
    s_hat = spectral_from_mixture(fit.p, rho_grid, wgrid)

    # Barycentric localization (ε < min separation/2; here 0.1 < 0.25/2? actually 0.1 < 0.125)
    eps = 0.1
    ests, pwhite_hat, outside_mass = barycentric_poles(fit.p, rho_grid, rhos_true, eps=eps)
    bary_rhos = np.array([e[0] for e in ests], dtype=float)
    bary_masses = np.array([e[1] for e in ests], dtype=float)

    sup_err = float(np.max(np.abs(s_hat - s_true)))
    l2_err = float(np.sqrt(np.mean((s_hat - s_true)**2)))

    # Closed-form EM M-step under unit-variance sum constraint
    w_em = np.ones_like(rhos_true) * (0.9 / len(rhos_true))
    sigma2_em = 1.0 - np.sum(w_em)
    ll_path = [observed_loglik(y, rhos_true, w_em, sigma2_em)]
    em_iters = 1000
    for _ in range(em_iters):
        A, B, _, _ = em_A_B_from_smoother(y, rhos_true, w_em, sigma2_em)
        w_em, sigma2_em = em_mstep_unit_sum(A, B, Tn, S=1.0)
        ll_path.append(observed_loglik(y, rhos_true, w_em, sigma2_em))

    # ---- Print a compact summary ----
    print("=== Sieve–Whittle fit ===")
    print(f"Converged: {fit.converged}")
    print(f"Objective start -> final: {fit.obj_path[0]:.6f} -> {fit.obj_path[-1]:.6f}")
    print(f"Spectral error (sup-norm on s=2πf): {sup_err:.6f}")
    print(f"Spectral error (L2 on s=2πf):      {l2_err:.6f}")
    print()
    print("=== Support localization (barycentric) ===")
    for j, rstar in enumerate(rhos_true):
        print(f"Pole {j+1} true ρ={rstar: .3f}  |  bary ρ̂={bary_rhos[j]: .3f}, mass in ±{eps} = {bary_masses[j]: .3f}")
    print(f"White-noise weight estimate (p[white]): {pwhite_hat:.4f} (true σ^2={sigma2_true:.4f})")
    print(f"Mass outside the union of ε-balls:     {outside_mass:.4f}")
    print()
    print("=== EM with closed-form M-step (sum constraint) ===")
    print("Log-likelihood path (monotone increase expected):")
    # for i, ll in enumerate(ll_path):
    #     print(f"  iter {i:2d}: {ll:.6f}")
    print(f"Final EM weights: {w_em}, final σ^2: {sigma2_em:.6f}")

    # ---- Plots ----
    plt.figure(figsize=(8, 4))
    plt.plot(wgrid, s_true / (2.0 * np.pi), label="True f(ω)")
    plt.plot(wgrid, s_hat / (2.0 * np.pi), label="Sieve–Whittle f̂(ω)")
    plt.title("Spectral density: true vs Sieve–Whittle estimate")
    plt.xlabel("ω")
    plt.ylabel("f(ω)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectra_true_vs_sieve.png", dpi=150)

    plt.figure(figsize=(8, 3))
    plt.stem(np.concatenate(([0.0], rho_grid)), fit.p, basefmt=" ")
    for r in rhos_true:
        plt.axvline(r, linestyle="--")
    plt.title("Estimated mixing measure on {white}∪grid (vertical lines: true poles)")
    plt.xlabel("ρ")
    plt.ylabel("weight")
    plt.tight_layout()
    plt.savefig("mixing_weights_grid.png", dpi=150)

    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(len(ll_path)), ll_path, marker="o")
    plt.title("EM with closed-form sum-constrained M-step: log-likelihood increases")
    plt.xlabel("EM iteration")
    plt.ylabel("log-likelihood")
    plt.tight_layout()
    plt.savefig("em_loglik_path.png", dpi=150)

    print("\nSaved figures: spectra_true_vs_sieve.png, mixing_weights_grid.png, em_loglik_path.png")

if __name__ == "__main__":
    main()
