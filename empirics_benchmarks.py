import numpy as np
import pandas as pd
import warnings
import math

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from scipy.linalg import solve_discrete_lyapunov
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ============================================================
# Utilities (unchanged baselines)
# ============================================================

def train_test_split_series(y, test_frac=0.2, min_train=120):
    T = len(y)
    test_len = max(int(np.floor(T * test_frac)), 1)
    train_len = max(T - test_len, min_train)
    train = y.iloc[:train_len].copy()
    test = y.iloc[train_len:].copy()
    return train, test

def log_score_gaussian(err, var):
    return -0.5 * (np.log(2.0 * np.pi * var) + (err ** 2) / var)

def standardize_train_test(train, test):
    mu = train.mean()
    sd = train.std(ddof=0)
    if sd <= 0: sd = 1.0
    return (train-mu)/sd, (test-mu)/sd, mu, sd

def fit_best_arma(train, pmax=3, qmax=3):
    best_aic, best_order, best_res = np.inf, None, None
    for p in range(pmax+1):
        for q in range(qmax+1):
            try:
                mod = SARIMAX(train, order=(p,0,q), trend='c',
                              enforce_stationarity=True, enforce_invertibility=True)
                res = mod.fit(disp=False)
                if res.aic < best_aic:
                    best_aic, best_order, best_res = res.aic, (p,0,q), res
            except Exception:
                pass
    return best_res, best_order

def rolling_forecast_statespace(res, train, test):
    preds, pred_vars = [], []
    cur_res = res
    endog_name = getattr(train, 'name', None)
    for t in test.index:
        f = cur_res.get_forecast(steps=1)
        preds.append(f.predicted_mean.iloc[0])
        pred_vars.append(float(f.var_pred_mean.iloc[0]))
        cur_res = cur_res.append(pd.Series([test.loc[t]], index=[t], name=endog_name), refit=False)
    preds = pd.Series(preds, index=test.index)
    pred_vars = pd.Series(np.maximum(pred_vars, 1e-8), index=test.index)
    errs = test - preds
    return preds, pred_vars, float(np.mean(errs**2)), float(np.mean(log_score_gaussian(errs.values, pred_vars.values)))

def periodogram(y):
    y = np.asarray(y)
    T = len(y)
    fft = np.fft.rfft(y, n=T)
    I = (1.0/(2*np.pi*T)) * (fft*np.conj(fft)).real
    freqs = np.fft.rfftfreq(T, d=1.0) * 2*np.pi
    return freqs, I

def lag_window_smooth(I, L):
    I = I.copy()
    n = len(I)
    out = np.zeros_like(I)
    for k in range(n):
        s = 0.0; wsum = 0.0
        for h in range(-L, L+1):
            j = k + h
            if 0 <= j < n:
                w = 1.0 - abs(h)/(L+1)
                s += w * I[j]; wsum += w
        out[k] = s / max(wsum, 1e-12)
    return out

def gamma_from_spectrum(smooth_spec):
    n_rfft = len(smooth_spec)
    n_time = 2*(n_rfft - 1)
    cov = np.fft.irfft(smooth_spec, n=n_time)
    return cov

def spectral_ar_predictor(train, max_ar=20):
    """
    AR(p) chosen by AIC using autocovariances obtained from a lag-window–smoothed spectrum.
    Returns (phi, sigma_v, mean) where phi is length-p.
    """
    y = np.asarray(train - np.mean(train))
    T = len(y)

    # 1) Periodogram -> lag-window smooth
    freqs, I = periodogram(y)
    L = max(3, int(round(np.sqrt(T)/2)))
    I_s = lag_window_smooth(I, L)
    s_omega = 2*np.pi*I_s

    # 2) Back to time domain: gamma[0:p]
    gamma = np.asarray(gamma_from_spectrum(s_omega))
    gamma[0] = max(gamma[0], 1e-8)
    max_p = min(int(max_ar), len(gamma) - 1)

    best_ic, best_phi, best_sig = np.inf, None, None
    for p in range(1, max_p + 1):
        R = toeplitz(gamma[:p])               # p x p Toeplitz
        r = gamma[1:p+1]                      # length p
        try:
            a = np.linalg.solve(R, r)         # Yule–Walker AR coeffs
            sigma_v = float(gamma[0] - np.dot(a, r))
            if not np.isfinite(sigma_v) or sigma_v <= 0:
                continue
            aic = T * np.log(sigma_v) + 2 * p
            if aic < best_ic:
                best_ic, best_phi, best_sig = aic, a.copy(), sigma_v
        except np.linalg.LinAlgError:
            continue

    if best_phi is None:
        # safe AR(1) fallback
        phi1 = gamma[1] / gamma[0] if len(gamma) > 1 else 0.0
        best_phi = np.array([phi1], dtype=float)
        best_sig = float(max(gamma[0] * (1 - phi1**2), 1e-8))

    return best_phi, float(best_sig), float(np.mean(train))

def rolling_forecast_ar(phi, sigma_v, mean, train, test):
    p = len(phi)
    history = list(train.values)
    preds, vars_ = [], []
    for t in test.index:
        padded = history[-p:] if len(history)>=p else [history[-1]]*(p-len(history)) + history[-p:]
        x = np.array(padded[::-1])
        yhat = mean + np.dot(phi, x - mean)
        preds.append(yhat); vars_.append(max(sigma_v, 1e-8))
        history.append(test.loc[t])
    preds = pd.Series(preds, index=test.index)
    vars_ = pd.Series(vars_, index=test.index)
    errs = test - preds
    return preds, vars_, float(np.mean(errs**2)), float(np.mean(log_score_gaussian(errs.values, vars_.values)))

def gph_fractional_d(y, m=None):
    y = np.asarray(y - np.mean(y)); T = len(y)
    if m is None:
        m = int(T**0.6); m = max(10, min(m, T//2 - 1))
    freqs = 2*np.pi*np.arange(1, T//2+1)/T
    fft = np.fft.fft(y)
    I = (1.0/(2*np.pi*T)) * np.abs(fft[1:T//2+1])**2
    lam = freqs[:m]; I_m = I[:m]
    x = np.log(4*(np.sin(lam/2.0)**2)); Y = np.log(I_m)
    X = np.vstack([np.ones_like(x), -x]).T
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    d_hat = float(np.clip(beta[1], -0.45, 0.45))
    return d_hat

def fracdiff_weights(d, L):
    w = [1.0]
    for j in range(1, L+1):
        w.append(w[-1] * (d - j + 1)/j)
    w = np.array([((-1)**j) * w[j] for j in range(L+1)])
    return w

def fracint_weights(d, L):
    pi = [1.0]
    for j in range(1, L+1):
        pi.append(pi[-1]*(d + j - 1)/j)
    return np.array(pi)

def fracdiff_series(y, d, L=None):
    y = np.asarray(y); T = len(y)
    if L is None: L = min(200, T-1)
    w = fracdiff_weights(d, L)
    z = np.zeros(T)
    for t in range(T):
        jmax = min(t, L)
        z[t] = np.dot(w[:jmax+1], y[t-jmax:t+1][::-1])
    return z, w

def arfima_fit_predict(train, test, pmax=2, qmax=2, L=150):
    y_train = train.values; y_test = test.values
    mu = train.mean()
    y_train_c = y_train - mu; y_test_c = y_test - mu
    d_hat = gph_fractional_d(y_train_c)
    L_use = min(L, len(y_train_c)-1)
    z_train, w = fracdiff_series(y_train_c, d_hat, L=L_use)
    z_train_series = pd.Series(z_train, index=train.index, name=train.name)
    res_arma, order = fit_best_arma(z_train_series, pmax=pmax, qmax=qmax)
    pi = fracint_weights(d_hat, L=L_use+len(y_test)+5)
    preds, vars_ = [], []
    cur_res = res_arma
    z_history = list(z_train)
    for i, t in enumerate(test.index):
        f = cur_res.get_forecast(steps=1)
        zhat = f.predicted_mean.iloc[0]; zvar = float(f.var_pred_mean.iloc[0])
        jmax = min(len(z_history), len(pi)-1)
        past_sum = float(np.dot(pi[1:jmax+1], z_history[::-1][:jmax]))
        yhat = mu + zhat * pi[0] + past_sum
        preds.append(yhat); vars_.append(max(zvar, 1e-8))
        pass_series = np.concatenate([y_train_c, y_test_c[:i+1]])
        L_eff = min(L_use, len(pass_series)-1)
        z_new = np.dot(w[:L_eff+1], pass_series[-(L_eff+1):][::-1])
        z_history.append(z_new)
        cur_res = cur_res.append(pd.Series([z_new], index=[t], name=train.name), refit=False)
    preds = pd.Series(preds, index=test.index)
    vars_ = pd.Series(vars_, index=test.index)
    errs = test - preds
    return preds, vars_, float(np.mean(errs**2)), float(np.mean(log_score_gaussian(errs.values, vars_.values))), d_hat, order

# ============================================================
# S1. Extended kernels (negative-ρ AR(1) + AR(2) cycles)
# ============================================================

def ar1_kernel_full(rho, omega):
    """AR(1) kernel φ(ρ, ω) for ρ ∈ (-1,1), normalized so ∫ φ dω = 2π."""
    return (1 - rho**2) / (1 + rho**2 - 2 * rho * np.cos(omega))

def cycle_kernel_paper(r, psi, omega):
    """
    Paper-defined cycle kernel (even average of two AR(1) Poisson kernels):
        Φ_cyc(r, ψ; ω) = 0.5 * { φ(r, ω - ψ) + φ(r, ω + ψ) }
    Normalization: each φ integrates to 2π, so Φ_cyc integrates to 2π as well.
    """
    return 0.5 * (ar1_kernel_full(r, omega - psi) + ar1_kernel_full(r, omega + psi))

def _cycle_q_for_variance(r, psi, w_target):
    # rotation form; isotropic excitation makes the spectrum exactly Φ_cyc
    Tblk = r * np.array([[np.cos(psi), -np.sin(psi)],
                         [np.sin(psi),  np.cos(psi)]], dtype=float)
    Qunit = np.eye(2)
    P0 = solve_discrete_lyapunov(Tblk, Qunit)  # solves P = T P T' + Q
    var0 = float(P0[0, 0])
    var0 = max(var0, 1e-12)
    return w_target / var0


# ============================================================
# S2. Boundary-dense grids + two-pass refine
# ============================================================

def _map_exp_to_unit(x):
    return 1.0 - np.exp(-x)

def make_grid_ar1(K=24, rho_min=0.05, rho_max=0.995, allow_neg=True, mapping='exp'):
    eps = 1e-6
    rho_min = max(-0.999, min(rho_min, 0.999))
    rho_max = max(-0.999, min(rho_max, 0.999))
    if mapping == 'exp':
        pos_min, pos_max = max(eps, abs(rho_min)), max(eps, abs(rho_max))
        lam_min = -np.log(max(eps, 1 - min(pos_min, 0.999)))
        lam_max = -np.log(max(eps, 1 - min(pos_max, 0.999)))
        lam = np.linspace(lam_min, lam_max, K)
        pos = _map_exp_to_unit(lam)
        pos = np.clip(pos, eps, 0.999)
    else:
        pos = np.linspace(max(eps, rho_min), min(0.999, rho_max), K)

    if allow_neg:
        neg = -pos[::-1]
        rhos = np.concatenate([neg, pos])
    else:
        rhos = pos
    rhos = rhos[np.abs(rhos) > 1e-4]
    rhos.sort()
    return rhos

def make_grid_cycles(K_r=6, K_psi=6, r_min=0.3, r_max=0.98, psi_min=0.1, psi_max=np.pi-0.1, mapping_r='exp'):
    eps = 1e-6
    r_min = max(eps, min(r_min, 0.999))
    r_max = max(r_min + 1e-4, min(r_max, 0.999))
    if mapping_r == 'exp':
        lam_min = -np.log(1 - r_min)
        lam_max = -np.log(1 - r_max)
        lam = np.linspace(lam_min, lam_max, K_r)
        r_grid = _map_exp_to_unit(lam)
    else:
        r_grid = np.linspace(r_min, r_max, K_r)
    psi_grid = np.linspace(psi_min, psi_max, K_psi)
    atoms = [('cycle', float(r), float(psi)) for r in r_grid for psi in psi_grid]
    return atoms

def build_phi_matrix(omegas, atoms):
    """
    atoms (list): ('white',) or ('ar1', rho) or ('cycle', r, psi)
    Returns Phi (M x J) and cols metadata.
    """
    M = len(omegas)
    cols = []
    Phi = []
    for atom in atoms:
        if atom[0] == 'white':
            Phi.append(np.ones(M)); cols.append(atom)
        elif atom[0] == 'ar1':
            rho = float(atom[1])
            Phi.append(ar1_kernel_full(rho, omegas)); cols.append(atom)
        elif atom[0] == 'cycle':
            r, psi = float(atom[1]), float(atom[2])
            Phi.append(cycle_kernel_paper(r, psi, omegas)); cols.append(atom)
        else:
            raise ValueError(f"Unknown atom {atom}")
    Phi = np.column_stack(Phi) if Phi else np.zeros((M,0))
    return Phi, cols

def refine_grid_by_local_mass(p, cols, rhos_extra=8, cycles_extra=(4, 4), r_band=0.02, psi_band=0.15):
    """
    Two-pass refine: densify neighborhoods around active mass.
    """
    p = np.asarray(p)
    new_atoms = [('white',)] if ('white',) in cols else []
    thresh = 0.02 * float(p.max()) if p.size else 0.0

    # AR(1) centers
    ar1_centers = [float(c[1]) for c, w in zip(cols, p) if c[0]=='ar1' and w >= thresh]
    for rho0 in ar1_centers:
        lo = max(-0.999, rho0 - r_band); hi = min(0.999, rho0 + r_band)
        grid = np.linspace(lo, hi, rhos_extra)
        for r in grid:
            if abs(r) > 1e-4:
                new_atoms.append(('ar1', float(r)))

    # Cycle centers
    cycle_centers = [(float(c[1]), float(c[2])) for c, w in zip(cols, p) if c[0]=='cycle' and w >= thresh]
    for r0, psi0 in cycle_centers:
        r_lo, r_hi = max(0.05, r0 - r_band), min(0.999, r0 + r_band)
        p_lo, p_hi = max(0.05, psi0 - psi_band), min(np.pi - 0.05, psi0 + psi_band)
        r_grid = np.linspace(r_lo, r_hi, max(2, cycles_extra[0]))
        psi_grid = np.linspace(p_lo, p_hi, max(2, cycles_extra[1]))
        for r in r_grid:
            for psi in psi_grid:
                new_atoms.append(('cycle', float(r), float(psi)))

    # Deduplicate
    uniq = []
    for a in new_atoms:
        if a[0] == 'white':
            if ('white',) not in uniq:
                uniq.append(a)
        elif a[0]=='ar1':
            if not any(b[0]=='ar1' and abs(b[1]-a[1])<1e-6 for b in uniq):
                uniq.append(a)
        else:
            if not any(b[0]=='cycle' and abs(b[1]-a[1])<1e-6 and abs(b[2]-a[2])<1e-6 for b in uniq):
                uniq.append(a)
    return uniq

# ============================================================
# S3. Extended Whittle on atoms + EM for AR(1)
# ============================================================

def _periodogram_from_series(y):
    y = np.asarray(y, dtype=float)
    T = len(y)
    fft = np.fft.fft(y)
    M = (T - 1) // 2
    freqs = 2*np.pi*np.arange(1, M+1) / T
    I = (1.0/(2*np.pi*T)) * np.abs(fft[1:M+1])**2
    return freqs, I

def whittle_mixture_fit_extended(train, atoms, alpha=1.0, sum_to_var=True, bounds_floor=1e-12, maxiter=400):
    """
    Fit nonnegative weights p_j (sum to sample variance if sum_to_var=True) via Whittle over general atoms.
    """
    y = np.asarray(train - np.mean(train), dtype=float)
    T = len(y)
    omegas, I = _periodogram_from_series(y)
    Phi, cols = build_phi_matrix(omegas, atoms)
    if Phi.shape[1] == 0:
        raise ValueError("No atoms provided.")

    var_y = float(np.var(y, ddof=0)) if sum_to_var else 1.0
    J = Phi.shape[1]
    p0 = np.full(J, var_y / J)

    def obj(p):
        s = Phi @ p
        if np.any(s <= 1e-12): return 1e12
        val = np.mean(np.log(s) + (2*np.pi)*I / s)
        # optional symmetric Dirichlet stabilization (alpha)
        if alpha is not None and alpha != 1.0:
            if np.any(p <= bounds_floor): return 1e12
            val += - (alpha - 1.0) * np.mean(np.log(p))
        return val

    def grad(p):
        s = Phi @ p
        inv = 1.0 / s
        g = np.mean(Phi * (inv - (2*np.pi)*I * inv**2)[:, None], axis=0)
        if alpha is not None and alpha != 1.0:
            g += - (alpha - 1.0) * (1.0/len(p)) * (1.0/np.maximum(p, bounds_floor))
        return g

    cons = []
    if sum_to_var:
        cons.append({'type': 'eq', 'fun': lambda p: np.sum(p) - var_y, 'jac': lambda p: np.ones_like(p)})
    bnds = [(bounds_floor, var_y if sum_to_var else None) for _ in range(J)]
    res = minimize(obj, p0, method='SLSQP', jac=grad, bounds=bnds, constraints=cons,
                   options={'maxiter': maxiter, 'ftol': 1e-9})
    return res.x, cols, {'success': res.success, 'message': res.message, 'nit': res.nit, 'fun': float(res.fun)}

# ---------- EM for AR(1)+white noise (fixed rhos) ----------

def _kalman_filter_store(y, rhos, w, sigma2):
    y = np.asarray(y, dtype=float)
    Tn = np.array(rhos, dtype=float)
    K = len(Tn)
    Tmat = np.diag(Tn)
    Z = np.ones((1, K))
    Q = np.diag((1 - Tn**2) * w)
    H = float(sigma2)

    a = np.zeros(K)
    P = np.diag(w)
    v_list, F_list, K_list, L_list, a_list, P_list = [], [], [], [], [], []

    for t in range(len(y)):
        a_list.append(a.copy())
        P_list.append(P.copy())

        # One-step prediction
        y_pred = float(Z @ a)
        F = float(Z @ P @ Z.T + H)
        v = y[t] - y_pred

        # Filter gain (for the update)
        K_filt = (P @ Z.T) / F              # \bar K_t = P Z' / F

        # *** Smoother gain (this is the one needed in r,N,L recursions) ***
        K_smooth = Tmat @ K_filt            # K_t = T P Z' / F

        # L_t for the smoother uses K_smooth
        L = Tmat - K_smooth @ Z

        v_list.append(v)
        F_list.append(F)
        K_list.append(K_smooth.copy())      # store K_smooth (NOT K_filt)
        L_list.append(L.copy())

        # Measurement update (use the filter gain)
        a_filt = a + (K_filt.flatten() * v)
        P_filt = P - K_filt @ K_filt.T * F

        # Time update
        a = Tmat @ a_filt
        P = Tmat @ P_filt @ Tmat.T + Q

    return {
        'a_list': a_list, 'P_list': P_list, 'v_list': np.array(v_list),
        'F_list': np.array(F_list), 'K_list': K_list, 'L_list': L_list,
        'Tmat': Tmat, 'Z': Z, 'Q': Q, 'H': H
    }


def _disturbance_smoothing(y, filt):
    y = np.asarray(y, dtype=float)
    K = filt['Tmat'].shape[0]
    T = len(y)
    Z, H, Q = filt['Z'], filt['H'], filt['Q']
    v, F = filt['v_list'], filt['F_list']
    K_list, L_list = filt['K_list'], filt['L_list']
    a_list, P_list = filt['a_list'], filt['P_list']

    r = np.zeros((K,))         # this holds r_t, then r_{t-1}, ...
    N = np.zeros((K, K))       # this holds N_t, then N_{t-1}, ...

    alpha_smooth = [None]*T
    P_smooth = [None]*T
    eps_hat = np.zeros(T); Var_eps = np.zeros(T)
    eta_hat = np.zeros((T, K)); Var_eta_diag = np.zeros((T, K))
    Qdiag = np.diag(Q)

    for t in range(T-1, -1, -1):
        Ft = F[t]; vt = v[t]
        Kt = K_list[t]; Lt = L_list[t]
        Pt = P_list[t]; at = a_list[t]

        # ---- disturbances use (r_t, N_t) BEFORE the backward update ----
        eps_hat[t]  = H * (vt / Ft - float(Kt.T @ r))
        Var_eps[t]  = H - H * (1.0/Ft + float(Kt.T @ N @ Kt)) * H
        eta_hat[t]  = Qdiag * r
        Var_eta_diag[t] = Qdiag - (Qdiag**2) * np.diag(N)

        # ---- now update to (r_{t-1}, N_{t-1}) ----
        r_prev = (Z.T.flatten() / Ft) * vt + Lt.T @ r
        N_prev = (Z.T @ Z) / Ft + Lt.T @ N @ Lt

        # smoothed state uses (r_{t-1}, N_{t-1})
        alpha_smooth[t] = at + Pt @ r_prev
        P_smooth[t]     = Pt - Pt @ N_prev @ Pt

        r, N = r_prev, N_prev

    E_eta2 = eta_hat**2 + Var_eta_diag
    E_eps2 = eps_hat**2 + Var_eps
    alpha1, P1 = alpha_smooth[0], P_smooth[0]
    E_zeta2 = (alpha1**2 + np.diag(P1))

    # A_k and B as in Theorem 6.1
    A = E_zeta2 + np.sum(E_eta2[1:, :] / np.maximum(1e-12, (1 - np.diag(filt['Tmat'])**2)), axis=0)
    B = float(np.sum(E_eps2))
    return A, B, alpha_smooth, P_smooth


def _sum_constrained_update(A, B, S, T_obs):
    """
    Closed-form M-step under sum constraint ∑ w_k + σ^2 = S.
    Uses the sample size T_obs in: 2 λ x^2 + T_obs x − C = 0.
    """
    C_all = np.concatenate([A, [B]])
    C_max = float(np.max(C_all))
    lam_min = - (T_obs**2) / (8.0 * C_max)   # domain lower bound so discriminant ≥ 0

    def x_of_lambda(lam, C):
        disc = T_obs**2 + 8.0 * lam * C
        if disc <= 0:
            return np.nan
        return (2.0 * C) / (T_obs + np.sqrt(disc))

    def F_of_lambda(lam):
        xs = [x_of_lambda(lam, C) for C in C_all]
        return float(np.sum(xs))

    # Monotone bracketing for λ*: F(λ) is continuous, strictly decreasing on [lam_min, ∞)
    if S <= F_of_lambda(0.0):                 # then λ* ≥ 0
        lo, hi = 0.0, 1.0
        while F_of_lambda(hi) > S:
            hi *= 2.0
    else:                                      # then λ* ∈ (lam_min, 0)
        lo, hi = lam_min + 1e-12, 0.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if F_of_lambda(mid) > S:
            lo = mid
        else:
            hi = mid

    lam_star = 0.5 * (lo + hi)
    xs = np.array([x_of_lambda(lam_star, C) for C in C_all])
    w_new = xs[:-1]; sigma2_new = xs[-1]
    return w_new, float(sigma2_new)


def em_mixture_ar1(train, rhos, max_iter=200, tol=1e-6, init_w=None, init_sigma2=None, sum_to_var=True):
    """
    EM for AR(1) (+ white-noise) weights with fixed rhos.
    """
    y = np.asarray(train - np.mean(train), dtype=float)
    S = float(np.var(y, ddof=0)) if sum_to_var else 1.0
    K = len(rhos)

    if init_w is None:
        init_w = np.full(K, S / (K + 1))
    if init_sigma2 is None:
        init_sigma2 = max(1e-8, S - np.sum(init_w))

    w = np.array(init_w, dtype=float)
    sigma2 = float(init_sigma2)

    delta = np.inf
    for it in range(max_iter):
        filt = _kalman_filter_store(y, rhos, w, sigma2)
        A, B, _, _ = _disturbance_smoothing(y, filt)
        w_new, sigma2_new = _sum_constrained_update(A, B, S, T_obs=len(y))
        delta = np.max(np.abs(np.concatenate([w_new - w, [sigma2_new - sigma2]])))
        w, sigma2 = w_new, max(1e-12, float(sigma2_new))
        if delta < tol:
            break

    info = {'nit': it+1, 'converged': delta < tol, 'delta': float(delta)}
    return w, sigma2, info

# ============================================================
# State-space forecaster for AR(1)+cycles (and/or white noise)
# ============================================================

def kalman_forecast_mixture_general(train, test, ar1_atoms, cycle_atoms, w_ar1, w_cycles, sigma2):
    """
    Forecast with a state-space combining AR(1) states and AR(2) cycle blocks.
    Observation: y_t = sum(α_{k,t}) + sum(cycle_first_coord) + u_t,  u_t ~ N(0, σ²).
    """
    y_train = np.asarray(train, dtype=float)
    y_test = np.asarray(test, dtype=float)
    mu = float(np.mean(train))
    y_train_c = y_train - mu; y_test_c = y_test - mu

    rhos = np.array([a[1] for a in ar1_atoms], dtype=float) if ar1_atoms else np.array([], dtype=float)
    K1 = len(rhos)
    K2 = len(cycle_atoms)
    dim = K1 + 2*K2

    Tmat = np.zeros((dim, dim))
    Qmat = np.zeros((dim, dim))
    Z = np.zeros((1, dim))

    # AR(1) blocks
    for i, rho in enumerate(rhos):
        Tmat[i, i] = rho
        Qmat[i, i] = (1 - rho**2) * (w_ar1[i] if len(w_ar1) else 0.0)
        Z[0, i] = 1.0

    # Cycles
    for j, atom in enumerate(cycle_atoms):
        r, psi = float(atom[1]), float(atom[2])
        idx = K1 + 2*j
        # rotation form with isotropic noise — matches Φ_cyc exactly
        Tblk = r * np.array([[np.cos(psi), -np.sin(psi)],
                            [np.sin(psi),  np.cos(psi)]], dtype=float)
        Tmat[idx:idx+2, idx:idx+2] = Tblk

        wj = w_cycles[j] if len(w_cycles) else 0.0
        q = _cycle_q_for_variance(r, psi, wj)
        Qmat[idx:idx+2, idx:idx+2] += q * np.eye(2)

        Z[0, idx] = 1.0  # observe first coord



    H = float(sigma2)

    a = np.zeros(dim)
    P = np.zeros((dim, dim))
    if K1:
        P[:K1, :K1] = np.diag(w_ar1 if len(w_ar1) else np.zeros(K1))
    for j in range(K2):
        idx = K1 + 2*j
        wj = w_cycles[j] if len(w_cycles) else 0.0
        P[idx:idx+2, idx:idx+2] = np.eye(2) * wj

    preds, vars_ = [], []

    # Filter through training window
    for yt in y_train_c:
        yhat = float(Z @ a); F = float(Z @ P @ Z.T + H)
        v = yt - yhat
        Kgain = (P @ Z.T) / F
        a = a + (Kgain.flatten() * v)
        I = np.eye(dim)
        P = (I - Kgain @ Z) @ P @ (I - Kgain @ Z).T + H * (Kgain @ Kgain.T)
        a = Tmat @ a
        P = Tmat @ P @ Tmat.T + Qmat

    # Predict and update over test
    for yt in y_test_c:
        yhat = float(Z @ a); F = float(Z @ P @ Z.T + H)
        preds.append(mu + yhat); vars_.append(max(F, 1e-10))
        v = yt - yhat
        Kgain = (P @ Z.T) / F
        a = a + (Kgain.flatten() * v)
        I = np.eye(dim)
        P = (I - Kgain @ Z) @ P @ (I - Kgain @ Z).T + H * (Kgain @ Kgain.T)
        a = Tmat @ a
        P = Tmat @ P @ Tmat.T + Qmat

    preds = pd.Series(preds, index=test.index)
    vars_ = pd.Series(vars_, index=test.index)
    errs = test - preds
    return preds, vars_, float(np.mean(errs**2)), float(np.mean(log_score_gaussian(errs.values, vars_.values)))

# ============================================================
# Experiment runner (extended)
# ============================================================

def run_experiment():
    # Data and series (same as your original)
    macro = sm.datasets.macrodata.load_pandas().data
    macro.index = pd.period_range(start='1959Q1', periods=len(macro), freq='Q').to_timestamp()
    series = {}
    series['Inflation (CPI, % q/q ann.)'] = macro['infl']
    series['3m T-bill rate (%)'] = macro['tbilrate']
    series['Unemployment rate (%)'] = macro['unemp']
    for col, name in [('realgdp', 'Real GDP growth (%, ann.)'),
                      ('m1', 'M1 growth (%, ann.)')]:
        s = macro[col].astype(float)
        g = 400*np.log(s).diff()
        g.name = name
        series[name] = g
    for k in list(series.keys()):
        series[k] = series[k].dropna()
    common_index = None
    for s in series.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)
    for k in list(series.keys()):
        series[k] = series[k].loc[common_index]

    rows = []; details = {}

    for name, y in series.items():
        # Standard split and z-scoring
        train, test = train_test_split_series(y, test_frac=0.2, min_train=120)
        train_z, test_z, mu, sd = standardize_train_test(train, test)
        train_z.name = name; test_z.name = name
        series_payload = {'train_z': train_z, 'test_z': test_z}

        # ----- Baselines -----
        arma_res, arma_order = fit_best_arma(train_z, pmax=3, qmax=3)
        arma_preds, arma_vars, arma_mse, arma_ls = rolling_forecast_statespace(arma_res, train_z, test_z)

        uc_res = UnobservedComponents(endog=train_z, level='llevel', cycle=True, stochastic_cycle=True, trend=False).fit(disp=False)
        uc_preds, uc_vars, uc_mse, uc_ls = rolling_forecast_statespace(uc_res, train_z, test_z)

        phi, sig2, mean_tr = spectral_ar_predictor(train_z, max_ar=20)
        sp_preds, sp_vars, sp_mse, sp_ls = rolling_forecast_ar(phi, sig2, mean_tr, train_z, test_z)

        arfima_preds, arfima_vars, arfima_mse, arfima_ls, d_hat, arfima_order = arfima_fit_predict(train_z, test_z, pmax=2, qmax=2, L=150)

        # ----- Mix (Whittle AR1+cycles) with two-pass refine -----
        atoms = [('white',)]
        rhos0 = make_grid_ar1(K=24, rho_min=0.05, rho_max=0.995, allow_neg=True, mapping='exp')
        cycles0 = make_grid_cycles(K_r=6, K_psi=6, r_min=0.30, r_max=0.985, psi_min=0.10, psi_max=np.pi-0.10, mapping_r='exp')
        atoms += [('ar1', float(r)) for r in rhos0]
        atoms += cycles0

        # pass 1
        p1, cols1, info1 = whittle_mixture_fit_extended(train_z, atoms, alpha=1.0, sum_to_var=True)

        # two-pass refine around local mass (both AR1 and cycles)
        atoms2 = refine_grid_by_local_mass(p1, cols1, rhos_extra=8, cycles_extra=(4, 4), r_band=0.03, psi_band=0.20)
        # keep white + everything refined (no filtering out cycles)
        if ('white',) not in atoms2:
            atoms2 = [('white',)] + atoms2

        # pass 2
        p2, cols2, info2 = whittle_mixture_fit_extended(train_z, atoms2, alpha=1.0, sum_to_var=True)

        # extract weights by class
        w_white = sum(p2[i] for i, c in enumerate(cols2) if c[0] == 'white')
        ar1_atoms = [c for c in cols2 if c[0] == 'ar1']
        cycle_atoms = [c for c in cols2 if c[0] == 'cycle']
        w_ar1 = np.array([p2[i] for i, c in enumerate(cols2) if c[0] == 'ar1'])
        w_cycles = np.array([p2[i] for i, c in enumerate(cols2) if c[0] == 'cycle'])

        # state-space forecasting with AR(1)+cycles+white
        mixcyc_preds, mixcyc_vars, mixcyc_mse, mixcyc_ls = kalman_forecast_mixture_general(
            train_z, test_z, ar1_atoms, cycle_atoms, w_ar1, w_cycles, w_white
        )


        # ----- Mix (EM AR1 + white) -----
        rhos_em = make_grid_ar1(K=32, rho_min=0.05, rho_max=0.999, allow_neg=True, mapping='exp')
        w_em, sigma2_em, em_info = em_mixture_ar1(train_z, rhos_em, max_iter=200, tol=1e-6, sum_to_var=True)
        ar1_atoms_em = [('ar1', float(r)) for r in rhos_em]

        em_preds, em_vars, em_mse, em_ls = kalman_forecast_mixture_general(
            train_z, test_z, ar1_atoms_em, [], w_em, np.array([]), sigma2_em
        )

        # stash pass-1/2 atoms + weights for plotting (Figure 1)
        pass1_payload = {'p': p1, 'cols': cols1}
        pass2_payload = {'p': p2, 'cols': cols2}

        # stash per-model forecasts for reliability (Figure 2)
        forecasts_payload = {
            'ARMA':           {'preds': arma_preds,    'vars': arma_vars},
            'UC':             {'preds': uc_preds,      'vars': uc_vars},
            'Spec. smooth':   {'preds': sp_preds,      'vars': sp_vars},
            'ARFIMA':         {'preds': arfima_preds,  'vars': arfima_vars},
            'Mix (EM AR1)':   {'preds': em_preds,      'vars': em_vars},
            'Mix (AR1+cycles)': {'preds': mixcyc_preds, 'vars': mixcyc_vars},
        }

        rows.append({
            'Series': name,
            'ARMA MSE': arma_mse, 'ARMA log score': arma_ls,
            'UC MSE': uc_mse, 'UC log score': uc_ls,
            'Spec. smooth (AR) MSE': sp_mse, 'Spec. smooth (AR) log score': sp_ls,
            'ARFIMA MSE': arfima_mse, 'ARFIMA log score': arfima_ls,
            'Mix (EM AR1) MSE': em_mse, 'Mix (EM AR1) log score': em_ls,
            'Mix (AR1+cycles) MSE': mixcyc_mse, 'Mix (AR1+cycles) log score': mixcyc_ls,
        })


        details[name] = {
            'arma_order': arma_order,
            'arfima_order': arfima_order,
            'arfima_d': d_hat,
            'whittle_pass1': info1,
            'whittle_pass2': info2,
            'em_info': em_info,
            'mix_atoms_counts': {
                'white': int(('white',) in cols2),
                'ar1': len(ar1_atoms),
                'cycles': len(cycle_atoms)
            },
            # new:
            'series': series_payload,
            'pass1': pass1_payload,
            'pass2': pass2_payload,
            'forecasts': forecasts_payload,
        }


    results_df = pd.DataFrame(rows).set_index('Series')
    return results_df, details


# ============================================================
# S4. Table & Figures
# ============================================================

def _evaluate_mixture_spectrum(omega_grid, cols, weights):
    """
    Return f_hat(omega) on omega_grid for a given (cols, weights) mixture,
    in the *standard* units f(omega), i.e., divide Phi @ p by 2π.
    """
    Phi, _ = build_phi_matrix(omega_grid, cols)
    s = Phi @ np.asarray(weights, dtype=float)  # s = 2π f(ω)
    return s / (2.0 * np.pi)

def _smoothed_periodogram(train_z):
    # your periodogram is already normalized to f(ω); smooth a bit for display
    freqs, I = periodogram(np.asarray(train_z))
    L = max(3, int(round(np.sqrt(len(train_z))/2)))
    I_s = lag_window_smooth(I, L)
    return freqs, I_s  # same units as f(ω)

def save_table1(results_df, csv_path="Table1_oos_mse_logscore.csv", tex_path="Table1_oos_mse_logscore.tex"):
    """
    Save Table 1 as CSV and LaTeX (booktabs) with nice rounding.
    """
    df = results_df.copy()
    df_rounded = df.copy()
    for c in df.columns:
        df_rounded[c] = df[c].astype(float).round(4)
    df_rounded.to_csv(csv_path, index=True)

    # a compact LaTeX with booktabs, keep your column names
    with open(tex_path, "w") as f:
        f.write(df_rounded.to_latex(escape=False, float_format="%.4f", na_rep="", bold_rows=False, index=True, longtable=False, caption="Out-of-sample MSE and log scores", label="tab:oos"))
    print(f"[Table 1] Wrote {csv_path} and {tex_path}")

def plot_figure1_gdp(details, out_path="Figure1_GDP_spectra_weights.png"):
    """
    Figure 1: real GDP growth, pass 1 vs pass 2 spectra and mixing weights.
    Panels:
      (A) Smoothed periodogram + fitted spectra (pass 1 and pass 2)
      (B) AR(1) weight stems vs ρ, pass 1 vs pass 2
      (C) Cycle atoms: scatter over (ψ/π on x, r on y), marker size ∝ weight
    """
    gdp_name = 'Real GDP growth (%, ann.)'
    if gdp_name not in details:
        raise RuntimeError(f"Series '{gdp_name}' not found in details.")

    train_z = details[gdp_name]['series']['train_z']
    pass1 = details[gdp_name]['pass1']
    pass2 = details[gdp_name]['pass2']

    # Spectra on a dense grid in (0, π)
    omega_grid = np.linspace(1e-3, np.pi-1e-3, 1024)
    fhat1 = _evaluate_mixture_spectrum(omega_grid, pass1['cols'], pass1['p'])
    fhat2 = _evaluate_mixture_spectrum(omega_grid, pass2['cols'], pass2['p'])
    freqs, I_s = _smoothed_periodogram(train_z)

    # Split weights by atom type for pass1 / pass2
    def split_weights(cols, p):
        ar1_rho, ar1_w = [], []
        cyc_r, cyc_psi, cyc_w = [], [], []
        for cc, ww in zip(cols, p):
            if cc[0] == 'ar1':
                ar1_rho.append(float(cc[1])); ar1_w.append(float(ww))
            elif cc[0] == 'cycle':
                cyc_r.append(float(cc[1])); cyc_psi.append(float(cc[2])); cyc_w.append(float(ww))
        return np.array(ar1_rho), np.array(ar1_w), np.array(cyc_r), np.array(cyc_psi), np.array(cyc_w)

    rho1, w1, r1, psi1, cw1 = split_weights(pass1['cols'], pass1['p'])
    rho2, w2, r2, psi2, cw2 = split_weights(pass2['cols'], pass2['p'])

    fig = plt.figure(figsize=(12, 9))

    # (A) spectra
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(freqs, I_s, label="Smoothed periodogram")
    ax1.plot(omega_grid, fhat1, label="Fitted spectrum (pass 1)")
    ax1.plot(omega_grid, fhat2, label="Fitted spectrum (pass 2)")
    ax1.set_title("GDP growth: spectrum (train window)")
    ax1.set_xlabel("Frequency ω")
    ax1.set_ylabel("Spectral density f(ω)")
    ax1.legend(loc="best")

    # (B) AR(1) weights
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    if len(rho1):
        markerline1, stemlines1, baseline1 = ax2.stem(rho1, w1, linefmt='-', markerfmt='o', basefmt=' ')
    if len(rho2):
        markerline2, stemlines2, baseline2 = ax2.stem(rho2, w2, linefmt='--', markerfmt='s', basefmt=' ')
    ax2.set_title("AR(1) mixture weights vs pole ρ")
    ax2.set_xlabel("ρ")
    ax2.set_ylabel("Weight")
    ax2.xaxis.set_major_locator(MaxNLocator(6))
    ax2.legend(["pass 1", "pass 2"])

    # (C) Cycle atoms (size ∝ weight)
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    h1 = h2 = None
    if len(cw1):
        h1 = ax3.scatter(psi1/np.pi, r1, s=3000*np.maximum(cw1, 1e-8), marker='o', label='_nolegend_')
    if len(cw2):
        h2 = ax3.scatter(psi2/np.pi, r2, s=3000*np.maximum(cw2, 1e-8), marker='s', label='_nolegend_')

    ax3.set_title("Cycle atoms: center freq (ψ/π) vs radius r")
    ax3.set_xlabel("ψ / π")
    ax3.set_ylabel("r")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # --- small legend proxies ---
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', linestyle='None', color='C0', label='pass 1', markersize=7),
        Line2D([0], [0], marker='s', linestyle='None', color='C1', label='pass 2', markersize=7),
    ]
    ax3.legend(handles=handles, loc="best", handletextpad=0.4)


    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[Figure 1] Wrote {out_path}")

def _reliability_points(preds, vars_, y_true, coverages=(0.50,0.60,0.70,0.80,0.90,0.95)):
    """
    For each nominal coverage c, compute observed coverage of 1-step PI:
    interval is [μ ± z_{c} σ], where z_c is two-sided quantile.
    """
    mu = np.asarray(preds.values, dtype=float)
    var = np.asarray(vars_.values, dtype=float)
    y  = np.asarray(y_true.values, dtype=float)

    obs = []
    for c in coverages:
        z = norm.ppf(0.5*(1.0 + c))
        lo = mu - z * np.sqrt(var)
        hi = mu + z * np.sqrt(var)
        covered = np.mean((y >= lo) & (y <= hi))
        obs.append(float(covered))
    return np.array(coverages, dtype=float), np.array(obs, dtype=float)

def plot_figure2_reliability(details, results_df,
                             series_list=('Real GDP growth (%, ann.)', '3m T-bill rate (%)'),
                             models=('ARMA','UC','Spec. smooth','ARFIMA','Mix (EM AR1)','Mix (AR1+cycles)'),
                             out_path="Figure2_Reliability_GDP_Tbill.png"):
    """
    Reliability diagrams for selected series. Also annotates Δ log-score of the
    mixture vs ARMA (for a quick highlight).
    """
    fig, axes = plt.subplots(
        1, len(series_list),
        figsize=(6*len(series_list), 5),
        sharey=True,
        constrained_layout=True   # <— prevents legend/text/markers getting cut
    )

    if len(series_list) == 1:
        axes = [axes]

    for ax, sname in zip(axes, series_list):
        test_z = details[sname]['series']['test_z']
        ax.plot([0,1],[0,1], linestyle=':', label='ideal')  # y=x

        # compute and plot each model
        for m in models:
            preds = details[sname]['forecasts'][m]['preds']
            vars_ = details[sname]['forecasts'][m]['vars']
            c, obs = _reliability_points(preds, vars_, test_z)
            ax.plot(c, obs, marker='o', label=m)

        ax.set_title(f"Reliability: {sname}")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Observed coverage")
        ax.set_xlim(0.48, 1.02)   # <— extend to 1.02 so points at 1.0 aren’t clipped
        ax.set_ylim(0.45, 1.02)
        ax.legend(loc='lower right')

        # annotate Δ log-score (mixture vs ARMA) from Table 1
        if sname in results_df.index:
            ls_mix = float(results_df.loc[sname, 'Mix (AR1+cycles) log score'])
            ls_arma = float(results_df.loc[sname, 'ARMA log score'])
            delta = ls_mix - ls_arma
            ax.text(0.02, 0.95, f"Δ log-score (Mix−ARMA): {delta:.3f}",
                transform=ax.transAxes, ha='left', va='top')


    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[Figure 2] Wrote {out_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results_df, details = run_experiment()
    # console print
    print(results_df.round(4).to_string())

    # === Table 1 ===
    save_table1(results_df,
                csv_path="Table1_oos_mse_logscore.csv",
                tex_path="Table1_oos_mse_logscore.tex")

    # === Figure 1 (GDP spectra + weights, pass 1 vs pass 2) ===
    plot_figure1_gdp(details, out_path="Figure1_GDP_spectra_weights.png")

    # === Figure 2 (Reliability for GDP & T-bill) ===
    plot_figure2_reliability(details, results_df,
                             series_list=('Real GDP growth (%, ann.)', '3m T-bill rate (%)'),
                             models=('ARMA','UC','Spec. smooth','ARFIMA','Mix (EM AR1)','Mix (AR1+cycles)'),
                             out_path="Figure2_Reliability_GDP_Tbill.png")

