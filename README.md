[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Latest PDF:** [Mixtures_AR1_Xu2025.pdf](Mixtures_AR1_Xu2025.pdf)


# Mixtures of AR(1) Components — Companion Code

Companion repository for **Xu (2025), “Mixtures of AR(1) Components: Sieve–Whittle Estimation, Support Localization, and a Closed‑Form EM Weight Update.”**

## What this implements
- **Sieve–Whittle estimator** on fixed and shrinking boundaries (discrete Whittle objective with correct `2π` scaling).
- **Uniform LLN checks** via simulation (fixed and shrinking boundary regimes).
- **Support localization** + **barycentric pole** recovery from sieve mass.
- **Closed‑form EM M‑step** for mixture weights under a unit‑variance sum constraint (one‑dimensional dual; monotone likelihood ascent).
- Replication scaffold for the **U.S. unemployment** application (test MSE & log score vs. Yule‑Walker AR(2)).

> Paper reference sections: uniform LLN (Sec. 3; Lemmas 3.3–3.4), localization (Sec. 4), EM M-step (Sec. 5; Theorem 5.2), simulations + unemployment results (Sec. 6).

## Notes
- Periodogram scaling uses `s(ω)=2π f(ω)` so the discrete objective is `log s + (2π)I/s` (see paper’s Remark 3.1).
- For shrinking-boundary runs, pick meshes that satisfy the rate conditions discussed in Assumption 3.7.
- The unemployment replication uses `statsmodels` macrodata (or you can supply your own series).

## Citation
Xu, Z. (2025). *Mixtures of AR(1) Components: Sieve–Whittle Estimation, Support Localization, and a Closed‑Form EM Weight Update*.

## Cite this work

```bibtex
@unpublished{Xu2025Mixtures,
  title   = {Mixtures of AR(1) Components: Sieve--Whittle Estimation, Support Localization, and a Closed-Form EM Weight Update},
  author  = {Xu, Zhaoyang},
  note    = {Working paper},
  year    = {2025},
  month   = {October},
  url     = {https://github.com/zhaoyang8/Mixtures-AR1}
}
```
