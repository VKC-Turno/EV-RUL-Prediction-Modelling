#!/usr/bin/env python3
"""Hierarchical Bayesian degradation model — "vehicles that charge & drive alike degrade alike".

We do NOT treat km/%SoC or %SoC/hr as instantaneous SoH proxies (they fail: they don't track coulomb). Instead we
model the DEGRADATION TRAJECTORY and let behaviour (a stress fingerprint) modulate its rate:

    SoH_ij  ~ Normal(a_i + b_i * age_ij, sigma^2)         # vehicle i, observation j; age in months
    a_i     ~ Normal(mu_a, sigma_a^2)                     # initial SoH (~100), partial-pooled
    b_i     ~ Normal(x_i . beta, sigma_b^2)               # degradation SLOPE; behaviour x_i predicts it

age is the backbone (SoH ~ declines with age); behaviour only tilts the slope. Fully conjugate -> Gibbs sampler in
numpy (no PyMC/Stan needed). The posterior predictive for a vehicle with NO SoH data (Mahindra-native) integrates
parameter uncertainty (beta), between-vehicle heterogeneity (sigma_b) and — for its behaviour x_j — gives an honest
credible band. If behaviour carries little signal, beta ~ 0 and every native curve collapses to the shared age
prior with wide bands — the truthful outcome.

fit_gibbs(...) -> posterior draws ; predict_curve(...) -> p10/p50/p90 SoH over an age grid.
"""
import numpy as np


def _inv_gamma(rng, shape, scale):
    # scale = sum of squares / 2 term; draw variance ~ InvGamma(shape, scale)
    return scale / rng.standard_gamma(shape, size=np.shape(scale)) if np.ndim(scale) else scale / rng.standard_gamma(shape)


def fit_gibbs(soh, age, vin_idx, X, group=None, n_iter=4000, burn=1500, thin=2, seed=0,
              beta_prior_sd=5.0, mu_a_prior=(100.0, 10.0)):
    """Gibbs sampler.
      soh, age : (M,) float observation arrays
      vin_idx  : (M,) int in [0, N) vehicle id per observation
      X        : (N, P) behaviour design (source-baseline dummies + shared, globally-scaled behaviour covariates)
      group    : (N,) int in [0, G) per-vehicle group (e.g. source/OEM). If given, the between-vehicle rate
                 heterogeneity sigma_b2 is estimated PER GROUP so each OEM's band is calibrated to its own fade-rate
                 spread. If None, a single global sigma_b2 is used.
    Returns posterior draws: beta (S,P), mu_a (S,), sigma2, sigma_a2 (S,), sigma_b2 (S,) or (S,G).
    """
    rng = np.random.default_rng(seed)
    soh = np.asarray(soh, float); age = np.asarray(age, float); vin_idx = np.asarray(vin_idx, int)
    X = np.asarray(X, float)
    N, P = X.shape
    M = len(soh)
    if group is None:
        group = np.zeros(N, int)
    group = np.asarray(group, int); G = int(group.max()) + 1

    # per-vehicle sufficient statistics for Z=[1, age]
    n_i = np.bincount(vin_idx, minlength=N).astype(float)
    s_age = np.bincount(vin_idx, weights=age, minlength=N)
    s_age2 = np.bincount(vin_idx, weights=age * age, minlength=N)
    s_y = np.bincount(vin_idx, weights=soh, minlength=N)
    s_agey = np.bincount(vin_idx, weights=age * soh, minlength=N)
    grp_n = np.bincount(group, minlength=G).astype(float)

    # inits
    a = np.full(N, 100.0); b = np.full(N, -0.2)
    beta = np.zeros(P); mu_a = 100.0
    sigma2 = 4.0; sigma_a2 = 4.0; sigma_b2 = np.full(G, 0.04)
    XtX_prior = np.eye(P) / beta_prior_sd**2
    draws = {k: [] for k in ["beta", "mu_a", "sigma2", "sigma_a2", "sigma_b2"]}

    for it in range(n_iter):
        m_b = X @ beta                                        # prior mean of each b_i from behaviour
        sb2_i = sigma_b2[group]                               # per-vehicle rate-prior variance
        # ---- sample (a_i, b_i) jointly per vehicle (vectorised 2x2) ----
        p = n_i / sigma2 + 1.0 / sigma_a2                     # [1,1] precision term
        q = s_age / sigma2                                    # off-diag
        r = s_age2 / sigma2 + 1.0 / sb2_i                     # [age,age]
        rhs0 = s_y / sigma2 + mu_a / sigma_a2
        rhs1 = s_agey / sigma2 + m_b / sb2_i
        det = p * r - q * q
        Saa = r / det; Sbb = p / det; Sab = -q / det
        mean_a = Saa * rhs0 + Sab * rhs1
        mean_b = Sab * rhs0 + Sbb * rhs1
        L00 = np.sqrt(np.maximum(Saa, 1e-12))
        L10 = Sab / L00
        L11 = np.sqrt(np.maximum(Sbb - L10 * L10, 1e-12))
        z0 = rng.standard_normal(N); z1 = rng.standard_normal(N)
        a = mean_a + L00 * z0
        b = mean_b + L10 * z0 + L11 * z1

        # ---- sample beta | b  (Bayesian linear reg: b ~ N(X beta, sigma_b2_i)); weight rows by 1/sb2_i ----
        w = 1.0 / sb2_i
        prec = XtX_prior + (X.T * w) @ X
        cov = np.linalg.inv(prec)
        mean = cov @ ((X.T * w) @ b)
        beta = rng.multivariate_normal(mean, cov)

        # ---- sample mu_a | a ----
        m0, s0 = mu_a_prior
        prec_mu = N / sigma_a2 + 1.0 / s0**2
        mean_mu = (a.sum() / sigma_a2 + m0 / s0**2) / prec_mu
        mu_a = rng.normal(mean_mu, np.sqrt(1.0 / prec_mu))

        # ---- variances (Inverse-Gamma, weak priors) ----
        resid = soh - (a[vin_idx] + b[vin_idx] * age)
        sigma2 = (0.5 * (resid @ resid) + 1.0) / rng.standard_gamma(0.5 * M + 1.0)
        da = a - mu_a
        sigma_a2 = (0.5 * (da @ da) + 1.0) / rng.standard_gamma(0.5 * N + 1.0)
        db = b - m_b
        ss_b = np.bincount(group, weights=db * db, minlength=G)      # per-group residual SS
        sigma_b2 = (0.5 * ss_b + 1e-3) / rng.standard_gamma(0.5 * grp_n + 1.0)

        if it >= burn and (it - burn) % thin == 0:
            draws["beta"].append(beta.copy()); draws["mu_a"].append(mu_a)
            draws["sigma2"].append(sigma2); draws["sigma_a2"].append(sigma_a2)
            draws["sigma_b2"].append(sigma_b2.copy())

    out = {k: np.array(v) for k, v in draws.items()}
    if G == 1:                                                # squeeze single-group sigma_b2 to (S,)
        out["sigma_b2"] = out["sigma_b2"][:, 0]
    return out


def _sigma_b(draws, group):
    """Per-draw between-vehicle rate SD for the requested group (handles both (S,) and (S,G) sigma_b2)."""
    sb2 = draws["sigma_b2"]
    return np.sqrt(sb2 if sb2.ndim == 1 else sb2[:, group])


def predict_curve(draws, x_j, age_grid, group=0, anchor_intercept=100.0, intercept_sd=1.0,
                  obs_noise=False, seed=1, qs=(10, 50, 90)):
    """Posterior-predictive SoH curve for a vehicle with behaviour x_j (design row) and NO SoH data.
    SoH(age) = a_j + b_j*age, with b_j ~ N(x_j.beta, sigma_b^2[group]) drawn per posterior sample using the group's
    OWN heterogeneity. a_j = anchor_intercept + N(0, intercept_sd^2) (real initial-SoH spread is ~+-1pp, so the band
    has non-zero width at age 0 instead of collapsing); set anchor_intercept=None to sample a_j ~ N(mu_a, sigma_a^2).
    Returns dict {q10, q50, q90, mean} each shape == age_grid."""
    rng = np.random.default_rng(seed)
    beta = draws["beta"]; S = len(beta)
    age_grid = np.asarray(age_grid, float)
    b_j = beta @ x_j + rng.standard_normal(S) * _sigma_b(draws, group)           # rate incl. group heterogeneity
    if anchor_intercept is None:
        a_j = draws["mu_a"] + rng.standard_normal(S) * np.sqrt(draws["sigma_a2"])
    else:
        a_j = float(anchor_intercept) + rng.standard_normal(S) * float(intercept_sd)
    curves = a_j[:, None] + b_j[:, None] * age_grid[None, :]                     # (S, len(age))
    if obs_noise:
        curves = curves + rng.standard_normal(curves.shape) * np.sqrt(draws["sigma2"])[:, None]
    out = {f"q{q}": np.percentile(curves, q, axis=0) for q in qs}
    out["mean"] = curves.mean(axis=0)
    return out


def predict_rate(draws, x_j, group=0, seed=1):
    """Posterior-predictive degradation rate (SoH/month, negative) for behaviour x_j: mean and 95% CI."""
    rng = np.random.default_rng(seed)
    b_j = draws["beta"] @ x_j + rng.standard_normal(len(draws["beta"])) * _sigma_b(draws, group)
    return dict(mean=float(b_j.mean()), lo=float(np.percentile(b_j, 2.5)), hi=float(np.percentile(b_j, 97.5)))


if __name__ == "__main__":
    # ---- synthetic self-test: recover a known behaviour effect ----
    rng = np.random.default_rng(42)
    N, P = 300, 3            # intercept + 2 behaviour features
    beta_true = np.array([-0.20, -0.06, 0.00])   # feature 1 accelerates fade; feature 2 irrelevant
    Xb = rng.standard_normal((N, P - 1))
    X = np.column_stack([np.ones(N), Xb])
    b_true = X @ beta_true + rng.standard_normal(N) * 0.05
    a_true = 100 + rng.standard_normal(N) * 1.0
    rows = []
    for i in range(N):
        nobs = rng.integers(5, 16); ages = np.sort(rng.uniform(0, 30, nobs))
        soh = a_true[i] + b_true[i] * ages + rng.standard_normal(nobs) * 1.5
        for ag, sh in zip(ages, soh):
            rows.append((i, ag, sh))
    idx, age, soh = map(np.array, zip(*rows))
    d = fit_gibbs(soh, age.astype(float), idx.astype(int), X, seed=0)
    bm = d["beta"].mean(0); blo, bhi = np.percentile(d["beta"], [2.5, 97.5], axis=0)
    print("beta_true :", beta_true)
    print("beta_post :", np.round(bm, 3))
    print("beta 95%CI:", [f"[{lo:+.3f},{hi:+.3f}]" for lo, hi in zip(blo, bhi)])
    ok = all(blo[k] <= beta_true[k] <= bhi[k] for k in range(P))
    print("recovered within 95% CI:", ok)
