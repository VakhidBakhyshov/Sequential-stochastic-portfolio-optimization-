import numpy as np

from scripts.theory.diffusion_benchmark import (
    ArithmeticBrownianSpec,
    bm_linear_portfolio_price_moments,
    bm_nonpositive_price_probability,
    bm_terminal_price_moments,
    correlated_bm_price_paths,
    DiffusionSpec,
    centered_normal_cvar,
    correlated_gbm_paths,
    gaussian_reversal_probability,
    gbm_simple_loss_var_cvar,
    normal_loss_var_cvar,
    partial_execution,
    portfolio_log_return_moments,
    solve_return_cvar_lp,
    topk_margin_certificate,
)


def _spec() -> DiffusionSpec:
    return DiffusionSpec(
        mu=np.array([0.06, 0.08, 0.10]),
        sigma=np.array([0.12, 0.18, 0.22]),
        corr=np.array([[1.0, 0.30, 0.20], [0.30, 1.0, 0.40], [0.20, 0.40, 1.0]]),
    )


def test_gbm_paths_repeat_and_positive():
    a = correlated_gbm_paths(_spec(), n_steps=20, n_paths=3, seed=7)
    b = correlated_gbm_paths(_spec(), n_steps=20, n_paths=3, seed=7)
    assert a.shape == (3, 21, 3)
    assert np.all(a > 0)
    assert np.allclose(a, b)


def test_centering_removes_mean_location():
    s = 0.02
    vals = [centered_normal_cvar(s, 0.95) for m in [-0.03, 0, 0.04]]
    assert max(vals) - min(vals) < 1e-14
    raw0 = normal_loss_var_cvar(0.0, s, 0.95)[1]
    raw1 = normal_loss_var_cvar(0.01, s, 0.95)[1]
    assert np.isclose(raw1 - raw0, -0.01)


def test_gbm_simple_loss_formula_matches_monte_carlo():
    spec = _spec()
    w = np.array([0.2, 0.3, 0.5])
    m, v = portfolio_log_return_moments(w, spec, 21)
    _, exact = gbm_simple_loss_var_cvar(m, np.sqrt(v), 0.95)
    rng = np.random.default_rng(123)
    y = rng.normal(m, np.sqrt(v), 250_000)
    loss = 1.0 - np.exp(y)
    k = int(np.ceil(0.05 * len(loss)))
    mc = np.mean(np.partition(loss, len(loss) - k)[-k:])
    assert abs(mc - exact) < 0.002


def test_selection_and_partial_execution_identities():
    cert = topk_margin_certificate(np.array([0.9, 0.8, 0.7, 0.4]), 2, 0.04)
    assert cert["stable"]
    p = gaussian_reversal_probability(0.60, 0.55, 0.03, 0.03)
    assert 0 < p < 0.5
    prev = np.array([0.5, 0.5, 0.0])
    tar = np.array([0.2, 0.4, 0.4])
    exe, ids = partial_execution(prev, tar, 0.55)
    assert np.allclose(exe, prev + 0.55 * (tar - prev))
    assert abs(ids["turnover_identity_error"]) < 1e-12
    assert abs(ids["contraction_identity_error"]) < 1e-12


def test_small_cvar_lp_is_feasible_and_respects_budget():
    rng = np.random.default_rng(3)
    mu = np.array([0.0002, 0.0003, 0.0001, 0.00025])
    A = np.array([
        [0.0004, 0.0001, 0.0001, 0.0000],
        [0.0001, 0.0005, 0.0002, 0.0001],
        [0.0001, 0.0002, 0.0006, 0.0002],
        [0.0000, 0.0001, 0.0002, 0.0005],
    ])
    R = rng.multivariate_normal(mu, A, size=2000)
    prev = np.ones(4) / 4
    out = solve_return_cvar_lp(R, mu, prev, beta=0.95, max_weight=0.40, turnover_penalty=0.001)
    assert out["success"]
    w = out["weights"]
    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w >= -1e-10)
    assert np.all(w <= 0.40 + 1e-9)
    assert out["cvar"] <= out["cvar_budget"] + 5e-4


def test_arithmetic_bm_exact_terminal_law_and_nonpositive_probability():
    spec = ArithmeticBrownianSpec(
        drift=np.array([2.0, -1.0]),
        abs_sigma=np.array([10.0, 15.0]),
        corr=np.array([[1.0, 0.35], [0.35, 1.0]]),
        s0=np.array([100.0, 80.0]),
    )
    mean, cov = bm_terminal_price_moments(spec, 252)
    assert np.allclose(mean, [102.0, 79.0])
    assert np.allclose(cov, spec.covariance)
    pneg = bm_nonpositive_price_probability(spec, 252)
    assert np.all((pneg >= 0.0) & (pneg <= 1.0))

    paths = correlated_bm_price_paths(spec, n_steps=252, n_paths=80_000, seed=123)
    terminal = paths[:, -1, :]
    assert np.allclose(terminal.mean(axis=0), mean, atol=0.12)
    assert np.allclose(np.cov(terminal.T), cov, atol=1.5)

    h = np.array([0.6, 0.4])
    pm, pv = bm_linear_portfolio_price_moments(h, spec, 252)
    sample = terminal @ h
    assert abs(sample.mean() - pm) < 0.15
    assert abs(sample.var(ddof=1) - pv) < 1.5


def test_empirical_cvar_matches_ru_tail_mass_at_integer_tail_count():
    from scripts.theory.diffusion_benchmark import empirical_cvar_loss
    rng = np.random.default_rng(77)
    r = rng.normal(0.0, 0.01, size=1000)
    losses = np.sort(-r)[::-1]
    expected = losses[:50].mean()  # exactly 5% of 1000 observations
    assert abs(empirical_cvar_loss(r, 0.95) - expected) < 1e-14
