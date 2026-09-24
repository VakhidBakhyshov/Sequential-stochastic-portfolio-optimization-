import numpy as np

from scripts.theory.sequential_diffusion_analysis import (
    correlation_cvar_derivative,
    gaussian_cvar_gradient,
    gaussian_loss_cvar,
    solve_exact_gaussian_cvar_program,
)


def test_centered_gaussian_cvar_is_translation_invariant():
    w = np.array([0.4, 0.35, 0.25])
    mu = np.array([0.01, 0.02, 0.03])
    cov = np.array([[0.04, 0.01, 0.00], [0.01, 0.05, 0.01], [0.00, 0.01, 0.03]])
    a = gaussian_loss_cvar(w, mu, cov, centered=True)
    b = gaussian_loss_cvar(w, mu + 0.20, cov, centered=True)
    assert abs(a - b) < 1e-12


def test_gaussian_cvar_gradient_matches_finite_difference():
    w = np.array([0.4, 0.35, 0.25])
    mu = np.array([0.01, 0.02, 0.03])
    cov = np.array([[0.04, 0.01, 0.00], [0.01, 0.05, 0.01], [0.00, 0.01, 0.03]])
    g = gaussian_cvar_gradient(w, mu, cov, 0.95)
    h = 1e-6
    fd = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy(); wp[i] += h
        wm = w.copy(); wm[i] -= h
        fd[i] = (gaussian_loss_cvar(wp, mu, cov, 0.95) - gaussian_loss_cvar(wm, mu, cov, 0.95)) / (2*h)
    assert np.max(np.abs(g - fd)) < 1e-6


def test_correlation_derivative_positive_for_long_only_book():
    w = np.array([0.5, 0.3, 0.2])
    sigma = np.array([0.15, 0.20, 0.25])
    corr = np.eye(3)
    d = correlation_cvar_derivative(w, sigma, corr, 0, 1, beta=0.95, horizon_years=21/252)
    assert d > 0


def test_exact_gaussian_program_respects_budget():
    mu = np.array([0.010, 0.008, 0.006, 0.004, 0.003, 0.002])
    cov = 0.0004 * np.eye(6)
    prev = np.ones(6) / 6
    budget = gaussian_loss_cvar(prev, mu, cov, 0.95)
    res = solve_exact_gaussian_cvar_program(
        mu, cov, prev, beta=0.95, cvar_budget=budget,
        max_weight=0.25, turnover_penalty=0.001,
    )
    assert res["success"]
    assert abs(np.sum(res["weights"]) - 1.0) < 1e-8
    assert res["cvar"] <= budget + 1e-7
