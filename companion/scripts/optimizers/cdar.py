"""Bayesian return maximization under Conditional Drawdown-at-Risk constraints.

The LP uses uncompounded cumulative scenario-path returns, following the
Chekhlov-Uryasev-Zabarankin CDaR construction.  It is intentionally separate
from the existing terminal-loss CVaR optimizer so CVaR/CDaR can be compared
without silently changing either definition.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.optimize import linprog
from scipy import sparse

from scripts.optimizers.bayessian import BayessianCVaR
from scripts.calculations.drawdown import empirical_cdar, portfolio_path_returns


class BayessianCDaR(BayessianCVaR):
    """Return-maximizing optimizer with a sample-average CDaR risk budget."""

    def __init__(
        self,
        config: dict[str, Any],
        market_cap: np.ndarray,
        historical_returns: np.ndarray,
        pred_returns: np.ndarray,
        w_previous: np.ndarray,
        bound: tuple = (0, None),
    ):
        self.scenario_paths = np.asarray(config.get("scenario_paths", []), dtype=float)
        super().__init__(config, market_cap, historical_returns, pred_returns, w_previous, bound)

    def function_to_optimize(
        self,
        weights: np.ndarray,
        pred_returns: np.ndarray,
        w_previous: np.ndarray,
        method_type: Literal[
            "cvar_returns", "return_cvar_constraint", "return_cdar_constraint",
            "mean_cvar_sharpe", "expected_returns", "deviation_from_target", "bayesian_sharpe"
        ],
    ) -> float:
        if method_type == "return_cdar_constraint":
            mean_ret = float(np.mean(np.asarray(pred_returns, dtype=float) @ np.asarray(weights, dtype=float)))
            return float(-mean_ret + self.calculate_turnover_penalty(weights))
        return super().function_to_optimize(weights, pred_returns, w_previous, method_type)

    def _solve_return_cdar_lp(self, R_terminal: np.ndarray, w_prev: np.ndarray, N: int):
        paths = np.asarray(self.scenario_paths, dtype=float)
        if paths.ndim != 3 or paths.shape[2] != N:
            raise ValueError(
                "CDaR requires scenario_paths with shape (S,H,N). "
                f"Received {paths.shape}; configure BayessianModel path export and run.py injection."
            )
        paths = np.nan_to_num(paths, nan=0.0, posinf=0.0, neginf=0.0)
        S, H, _ = paths.shape
        M = S * H
        alpha = float(self.config.get("cdar_confidence_level", self.config.get("confidence_level", 0.95)))
        alpha = float(np.clip(alpha, 0.0, 1.0 - 1e-12))
        lam = float(self.turnover_penalty)
        mult = float(self.config.get("cdar_budget_mult", 1.0))
        configured_wmax = self.bound[1] if self.bound[1] is not None else 1.0
        wmax = max(float(configured_wmax), 1.0 / max(N, 1))

        # Linear cumulative asset-return coefficients C[s,t,j].
        C = np.cumsum(paths, axis=1)
        mu_terminal = np.asarray(R_terminal, dtype=float).mean(axis=0)

        ew = np.ones(N, dtype=float) / N
        cdar_ew = empirical_cdar(portfolio_path_returns(paths, ew), alpha=alpha)
        budget = mult * cdar_ew
        if not np.isfinite(budget) or budget <= 0.0:
            budget = mult * max(float(cdar_ew) if np.isfinite(cdar_ew) else 0.0, 1e-6)

        # x = [w(N), y(M), d(M), zeta(1), u(M), tp(N), tn(N)]
        o_w = 0
        o_y = o_w + N
        o_d = o_y + M
        o_z = o_d + M
        o_u = o_z + 1
        o_tp = o_u + M
        o_tn = o_tp + N
        nvar = o_tn + N

        c = np.zeros(nvar)
        c[o_w:o_w + N] = -mu_terminal
        c[o_tp:o_tp + N] = lam
        c[o_tn:o_tn + N] = lam

        # Build sparse inequalities: O(S H N) non-zeros rather than a dense
        # O((S H)^2) matrix. This is essential for realistic path counts.
        data = []
        row_idx = []
        col_idx = []
        rhs = []
        rr = 0
        def add_entry(r, cidx, val):
            if val != 0.0:
                row_idx.append(r); col_idx.append(cidx); data.append(float(val))
        for s in range(S):
            for t in range(H):
                m = s * H + t
                # Cw - y <= 0
                for j, v in enumerate(C[s, t]): add_entry(rr, o_w+j, v)
                add_entry(rr, o_y+m, -1.0); rhs.append(0.0); rr += 1
                # y - Cw - d <= 0
                add_entry(rr, o_y+m, 1.0)
                for j, v in enumerate(C[s, t]): add_entry(rr, o_w+j, -v)
                add_entry(rr, o_d+m, -1.0); rhs.append(0.0); rr += 1
                if t > 0:
                    add_entry(rr, o_y + s*H + t-1, 1.0); add_entry(rr, o_y+m, -1.0)
                    rhs.append(0.0); rr += 1
                # d - zeta - u <= 0
                add_entry(rr, o_d+m, 1.0); add_entry(rr, o_z, -1.0); add_entry(rr, o_u+m, -1.0)
                rhs.append(0.0); rr += 1
        # CDaR budget row
        add_entry(rr, o_z, 1.0)
        coef = 1.0 / ((1.0-alpha)*M)
        for m in range(M): add_entry(rr, o_u+m, coef)
        rhs.append(float(budget)); rr += 1
        A_ub = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(rr, nvar))
        b_ub = np.asarray(rhs, dtype=float)

        # turnover epigraph and fully invested budget
        A_eq = sparse.lil_matrix((N + 1, nvar), dtype=float); b_eq = np.zeros(N + 1)
        for j in range(N):
            A_eq[j, o_w+j] = 1.0; A_eq[j, o_tp+j] = -1.0; A_eq[j, o_tn+j] = 1.0
            A_eq[N, o_w+j] = 1.0
        A_eq = A_eq.tocsr()
        b_eq[:N] = np.asarray(w_prev, dtype=float)[:N]
        b_eq[N] = 1.0

        bounds = []
        bounds += [(0.0, wmax)] * N       # w
        bounds += [(0.0, None)] * M       # running peak y, initial peak 0 included by lower bound
        bounds += [(0.0, None)] * M       # drawdown d
        bounds += [(0.0, None)]            # zeta (drawdown quantile)
        bounds += [(0.0, None)] * M       # excess drawdown u
        bounds += [(0.0, None)] * N       # tp
        bounds += [(0.0, None)] * N       # tn

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success or not np.all(np.isfinite(res.x[:N])):
            self.last_lp_diagnostics = {"lp_success": False, "risk_measure": "CDaR", "message": str(res.message)}
            return None

        w = np.clip(res.x[:N], 0.0, wmax)
        w /= max(float(w.sum()), 1e-12)
        p_ret = portfolio_path_returns(paths, w)
        cdar_w = empirical_cdar(p_ret, alpha=alpha)
        try:
            slack = float(np.asarray(res.ineqlin.residual)[-1])
            marginal = float(np.asarray(res.ineqlin.marginals)[-1])
        except Exception:
            slack = float(budget - cdar_w); marginal = float("nan")
        self.last_lp_diagnostics = {
            "lp_success": True,
            "risk_measure": "CDaR",
            "cdar_level": alpha,
            "cdar_budget": float(budget),
            "cdar_value": float(cdar_w),
            "cdar_constraint_slack": slack,
            "cdar_constraint_binding": bool(abs(slack) <= max(1e-8, 1e-6 * max(abs(budget), 1.0))),
            "cdar_budget_dual_scipy_min": marginal,
            "cdar_budget_shadow_price_max": -marginal if np.isfinite(marginal) else float("nan"),
            "objective_expected_return": float(mu_terminal @ w),
            "target_turnover_l1": float(np.sum(np.abs(w - np.asarray(w_prev, dtype=float)[:N]))),
            "scenario_paths": int(S),
            "path_horizon": int(H),
        }
        return w

    def optimizer(self, args: tuple, count_etf: int, method_type: str) -> None:
        if method_type == "return_cdar_constraint":
            x0 = np.ones(count_etf, dtype=float) / count_etf
            w_prev = np.asarray(args[1], dtype=float) if len(args) > 1 else x0
            if w_prev.shape[0] != count_etf or not np.all(np.isfinite(w_prev)):
                w_prev = x0
            w = self._solve_return_cdar_lp(np.asarray(args[0], dtype=float), w_prev, count_etf)
            if w is not None:
                self.optimal_values.append(float(self.function_to_optimize(w, *args, method_type)))
                self.optimal_weights.append(np.asarray(w, dtype=float))
                return
        super().optimizer(args, count_etf, method_type)
