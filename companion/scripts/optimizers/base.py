import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from beartype import beartype
from typing import Any, Callable
from scipy.optimize import minimize, linprog

from scripts.calculations.portfolio_utils import normalize_long_only

LAST_LP_DIAG: dict = {}   # diagnostics of the most recent return/CVaR LP solve (publication audit export)


class BaseCVaR:
    def __init__(
        self,
        config: dict[str, Any],
        market_cap: np.ndarray,
        historical_returns: np.ndarray,
        pred_returns: np.ndarray,
        w_previous: np.ndarray,
        bound: tuple = (0, None)
    ):
  
        self.config = config
        self.method = self.config.get("method", "SLSQP")
        self.min_weight = self.config.get("min_weight", 0.01)
        self.max_weight = self.config.get("max_weight", 0.1)
        self.is_all_methods = self.config.get("is_all_methods", True)
        self.max_iter = self.config.get('max_iter', None)
        
        print(f"{self.is_all_methods=}, {self.min_weight=}, {self.max_weight=}")
        
        self.constraint_max_weight = bool(self.config.get("constraint_max_weight", True))
        self.turnover_penalty = self.config.get("turnover_penalty", 1.0)
        self.penalty_type = self.config.get("penalty_type", "L1")
        
        self.market_cap = np.asarray(market_cap, dtype=float)
        self.historical_returns = np.asarray(historical_returns, dtype=float)
        self.pred_returns = np.asarray(pred_returns, dtype=float)
        self.w_previous = np.asarray(w_previous, dtype=float)
        
        if self.constraint_max_weight:
            self.bound = (0.0, self.max_weight) # mutilpe weights with constraint of max_weight
        else:
            self.bound = (0.0, None) # case where i'm getting 1 etf in portfolio with the best perfomance (no constraint on max_weight)
        
        self.all_methods: list[str] = []
        self.optimal_values: list[float] = []
        self.optimal_weights: list[np.ndarray] = []
        self.last_lp_diagnostics: dict[str, Any] = {}
        
        
    # @beartype
    # def remove_noise_cov_matrix(self):
    #     cov_matrix = np.cov(self.historical_returns, rowvar=False)
    #     eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        
    #     q = self.historical_returns.shape[1] / self.historical_returns.shape[0]
    #     sigma2 = np.mean(np.diag(cov_matrix))
    #     lam_plus = sigma2 * (1 + 1/np.sqrt(q))**2
        
    #     noise_mask = eigvals < lam_plus
    #     n_noise = noise_mask.sum()
    #     if n_noise > 0:
    #         eigvals[noise_mask] = eigvals[noise_mask].mean()
    #     cov_clean = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
    #     return cov_clean
    
    
    @beartype
    def remove_noise_cov_matrix(self) -> np.ndarray:
        """
        Marchenko-Pastur clipping with the correct upper edge when q=N/T.
        Previous code used (1 + 1/sqrt(q))^2, which over-clipped when T>N.
        """
        x = np.asarray(self.historical_returns, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
            return np.eye(max(1, x.shape[1] if x.ndim == 2 else 1)) * 1e-6

        cov_matrix = np.cov(x, rowvar=False)
        cov_matrix = np.atleast_2d(cov_matrix)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0

        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.clip(eigvals, 1e-12, None)

        q = x.shape[1] / max(x.shape[0], 1)  # N/T
        q = max(q, 1e-8)
        sigma2 = float(np.mean(np.diag(cov_matrix)))
        sigma2 = max(sigma2, 1e-12)
        lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2

        noise_mask = eigvals < lam_plus
        if np.any(noise_mask):
            eigvals[noise_mask] = float(np.mean(eigvals[noise_mask]))

        cov_clean = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cov_clean = (cov_clean + cov_clean.T) / 2.0
        cov_clean += 1e-10 * np.eye(cov_clean.shape[0])
        return cov_clean
        
        
    def calculate_turnover_penalty(self, weights: np.ndarray) -> float:
        # if self.penalty_type == "L1":
        #     turnover = np.sum(np.abs(weights - self.w_previous))
        # elif self.penalty_type == "L2":
        #     turnover = np.sum((weights - self.w_previous)**2)
        
        weights = np.asarray(weights, dtype=float)
        prev = np.asarray(self.w_previous, dtype=float)
        if prev.shape != weights.shape:
            prev = np.resize(prev, weights.shape)

        if self.penalty_type == "L1":
            turnover = np.sum(np.abs(weights - prev))
        elif self.penalty_type == "L2":
            turnover = np.sum((weights - prev) ** 2)
        else:
            raise ValueError(f'Unknown penalty type {self.penalty_type}')
        return float(self.turnover_penalty * turnover)
        
    
    def function_to_optimize(self, *args, **kwargs) -> Callable:
        raise NotImplementedError("Child class must implement function_to_optimize")
    
    
    def _postprocess_weights(self, weights: np.ndarray) -> np.ndarray:
        # # 1 method - simple
        # weights = np.where(weights >= self.min_weight, weights, 0) # constraint for min_weight - if weight < min_weight => weight = 0
        # # weights /= np.sum(weights) # we are normalizing weights after constraint on min_weight
        # return weights
        
        # 2 method - updated + complicated
        return normalize_long_only(
            weights,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            fallback_n=weights.size,
        )
    
    def _solve_return_cvar_lp(self, R: np.ndarray, w_prev: np.ndarray, N: int):
        """Experiment 2 objective: Maximize expected return subject to  CVaR_alpha(w) <= budget.
        Rockafellar-Uryasev CVaR as a *constraint*, expected return as the objective, with an
        L1 turnover penalty. Pure LP (exact). Requires a real (S x N) scenario fan; with the
        scenario-fan fix R has S=1000 distinct rows, so the CVaR constraint is a genuine tail cap.
        Budget = cvar_budget_mult * CVaR_alpha(equal-weight) -> 'earn the most you can without
        taking more tail risk than naive 1/N' (mult>1 loosens, mult<1 tightens).
        x = [w(N), eta(1), u(S), tp(N), tn(N)].
        """
        R = np.atleast_2d(np.asarray(R, dtype=float))
        if R.shape[1] != N:
            R = R.reshape(-1, N)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        S = R.shape[0]
        alpha = float(self.alpha_level)
        lam = float(self.turnover_penalty)
        configured_wmax = self.bound[1] if self.bound[1] is not None else 1.0
        # A max-weight below 1/N makes the fully-invested constraint infeasible.
        wmax = max(float(configured_wmax), 1.0 / max(N, 1))
        mult = float(self.config.get("cvar_budget_mult", 1.0))
        coef = 1.0 / max((1.0 - alpha) * S, 1e-12)
        mu = R.mean(axis=0)                                   # per-ETF expected return (N,)

        # reference tail: CVaR_alpha of the equal-weight book over the same scenarios
        port_ew = R @ (np.ones(N) / N)
        losses_ew = -port_ew
        var_ew = float(np.quantile(losses_ew, alpha))
        tail_ew = losses_ew[losses_ew >= var_ew]
        cvar_ew = float(tail_ew.mean()) if tail_ew.size else var_ew
        budget = mult * cvar_ew
        # if equal-weight has no positive-loss tail, give a small positive slack so LP is feasible
        if not np.isfinite(budget) or budget <= 0:
            budget = mult * max(abs(cvar_ew), float(np.std(losses_ew)) + 1e-6)

        nvar = 3 * N + S + 1
        c = np.concatenate([-mu, [0.0], np.zeros(S), lam * np.ones(N), lam * np.ones(N)])

        # tail rows: -R_s.w - eta - u_s <= 0   (S rows)
        A_tail = np.zeros((S, nvar))
        A_tail[:, :N] = -R
        A_tail[:, N] = -1.0
        A_tail[:, N + 1:N + 1 + S] = -np.eye(S)
        b_tail = np.zeros(S)
        # CVaR budget row: eta + coef*sum u_s <= budget
        A_cv = np.zeros((1, nvar))
        A_cv[0, N] = 1.0
        A_cv[0, N + 1:N + 1 + S] = coef
        b_cv = np.array([budget])
        A_ub = np.vstack([A_tail, A_cv])
        b_ub = np.concatenate([b_tail, b_cv])

        # turnover: w - tp + tn = w_prev (N rows); budget sum w = 1 (1 row)
        A_eq = np.zeros((N + 1, nvar))
        A_eq[:N, :N] = np.eye(N)
        A_eq[:N, N + 1 + S:N + 1 + S + N] = -np.eye(N)
        A_eq[:N, N + 1 + S + N:] = np.eye(N)
        A_eq[N, :N] = 1.0
        b_eq = np.concatenate([np.asarray(w_prev, dtype=float)[:N], [1.0]])

        bnds = ([(0.0, wmax)] * N + [(None, None)] + [(0.0, None)] * S
                + [(0.0, None)] * N + [(0.0, None)] * N)
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
        except Exception as e:
            print(f"[RC-LP] linprog raised {e}; fallback SLSQP")
            LAST_LP_DIAG.clear(); LAST_LP_DIAG.update({"lp_success": False, "lp_message": str(e)})
            return None
        if res.success and np.all(np.isfinite(res.x[:N])):
            w = np.clip(res.x[:N], 0.0, wmax)
            if w.sum() > 1e-9:
                w = w / w.sum()
                losses = -(R @ w)
                var_w = float(np.quantile(losses, alpha))
                tail_w = losses[losses >= var_w]
                cvar_w = float(tail_w.mean()) if tail_w.size else var_w
                try:
                    slack = float(np.asarray(res.ineqlin.residual, dtype=float)[-1])
                    marginal_min = float(np.asarray(res.ineqlin.marginals, dtype=float)[-1])
                except Exception:
                    slack = float(budget - cvar_w)
                    marginal_min = float("nan")
                self.last_lp_diagnostics = {
                    "lp_success": True,
                    "cvar_level": alpha,
                    "cvar_budget": float(budget),
                    "cvar_value": cvar_w,
                    "cvar_constraint_slack": slack,
                    "cvar_constraint_binding": bool(abs(slack) <= max(1e-8, 1e-6 * max(abs(budget), 1.0))),
                    "cvar_budget_dual_scipy_min": marginal_min,
                    "cvar_budget_shadow_price_max": -marginal_min if np.isfinite(marginal_min) else float("nan"),
                    "objective_expected_return": float(mu @ w),
                    "target_turnover_l1": float(np.sum(np.abs(w - np.asarray(w_prev, dtype=float)[:N]))),
                }
                return w
        self.last_lp_diagnostics = {
            "lp_success": False,
            "message": str(getattr(res, "message", "")),
            "cvar_level": alpha,
            "cvar_budget": float(budget),
        }
        print(f"[RC-LP] not successful ({getattr(res,'message','')}); fallback SLSQP")
        return None

    @beartype
    def optimizer(self, args: tuple, count_etf: int, method_type: str) -> None:
        bounds = [self.bound] * count_etf
        x0 = np.ones(count_etf, dtype=float) / count_etf
        w_prev = np.asarray(args[1], dtype=float) if len(args) > 1 else x0
        if w_prev.shape[0] != count_etf or not np.all(np.isfinite(w_prev)):
            w_prev = x0

        # ---- Experiment 2: exact LP for MAX-return s.t. CVaR constraint ----
        if method_type == "return_cvar_constraint":
            w_lp = self._solve_return_cvar_lp(np.asarray(args[0], dtype=float), w_prev, count_etf)
            if w_lp is not None:
                # Do not threshold the exact LP solution afterwards: hard min-weight
                # cleanup can violate the CVaR budget that the LP just enforced.
                weights = np.asarray(w_lp, dtype=float)
                weights = weights / max(float(weights.sum()), 1e-12)
                res = float(self.function_to_optimize(weights, *args, method_type))
                self.optimal_values.append(res)
                self.optimal_weights.append(weights)
                print(f"[RC-LP] {method_type=}", f"sum={np.sum(weights):.4f}", f"active={int(np.sum(weights > 1e-9))}")
                return

        # for optimizer method COBYLA
        if self.method == "COBYLA":
            constraints = [
                {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)},        # sum(x) <= 1
                {'type': 'ineq', 'fun': lambda x: np.sum(x) - 0.99}      # sum(x) >= 0.99
            ]
        # for optimizer method SLSQP
        else:
            constraints = [{'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}]

        objective = lambda w: self.function_to_optimize(w, *args, method_type)
        options = {'maxiter': self.max_iter} if self.max_iter is not None else None

        result = minimize(
            fun=objective,
            x0=x0,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        if not result.success or not np.all(np.isfinite(result.x)):
            weights = x0
            res = float(objective(weights))
        else:
            weights = result.x
            res = float(result.fun)

        weights = self._postprocess_weights(weights)
        self.optimal_values.append(res)
        self.optimal_weights.append(weights)

        print(f"{method_type=}", f"{np.sum(weights)=:.6f}", f"active={np.sum(weights > 0)}", f"{weights.shape=}")
        
        
    @beartype
    def get_results(self) -> tuple[list[float], list[np.ndarray]]:
        return self.optimal_values, self.optimal_weights
    
    
    @beartype
    def get_method_names(self) -> list:
        return self.all_methods
