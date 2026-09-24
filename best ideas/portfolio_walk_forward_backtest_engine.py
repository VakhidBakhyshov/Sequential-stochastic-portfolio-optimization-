"""
Walk-forward validation/test backtest engine for the ETF monthly-rebalancing framework.

Drop into: scripts/backtesting/walk_forward_engine.py

Main idea:
1. For each rebalance date, split already-known historical returns into:
   - train: older data used to fit optimizer/strategy weights;
   - validation: last ~6 months excluding last ~2 months;
   - internal test: last ~2 months.
2. Run candidate methods and parameter grids on train.
3. Rank all candidate weights on validation, keep top K.
4. Re-test only top K on the internal test split.
5. Select top-1 recipe/weights, optionally refit that selected recipe on all available history,
   then execute it on the real next month in your normal BaseExecution class.

This prevents choosing weights directly on the same month that is later reported as true OOS PnL.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Any, Callable, Iterable, Optional, Type

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class BacktestEngineConfig:
    """Configuration for the nested monthly backtest selector."""

    validation_months: int = 6
    internal_test_months: int = 2
    top_k_validation: int = 5
    select_metric: str = "total_return"  # total_return, sharpe, score
    periods_per_year: int = 12
    return_type: str = "log-returns"  # log-returns or returns
    c_bps: float = 1e-4
    min_history_months: int = 12
    refit_on_full_history: bool = True
    include_hold_current: bool = True
    include_strategies: bool = False
    alpha_grid: tuple[float, ...] = (1.0, 0.75, 0.50, 0.25)
    rank_penalty_turnover: float = 0.0  # optional stability penalty in score


@dataclass
class CandidateResult:
    candidate_id: str
    source: str  # optimizer, strategy, hold_current
    method: str
    params: dict[str, Any]
    alpha: float
    target_weights: pd.Series
    executed_weights: pd.Series
    turnover: float
    validation_total_return: float
    validation_sharpe: float
    validation_volatility: float
    validation_score: float
    test_total_return: float = np.nan
    test_sharpe: float = np.nan
    test_volatility: float = np.nan
    test_score: float = np.nan

    def to_record(self) -> dict[str, Any]:
        d = asdict(self)
        d["target_weights"] = self.target_weights.to_dict()
        d["executed_weights"] = self.executed_weights.to_dict()
        return d


@dataclass
class SelectionResult:
    selected: CandidateResult
    validation_table: pd.DataFrame
    test_table: pd.DataFrame
    split_info: dict[str, Any]
    selected_target_weights_full: Optional[pd.Series] = None


def flatten_config(section: Any) -> dict[str, Any]:
    """
    Your YAML sections are often represented as:
        [{'type': 'bayessian'}, {'min_weight': 0.01}, ...]
    This turns them into a normal flat dictionary.
    """
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if isinstance(section, list):
        out: dict[str, Any] = {}
        for item in section:
            if isinstance(item, dict):
                out.update(item)
        return out
    raise TypeError(f"Unsupported config section type: {type(section)!r}")


def expand_grid(base: dict[str, Any], grid: Optional[dict[str, Iterable[Any]]]) -> list[dict[str, Any]]:
    """Return list of base configs updated by every grid combination."""
    if not grid:
        return [dict(base)]

    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    configs: list[dict[str, Any]] = []
    for combo in product(*values):
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        configs.append(cfg)
    return configs


def as_return_frame(returns: pd.DataFrame | np.ndarray, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Convert returns to a clean numeric DataFrame with a DatetimeIndex when possible."""
    if isinstance(returns, pd.DataFrame):
        df = returns.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    arr = np.asarray(returns, dtype=float)
    return pd.DataFrame(arr, columns=columns)


def normalize_weights(weights: pd.Series, min_weight: float = 0.0, max_weight: Optional[float] = None) -> pd.Series:
    """Long-only clean-up: finite, non-negative, clipped and normalized to sum to 1."""
    w = weights.copy().astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w[w < min_weight] = 0.0
    if max_weight is not None:
        w = w.clip(lower=0.0, upper=float(max_weight))
    else:
        w = w.clip(lower=0.0)

    s = float(w.sum())
    if s <= EPS:
        w[:] = 1.0 / max(len(w), 1)
    else:
        w /= s
    return w


def simple_returns_from_frame(returns: pd.DataFrame, return_type: str) -> pd.DataFrame:
    if return_type == "log-returns":
        return np.exp(returns.astype(float)) - 1.0
    if return_type == "returns":
        return returns.astype(float)
    raise ValueError(f"Unknown return_type={return_type!r}; expected 'log-returns' or 'returns'.")


def portfolio_path_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    return_type: str,
    transaction_cost: float = 0.0,
) -> pd.Series:
    """Daily portfolio simple returns, with one initial cost charged on the first row."""
    if returns.empty:
        return pd.Series(dtype=float, name="portfolio_return")

    common = [c for c in returns.columns if c in weights.index]
    if not common:
        raise ValueError("No common assets between returns columns and weights index.")

    r_simple = simple_returns_from_frame(returns[common].fillna(0.0), return_type=return_type)
    w = weights.reindex(common).fillna(0.0).astype(float)
    w = normalize_weights(w)

    port = pd.Series(r_simple.values @ w.values, index=returns.index, name="portfolio_return")
    if len(port) > 0 and transaction_cost != 0.0:
        port.iloc[0] -= float(transaction_cost)
    return port


def performance_stats(
    path_returns: pd.Series,
    *,
    periods_per_year: int = 12,
    turnover: float = 0.0,
    turnover_penalty: float = 0.0,
) -> dict[str, float]:
    """Basic performance metrics for a candidate path."""
    r = path_returns.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if r.empty:
        return {"total_return": -np.inf, "sharpe": -np.inf, "volatility": np.inf, "score": -np.inf}

    total_return = float((1.0 + r).prod() - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0
    mean_ann = float(r.mean() * periods_per_year)
    sharpe = mean_ann / vol if vol > EPS else np.nan
    sharpe_for_score = sharpe if np.isfinite(sharpe) else -np.inf
    score = float(total_return + 0.05 * sharpe_for_score - turnover_penalty * turnover)

    return {
        "total_return": total_return,
        "sharpe": float(sharpe_for_score),
        "volatility": vol,
        "score": score,
    }


def split_train_validation_test_by_months(
    historical_returns: pd.DataFrame,
    *,
    validation_months: int = 6,
    internal_test_months: int = 2,
    min_history_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Calendar-month split of historical returns.

    Example with validation_months=6 and internal_test_months=2:
    [older train] [six validation months] [last two internal-test months]
    """
    df = as_return_frame(historical_returns)
    if df.empty:
        raise ValueError("historical_returns is empty.")

    if isinstance(df.index, pd.DatetimeIndex):
        months = pd.Index(df.index.to_period("M").unique()).sort_values()
        required = validation_months + internal_test_months + 1
        if len(months) >= max(required, min_history_months):
            test_months = months[-internal_test_months:]
            val_months = months[-(validation_months + internal_test_months):-internal_test_months]

            train = df[df.index.to_period("M").isin(months[: -(validation_months + internal_test_months)])]
            validation = df[df.index.to_period("M").isin(val_months)]
            test = df[df.index.to_period("M").isin(test_months)]
            split_info = {
                "mode": "calendar_months",
                "train_start": str(train.index.min()) if not train.empty else None,
                "train_end": str(train.index.max()) if not train.empty else None,
                "validation_months": [str(m) for m in val_months],
                "test_months": [str(m) for m in test_months],
            }
            return train, validation, test, split_info

    # Fallback for non-date index or very short history.
    n = len(df)
    test_n = max(1, int(round(n * internal_test_months / max(validation_months + internal_test_months + 4, 1))))
    val_n = max(1, int(round(n * validation_months / max(validation_months + internal_test_months + 4, 1))))
    train_end = max(1, n - val_n - test_n)
    train = df.iloc[:train_end]
    validation = df.iloc[train_end : n - test_n]
    test = df.iloc[n - test_n :]
    split_info = {
        "mode": "row_fallback",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    return train, validation, test, split_info


class PortfolioWalkForwardBacktestEngine:
    """
    Candidate selector for your monthly ETF framework.

    This class does not replace your outer monthly loop. Instead, call `select()` inside each
    rebalance iteration instead of the old results_by_fitting_model_alpha shortcut.
    """

    def __init__(
        self,
        *,
        engine_config: BacktestEngineConfig | dict[str, Any] | None = None,
        optimizer_registry: Optional[dict[str, Type[Any]]] = None,
        strategy_registry: Optional[dict[str, Type[Any]]] = None,
        model_class: Optional[Type[Any]] = None,
        model_config: Optional[dict[str, Any]] = None,
        optimizer_base_config: Optional[dict[str, Any]] = None,
        strategy_base_config: Optional[dict[str, Any]] = None,
        optimizer_param_grid: Optional[dict[str, Iterable[Any]]] = None,
        strategy_param_grid: Optional[dict[str, Iterable[Any]]] = None,
    ) -> None:
        if isinstance(engine_config, BacktestEngineConfig):
            self.cfg = engine_config
        else:
            self.cfg = BacktestEngineConfig(**(engine_config or {}))

        self.optimizer_registry = optimizer_registry or {}
        self.strategy_registry = strategy_registry or {}
        self.model_class = model_class
        self.model_config = model_config or {}
        self.optimizer_base_config = optimizer_base_config or {}
        self.strategy_base_config = strategy_base_config or {}
        self.optimizer_param_grid = optimizer_param_grid or {}
        self.strategy_param_grid = strategy_param_grid or {}

    def _predict_returns(
        self,
        *,
        etfs_list: list[str],
        train_returns: pd.DataFrame,
        train_ewma_returns: Optional[pd.DataFrame],
        eval_template_returns: pd.DataFrame,
        fallback_pred_returns: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate model scenarios without using validation/test returns as model inputs."""
        if self.model_class is None:
            if fallback_pred_returns is not None:
                return np.asarray(fallback_pred_returns, dtype=float)
            return train_returns[etfs_list].fillna(0.0).values

        ewma = train_ewma_returns if train_ewma_returns is not None else train_returns
        model = self.model_class(
            self.model_config,
            etfs_list,
            train_returns[etfs_list].copy(),
            ewma[etfs_list].copy(),
            eval_template_returns[etfs_list].fillna(0.0).values,
        )
        return np.asarray(model.prediction(), dtype=float)

    def _candidate_weights_from_optimizer(
        self,
        *,
        etfs_list: list[str],
        train_returns: pd.DataFrame,
        pred_returns: np.ndarray,
        market_cap_train: Optional[pd.DataFrame | np.ndarray],
        previous_weights: pd.Series,
    ) -> list[tuple[str, str, dict[str, Any], pd.Series]]:
        out: list[tuple[str, str, dict[str, Any], pd.Series]] = []
        base = dict(self.optimizer_base_config)

        optimizer_type_default = base.get("type")
        grid_configs = expand_grid(base, self.optimizer_param_grid)

        for cfg in grid_configs:
            opt_type = cfg.get("type", optimizer_type_default)
            if opt_type not in self.optimizer_registry:
                continue

            opt_cfg = dict(cfg)
            opt_cfg.setdefault("is_all_methods", False)

            opt_cls = self.optimizer_registry[opt_type]
            mc = np.asarray(market_cap_train, dtype=float) if market_cap_train is not None else np.ones_like(train_returns[etfs_list].values)
            w_prev = previous_weights.reindex(etfs_list).fillna(0.0).values

            try:
                opt = opt_cls(
                    opt_cfg,
                    mc,
                    train_returns[etfs_list].fillna(0.0).values,
                    np.asarray(pred_returns, dtype=float),
                    w_prev,
                )
                _, weights_list = opt.get_results()
            except Exception as exc:
                # Do not stop the whole walk-forward run because one parameter set failed.
                print(f"[BacktestEngine] Skipping optimizer={opt_type}, cfg={opt_cfg}, error={exc}")
                continue

            method_names = []
            try:
                method_names = list(opt.get_method_names())
            except Exception:
                pass

            for i, w_arr in enumerate(weights_list):
                if len(method_names) == len(weights_list):
                    method = method_names[i]
                else:
                    method = str(opt_cfg.get("task_type", opt_type))

                w = pd.Series(np.asarray(w_arr, dtype=float), index=etfs_list)
                w = normalize_weights(
                    w,
                    min_weight=float(opt_cfg.get("min_weight", 0.0)),
                    max_weight=opt_cfg.get("max_weight"),
                )
                out.append(("optimizer", method, opt_cfg, w))
        return out

    def _candidate_weights_from_strategy(
        self,
        *,
        etfs_list: list[str],
        train_returns: pd.DataFrame,
        pred_returns: np.ndarray,
        market_cap_train: Optional[pd.DataFrame | np.ndarray],
        previous_weights: pd.Series,
    ) -> list[tuple[str, str, dict[str, Any], pd.Series]]:
        if not self.cfg.include_strategies:
            return []

        out: list[tuple[str, str, dict[str, Any], pd.Series]] = []
        base = dict(self.strategy_base_config)
        strategy_type_default = base.get("type")
        grid_configs = expand_grid(base, self.strategy_param_grid)

        for cfg in grid_configs:
            strat_type = cfg.get("type", strategy_type_default)
            if strat_type not in self.strategy_registry:
                continue

            strat_cls = self.strategy_registry[strat_type]
            mc = np.asarray(market_cap_train, dtype=float) if market_cap_train is not None else None
            w_prev = previous_weights.reindex(etfs_list).fillna(0.0).values

            try:
                strat = strat_cls(
                    cfg,
                    train_returns[etfs_list].fillna(0.0).values,
                    np.asarray(pred_returns, dtype=float),
                    mc,
                    w_prev,
                )
                w_arr = strat.get_weights()
            except Exception as exc:
                print(f"[BacktestEngine] Skipping strategy={strat_type}, cfg={cfg}, error={exc}")
                continue

            w = pd.Series(np.asarray(w_arr, dtype=float), index=etfs_list)
            w = normalize_weights(w, min_weight=float(cfg.get("min_weight", 0.0)), max_weight=cfg.get("max_weight"))
            out.append(("strategy", strat_type, cfg, w))
        return out

    def _evaluate_candidate(
        self,
        *,
        candidate_id: str,
        source: str,
        method: str,
        params: dict[str, Any],
        target_weights: pd.Series,
        previous_weights: pd.Series,
        eval_returns: pd.DataFrame,
        alpha: float,
        prefix: str,
    ) -> tuple[CandidateResult, pd.Series]:
        prev = previous_weights.reindex(target_weights.index).fillna(0.0).astype(float)
        target = normalize_weights(target_weights)
        executed = prev + float(alpha) * (target - prev)
        executed = normalize_weights(executed)
        turnover = float(np.abs(executed - prev).sum())
        cost = float(self.cfg.c_bps) * turnover

        path = portfolio_path_returns(
            eval_returns,
            executed,
            return_type=self.cfg.return_type,
            transaction_cost=cost,
        )
        stats = performance_stats(
            path,
            periods_per_year=self.cfg.periods_per_year,
            turnover=turnover,
            turnover_penalty=self.cfg.rank_penalty_turnover,
        )

        result = CandidateResult(
            candidate_id=candidate_id,
            source=source,
            method=method,
            params=params,
            alpha=float(alpha),
            target_weights=target,
            executed_weights=executed,
            turnover=turnover,
            validation_total_return=stats["total_return"] if prefix == "validation" else np.nan,
            validation_sharpe=stats["sharpe"] if prefix == "validation" else np.nan,
            validation_volatility=stats["volatility"] if prefix == "validation" else np.nan,
            validation_score=stats["score"] if prefix == "validation" else np.nan,
            test_total_return=stats["total_return"] if prefix == "test" else np.nan,
            test_sharpe=stats["sharpe"] if prefix == "test" else np.nan,
            test_volatility=stats["volatility"] if prefix == "test" else np.nan,
            test_score=stats["score"] if prefix == "test" else np.nan,
        )
        return result, path

    def _rank_column(self, prefix: str) -> str:
        metric = self.cfg.select_metric
        if metric not in {"total_return", "sharpe", "score"}:
            raise ValueError("select_metric must be one of: total_return, sharpe, score")
        return f"{prefix}_{metric}"

    def _build_candidates(
        self,
        *,
        etfs_list: list[str],
        train_returns: pd.DataFrame,
        train_ewma_returns: Optional[pd.DataFrame],
        validation_returns: pd.DataFrame,
        market_cap_train: Optional[pd.DataFrame | np.ndarray],
        previous_weights: pd.Series,
        fallback_pred_returns: Optional[np.ndarray],
    ) -> list[tuple[str, str, dict[str, Any], pd.Series]]:
        pred_train = self._predict_returns(
            etfs_list=etfs_list,
            train_returns=train_returns,
            train_ewma_returns=train_ewma_returns,
            eval_template_returns=validation_returns,
            fallback_pred_returns=fallback_pred_returns,
        )

        candidates: list[tuple[str, str, dict[str, Any], pd.Series]] = []
        candidates.extend(
            self._candidate_weights_from_optimizer(
                etfs_list=etfs_list,
                train_returns=train_returns,
                pred_returns=pred_train,
                market_cap_train=market_cap_train,
                previous_weights=previous_weights,
            )
        )
        candidates.extend(
            self._candidate_weights_from_strategy(
                etfs_list=etfs_list,
                train_returns=train_returns,
                pred_returns=pred_train,
                market_cap_train=market_cap_train,
                previous_weights=previous_weights,
            )
        )

        if self.cfg.include_hold_current:
            hold = normalize_weights(previous_weights.reindex(etfs_list).fillna(0.0))
            candidates.append(("hold_current", "hold_current", {"type": "hold_current"}, hold))

        if not candidates:
            raise ValueError("No candidate weights were produced. Check optimizer/strategy registry and grids.")
        return candidates

    def _refit_selected_recipe_on_full_history(
        self,
        *,
        selected: CandidateResult,
        etfs_list: list[str],
        full_returns: pd.DataFrame,
        full_ewma_returns: Optional[pd.DataFrame],
        market_cap_full: Optional[pd.DataFrame | np.ndarray],
        previous_weights: pd.Series,
    ) -> pd.Series:
        """After hyperparameter selection, refit the winning recipe on all data known at rebalance date."""
        if not self.cfg.refit_on_full_history or selected.source == "hold_current":
            return selected.target_weights

        pred_full = self._predict_returns(
            etfs_list=etfs_list,
            train_returns=full_returns,
            train_ewma_returns=full_ewma_returns,
            eval_template_returns=full_returns.tail(max(20, min(len(full_returns), 60))),
        )

        if selected.source == "optimizer":
            cfg = dict(selected.params)
            opt_type = cfg.get("type")
            opt_cls = self.optimizer_registry.get(opt_type)
            if opt_cls is None:
                return selected.target_weights
            try:
                opt = opt_cls(
                    cfg,
                    np.asarray(market_cap_full, dtype=float) if market_cap_full is not None else np.ones_like(full_returns[etfs_list].values),
                    full_returns[etfs_list].fillna(0.0).values,
                    pred_full,
                    previous_weights.reindex(etfs_list).fillna(0.0).values,
                )
                _, weights_list = opt.get_results()
                if len(weights_list) == 0:
                    return selected.target_weights
                w = pd.Series(np.asarray(weights_list[0], dtype=float), index=etfs_list)
                return normalize_weights(w, min_weight=float(cfg.get("min_weight", 0.0)), max_weight=cfg.get("max_weight"))
            except Exception as exc:
                print(f"[BacktestEngine] Refit failed for optimizer recipe; using selected weights. error={exc}")
                return selected.target_weights

        if selected.source == "strategy":
            cfg = dict(selected.params)
            strat_type = cfg.get("type", selected.method)
            strat_cls = self.strategy_registry.get(strat_type)
            if strat_cls is None:
                return selected.target_weights
            try:
                strat = strat_cls(
                    cfg,
                    full_returns[etfs_list].fillna(0.0).values,
                    pred_full,
                    np.asarray(market_cap_full, dtype=float) if market_cap_full is not None else None,
                    previous_weights.reindex(etfs_list).fillna(0.0).values,
                )
                w = pd.Series(np.asarray(strat.get_weights(), dtype=float), index=etfs_list)
                return normalize_weights(w, min_weight=float(cfg.get("min_weight", 0.0)), max_weight=cfg.get("max_weight"))
            except Exception as exc:
                print(f"[BacktestEngine] Refit failed for strategy recipe; using selected weights. error={exc}")
                return selected.target_weights

        return selected.target_weights

    def select(
        self,
        *,
        etfs_list: list[str],
        historical_returns: pd.DataFrame,
        w_previous: pd.Series,
        historical_ewma_returns: Optional[pd.DataFrame] = None,
        market_cap_history: Optional[pd.DataFrame | np.ndarray] = None,
        fallback_pred_returns: Optional[np.ndarray] = None,
    ) -> SelectionResult:
        """Run nested validation/test selection and return the selected candidate."""
        full_returns = as_return_frame(historical_returns)[etfs_list].copy()
        full_ewma = as_return_frame(historical_ewma_returns)[etfs_list].copy() if historical_ewma_returns is not None else None

        train, validation, internal_test, split_info = split_train_validation_test_by_months(
            full_returns,
            validation_months=self.cfg.validation_months,
            internal_test_months=self.cfg.internal_test_months,
            min_history_months=self.cfg.min_history_months,
        )

        if train.empty or validation.empty or internal_test.empty:
            raise ValueError(f"Bad split: train={train.shape}, validation={validation.shape}, test={internal_test.shape}")

        if full_ewma is not None:
            train_ewma = full_ewma.reindex(train.index).dropna(how="all")
            full_ewma_for_refit = full_ewma
        else:
            train_ewma = None
            full_ewma_for_refit = None

        market_cap_train = None
        market_cap_full = None
        if market_cap_history is not None:
            if isinstance(market_cap_history, pd.DataFrame):
                mc = as_return_frame(market_cap_history)
                market_cap_train = mc.reindex(train.index)[etfs_list].fillna(0.0)
                market_cap_full = mc.reindex(full_returns.index)[etfs_list].fillna(0.0)
            else:
                market_cap_train = np.asarray(market_cap_history, dtype=float)
                market_cap_full = market_cap_train

        prev = normalize_weights(w_previous.reindex(etfs_list).fillna(0.0))

        raw_candidates = self._build_candidates(
            etfs_list=etfs_list,
            train_returns=train,
            train_ewma_returns=train_ewma,
            validation_returns=validation,
            market_cap_train=market_cap_train,
            previous_weights=prev,
            fallback_pred_returns=fallback_pred_returns,
        )

        validation_results: list[CandidateResult] = []
        for i, (source, method, params, target_w) in enumerate(raw_candidates):
            for alpha in self.cfg.alpha_grid:
                candidate_id = f"{source}:{method}:candidate={i}:alpha={alpha:.4f}"
                result, _ = self._evaluate_candidate(
                    candidate_id=candidate_id,
                    source=source,
                    method=method,
                    params=params,
                    target_weights=target_w,
                    previous_weights=prev,
                    eval_returns=validation,
                    alpha=float(alpha),
                    prefix="validation",
                )
                validation_results.append(result)

        validation_table = pd.DataFrame([r.to_record() for r in validation_results])
        val_rank_col = self._rank_column("validation")
        validation_table = validation_table.sort_values(val_rank_col, ascending=False).reset_index(drop=True)
        top_ids = validation_table.head(self.cfg.top_k_validation)["candidate_id"].tolist()
        top_validation_results = [r for r in validation_results if r.candidate_id in set(top_ids)]

        test_results: list[CandidateResult] = []
        for r in top_validation_results:
            test_r, _ = self._evaluate_candidate(
                candidate_id=r.candidate_id,
                source=r.source,
                method=r.method,
                params=r.params,
                target_weights=r.target_weights,
                previous_weights=prev,
                eval_returns=internal_test,
                alpha=r.alpha,
                prefix="test",
            )
            # Preserve validation metrics for final reporting.
            test_r.validation_total_return = r.validation_total_return
            test_r.validation_sharpe = r.validation_sharpe
            test_r.validation_volatility = r.validation_volatility
            test_r.validation_score = r.validation_score
            test_results.append(test_r)

        test_table = pd.DataFrame([r.to_record() for r in test_results])
        test_rank_col = self._rank_column("test")
        test_table = test_table.sort_values(test_rank_col, ascending=False).reset_index(drop=True)

        selected_id = test_table.iloc[0]["candidate_id"]
        selected = next(r for r in test_results if r.candidate_id == selected_id)

        selected_full = self._refit_selected_recipe_on_full_history(
            selected=selected,
            etfs_list=etfs_list,
            full_returns=full_returns,
            full_ewma_returns=full_ewma_for_refit,
            market_cap_full=market_cap_full,
            previous_weights=prev,
        )

        return SelectionResult(
            selected=selected,
            validation_table=validation_table,
            test_table=test_table,
            split_info=split_info,
            selected_target_weights_full=selected_full,
        )


def build_full_weight_series(
    *,
    portfolios_date: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    etfs_list: list[str],
    selected_asset_weights: pd.Series,
) -> pd.Series:
    """Convert ticker-indexed selected weights back into your full portfolios_date['weights'] Series."""
    w_target = pd.Series(0.0, index=portfolios_date.index, dtype=float)
    selected = selected_asset_weights.reindex(etfs_list).fillna(0.0).astype(float)
    selected = normalize_weights(selected)
    w_target.loc[mask] = selected.values
    return w_target
