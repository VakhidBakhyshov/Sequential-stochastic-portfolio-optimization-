import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from typing import Any
from beartype import beartype

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from models.base import BaseModel


class MLRegressionModel(BaseModel):
    def __init__(
        self,
        config: dict[str, Any],
        etfs_list: list,
        historical_returns: pd.DataFrame,
        historical_ewma_returns: pd.DataFrame,
        future_returns: np.ndarray
    ):
        super().__init__(
            config,
            etfs_list,
            historical_returns,
            historical_ewma_returns,
            future_returns
        )

        self.ml_type = self.config.get("ml_type", "ridge")
        self.horizon = int(self.config.get("horizon", 21))
        self.cv_splits = int(self.config.get("cv_splits", 3))

        self.min_train_size = int(self.config.get("min_train_size", 80))

        # Important: financial returns should not be clipped to huge values like 1e5.
        # These limits protect the model from bad prices, zeros, inf log-returns, and data errors.
        self.return_clip = float(self.config.get("return_clip", 0.50))     # daily log/simple return limit
        self.target_clip = float(self.config.get("target_clip", 1.50))     # forward horizon return limit

        self.var_floor = float(self.config.get("var_floor", 1e-6))
        self.random_state = int(self.config.get("random_state", 42))
        self.n_jobs = int(self.config.get("n_jobs", 1))

        if self.ml_type == "ridge":
            base_estimator = Ridge()
            param_grid = {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
            }

        elif self.ml_type == "lasso":
            base_estimator = Lasso(max_iter=20000)
            param_grid = {
                "model__alpha": [0.0001, 0.001, 0.01, 0.1]
            }

        elif self.ml_type == "elasticnet":
            base_estimator = ElasticNet(max_iter=20000)
            param_grid = {
                "model__alpha": [0.0001, 0.001, 0.01, 0.1],
                "model__l1_ratio": [0.2, 0.5, 0.8]
            }

        elif self.ml_type == "rf":
            base_estimator = RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, 7],
                "model__min_samples_leaf": [5, 10]
            }

        elif self.ml_type == "linear":
            base_estimator = LinearRegression()
            param_grid = {}

        else:
            raise ValueError(f"Unknown ml_type: {self.ml_type}")

        self.pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("model", base_estimator)
        ])

        self.param_grid = param_grid

    def _clean_return_series(self, returns_series: pd.Series) -> pd.Series:
        s = pd.to_numeric(returns_series, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)

        # Keep time index, but remove impossible numerical values.
        s = s.clip(lower=-self.return_clip, upper=self.return_clip)

        # Missing values inside historical returns should not become huge model inputs.
        s = s.fillna(0.0)

        return s.astype(float)

    def _safe_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.fillna(0.0)

        # Feature-specific safe clipping.
        for col in X.columns:
            if "vol" in col:
                X[col] = X[col].clip(lower=0.0, upper=self.return_clip)
            else:
                X[col] = X[col].clip(lower=-self.return_clip, upper=self.return_clip)

        return X.astype(float)

    def _safe_target(self, y: pd.Series) -> pd.Series:
        y = pd.to_numeric(y, errors="coerce")
        y = y.replace([np.inf, -np.inf], np.nan)
        y = y.clip(lower=-self.target_clip, upper=self.target_clip)
        return y.astype(float)

    @beartype
    def engineer_features_and_target(self, returns_series: pd.Series) -> pd.DataFrame:
        returns = self._clean_return_series(returns_series)

        df = pd.DataFrame(index=returns.index)
        df["returns"] = returns

        # Lag features
        df["lag_1"] = returns.shift(1)
        df["lag_2"] = returns.shift(2)
        df["lag_3"] = returns.shift(3)
        df["lag_5"] = returns.shift(5)

        # Volatility features
        df["vol_5"] = returns.rolling(5, min_periods=5).std()
        df["vol_10"] = returns.rolling(10, min_periods=10).std()
        df["vol_21"] = returns.rolling(21, min_periods=21).std()

        # Momentum / mean return features
        df["mean_5"] = returns.rolling(5, min_periods=5).mean()
        df["mean_10"] = returns.rolling(10, min_periods=10).mean()
        df["mean_21"] = returns.rolling(21, min_periods=21).mean()

        # Short-term reversal / momentum
        df["cum_5"] = returns.rolling(5, min_periods=5).sum()
        df["cum_10"] = returns.rolling(10, min_periods=10).sum()
        df["cum_21"] = returns.rolling(21, min_periods=21).sum()

        # Risk-adjusted momentum
        eps = 1e-8
        df["mean_10_div_vol_10"] = df["mean_10"] / (df["vol_10"] + eps)
        df["mean_21_div_vol_21"] = df["mean_21"] / (df["vol_21"] + eps)

        # Target: next horizon cumulative return.
        # At date t, this predicts r[t+1] + ... + r[t+horizon].
        df["target_forward_return"] = (
            returns
            .rolling(self.horizon, min_periods=self.horizon)
            .sum()
            .shift(-self.horizon)
        )

        df["target_forward_return"] = self._safe_target(df["target_forward_return"])

        return df

    def _fallback_prediction(self, train_df: pd.DataFrame) -> tuple[float, float]:
        if train_df.empty or "target_forward_return" not in train_df.columns:
            return 0.0, 1e-4

        y = self._safe_target(train_df["target_forward_return"]).dropna()

        if len(y) == 0:
            return 0.0, 1e-4

        recent_window = min(63, len(y))
        pred = float(y.tail(recent_window).mean())

        var = float(y.tail(recent_window).var(ddof=1))
        if not np.isfinite(var) or var < self.var_floor:
            var = 1e-4

        if not np.isfinite(pred):
            pred = 0.0

        return pred, var

    @beartype
    def prediction(self) -> np.ndarray:
        predictions = np.zeros(len(self.etfs_list), dtype=float)
        variances = np.ones(len(self.etfs_list), dtype=float) * 1e-4

        for i, etf in enumerate(self.etfs_list):
            if etf not in self.returns.columns:
                predictions[i] = 0.0
                variances[i] = 1e-4
                continue

            etf_returns = self.returns[etf]
            df_feats = self.engineer_features_and_target(etf_returns)

            feature_cols = [c for c in df_feats.columns if c != "target_forward_return"]

            current_features = df_feats.iloc[-1:][feature_cols]
            current_features = self._safe_matrix(current_features)

            train_df = df_feats.dropna(subset=["target_forward_return"]).copy()

            if len(train_df) < self.min_train_size:
                pred, var = self._fallback_prediction(train_df)
                predictions[i] = pred
                variances[i] = var
                continue

            X_train = train_df[feature_cols]
            y_train = train_df["target_forward_return"]

            X_train = self._safe_matrix(X_train)
            y_train = self._safe_target(y_train)

            valid_mask = np.isfinite(y_train.values)
            valid_mask &= np.all(np.isfinite(X_train.values), axis=1)

            X_train = X_train.loc[valid_mask]
            y_train = y_train.loc[valid_mask]

            if len(X_train) < self.min_train_size:
                pred, var = self._fallback_prediction(train_df)
                predictions[i] = pred
                variances[i] = var
                continue

            if y_train.nunique() <= 1:
                predictions[i] = 0.0
                variances[i] = 1e-4
                continue

            n_splits = min(self.cv_splits, max(2, len(X_train) // 40))

            if n_splits >= len(X_train):
                pred, var = self._fallback_prediction(train_df)
                predictions[i] = pred
                variances[i] = var
                continue

            tscv = TimeSeriesSplit(n_splits=n_splits)

            grid_search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=self.param_grid,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=self.n_jobs,
                error_score=np.nan
            )

            try:
                grid_search.fit(X_train, y_train)

                if not np.isfinite(grid_search.best_score_):
                    pred, var = self._fallback_prediction(train_df)
                    predictions[i] = pred
                    variances[i] = var
                    continue

                best_model = grid_search.best_estimator_

                pred = float(best_model.predict(current_features)[0])

                fitted = best_model.predict(X_train)
                residuals = y_train.values - fitted

                var = float(np.var(residuals, ddof=1))

                if not np.isfinite(pred):
                    pred, var = self._fallback_prediction(train_df)

                if not np.isfinite(var) or var < self.var_floor:
                    var = 1e-4

                predictions[i] = pred
                variances[i] = var

            except Exception:
                pred, var = self._fallback_prediction(train_df)
                predictions[i] = pred
                variances[i] = var

        predictions = np.nan_to_num(
            predictions,
            nan=0.0,
            posinf=self.target_clip,
            neginf=-self.target_clip
        )

        variances = np.nan_to_num(
            variances,
            nan=1e-4,
            posinf=1e-4,
            neginf=1e-4
        )

        variances = np.clip(variances, self.var_floor, self.target_clip ** 2)

        len_returns = len(self.future_returns)

        rng = np.random.default_rng(self.random_state)

        draws = rng.normal(
            loc=predictions,
            scale=np.sqrt(variances),
            size=(len_returns, len(self.etfs_list))
        )

        draws = np.nan_to_num(
            draws,
            nan=0.0,
            posinf=self.target_clip,
            neginf=-self.target_clip
        )

        draws = np.clip(draws, -self.target_clip, self.target_clip)

        return draws
