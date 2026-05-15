import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from stock_analysis.metrics import RegressionMetrics
from stock_analysis.models import Forecaster, ModelSpec, default_model_specs


class MetaLearner:
    """XGBoost meta-learner used to combine base model forecasts."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 5, learning_rate: float = 0.03):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for the stacking meta-learner. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",
            random_state=42,
        )

    def fit(self, features: np.ndarray, target: np.ndarray) -> "MetaLearner":
        self.model.fit(features, np.asarray(target).reshape(-1))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features).reshape(-1)

    def feature_importance(self, model_names: list[str]) -> dict[str, float]:
        importance = getattr(self.model, "feature_importances_", None)
        if importance is None:
            return {}
        return dict(zip(model_names, map(float, importance)))


class StackingEnsembleForecaster:
    """Train base learners with time-series OOF predictions, then stack them."""

    def __init__(
        self,
        model_specs: list[ModelSpec] | None = None,
        meta_learner: MetaLearner | None = None,
        n_splits: int = 5,
    ):
        self.model_specs = model_specs or default_model_specs()
        self.meta_learner = meta_learner or MetaLearner()
        self.n_splits = n_splits
        self.base_models_: dict[str, Forecaster] = {}

    @property
    def model_names(self) -> list[str]:
        return [spec.name for spec in self.model_specs]

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "StackingEnsembleForecaster":
        meta_features, meta_target = self._create_out_of_fold_features(x_train, y_train)
        self.meta_learner.fit(meta_features, meta_target)

        self.base_models_ = {}
        for spec in self.model_specs:
            self.base_models_[spec.name] = spec.factory().fit(x_train, y_train)

        return self

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        self._validate_fit()
        return self.meta_learner.predict(self._predict_base_models(x_values))

    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        inverse_transform,
    ) -> dict[str, RegressionMetrics]:
        self._validate_fit()
        actual = inverse_transform(y_test)
        results = {}

        for name, model in self.base_models_.items():
            predicted = inverse_transform(model.predict(x_test))
            results[name] = RegressionMetrics.from_predictions(actual, predicted)

        ensemble_predictions = inverse_transform(self.predict(x_test))
        results["Stacking Ensemble"] = RegressionMetrics.from_predictions(actual, ensemble_predictions)
        return results

    def feature_importance(self) -> dict[str, float]:
        return self.meta_learner.feature_importance(self.model_names)

    def _create_out_of_fold_features(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_splits = min(self.n_splits, len(x_values) - 1)
        if n_splits < 2:
            raise ValueError("Need more training samples to create time-series folds.")

        splitter = TimeSeriesSplit(n_splits=n_splits)
        features = np.full((len(x_values), len(self.model_specs)), np.nan)

        for train_index, validation_index in splitter.split(x_values):
            fold_x_train = x_values[train_index]
            fold_y_train = y_values[train_index]
            fold_x_valid = x_values[validation_index]

            for model_index, spec in enumerate(self.model_specs):
                model = spec.factory().fit(fold_x_train, fold_y_train)
                features[validation_index, model_index] = model.predict(fold_x_valid)

        valid_rows = ~np.isnan(features).any(axis=1)
        return features[valid_rows], y_values[valid_rows]

    def _predict_base_models(self, x_values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [self.base_models_[name].predict(x_values) for name in self.model_names]
        )

    def _validate_fit(self) -> None:
        if not self.base_models_:
            raise RuntimeError("The ensemble has not been fitted.")

