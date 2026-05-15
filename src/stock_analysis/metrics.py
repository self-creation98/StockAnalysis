from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float

    @classmethod
    def from_predictions(cls, actual: np.ndarray, predicted: np.ndarray) -> "RegressionMetrics":
        actual = np.asarray(actual).reshape(-1)
        predicted = np.asarray(predicted).reshape(-1)
        return cls(
            rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
            mae=float(mean_absolute_error(actual, predicted)),
            r2=float(r2_score(actual, predicted)),
        )

    def as_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2}

