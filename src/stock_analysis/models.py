from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class Forecaster(Protocol):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "Forecaster":
        ...

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], Forecaster]


class ARIMABaseModel:
    def __init__(self, order: tuple[int, int, int] = (2, 1, 2)):
        self.order = order
        self.fitted_model = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "ARIMABaseModel":
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for ARIMA. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        train_values = np.asarray(y_train).reshape(-1)

        try:
            self.fitted_model = ARIMA(train_values, order=self.order).fit()
        except Exception:
            self.fitted_model = ARIMA(train_values, order=(1, 1, 1)).fit()

        return self

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        self._validate_fit()
        forecast = self.fitted_model.forecast(steps=len(x_values))
        return np.asarray(forecast).reshape(-1)

    def _validate_fit(self) -> None:
        if self.fitted_model is None:
            raise RuntimeError("ARIMA model has not been fitted.")


class RandomForestBaseModel:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 12,
        random_state: int = 42,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "RandomForestBaseModel":
        self.model.fit(_flatten_sequences(x_train), np.asarray(y_train).reshape(-1))
        return self

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        return self.model.predict(_flatten_sequences(x_values)).reshape(-1)


class KerasBaseModel:
    def __init__(self, epochs: int = 30, batch_size: int = 32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "KerasBaseModel":
        self.model = self.build_model((x_train.shape[1], x_train.shape[2]))
        self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
        )
        return self

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        self._validate_fit()
        return self.model.predict(x_values, verbose=0).reshape(-1)

    def build_model(self, input_shape: tuple[int, int]):
        raise NotImplementedError

    def _validate_fit(self) -> None:
        if self.model is None:
            raise RuntimeError(f"{self.__class__.__name__} has not been fitted.")


class LSTMBaseModel(KerasBaseModel):
    def build_model(self, input_shape: tuple[int, int]):
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential

        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model


class GRUBaseModel(KerasBaseModel):
    def build_model(self, input_shape: tuple[int, int]):
        from tensorflow.keras.layers import GRU, Dense, Dropout
        from tensorflow.keras.models import Sequential

        model = Sequential(
            [
                GRU(64, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                GRU(32),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model


class TransformerBaseModel(KerasBaseModel):
    def build_model(self, input_shape: tuple[int, int]):
        from tensorflow import keras
        from tensorflow.keras.layers import (
            Dense,
            Dropout,
            GlobalAveragePooling1D,
            LayerNormalization,
            MultiHeadAttention,
        )
        from tensorflow.keras.models import Sequential

        inputs = keras.Input(shape=input_shape)
        x_values = inputs

        for _ in range(2):
            attention_output = MultiHeadAttention(key_dim=256, num_heads=4, dropout=0.1)(
                x_values,
                x_values,
            )
            attention_output = Dropout(0.1)(attention_output)
            x_values = LayerNormalization(epsilon=1e-6)(x_values + attention_output)

            feed_forward = Sequential(
                [
                    Dense(4, activation="relu"),
                    Dropout(0.1),
                    Dense(input_shape[1]),
                ]
            )
            x_values = LayerNormalization(epsilon=1e-6)(x_values + feed_forward(x_values))

        x_values = GlobalAveragePooling1D()(x_values)
        x_values = Dense(32, activation="relu")(x_values)
        x_values = Dropout(0.1)(x_values)
        outputs = Dense(1)(x_values)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model


def default_model_specs(random_state: int = 42) -> list[ModelSpec]:
    return [
        ModelSpec("ARIMA", lambda: ARIMABaseModel()),
        ModelSpec("Random Forest", lambda: RandomForestBaseModel(random_state=random_state)),
    ]


def deep_learning_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("LSTM", lambda: LSTMBaseModel()),
        ModelSpec("GRU", lambda: GRUBaseModel()),
        ModelSpec("Transformer", lambda: TransformerBaseModel()),
    ]


def _flatten_sequences(values: np.ndarray) -> np.ndarray:
    return values.reshape(values.shape[0], -1)
