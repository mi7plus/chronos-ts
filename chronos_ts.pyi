# chronos_ts.pyi
from typing import Dict, List, Optional, Tuple, TypedDict
import numpy as np
import numpy.typing as npt

class ForecastResultDict(TypedDict):
    """Dictionary containing point forecasts and confidence interval arrays."""
    mean: npt.NDArray[np.float64]
    lower_80: npt.NDArray[np.float64]
    upper_80: npt.NDArray[np.float64]
    lower_95: npt.NDArray[np.float64]
    upper_95: npt.NDArray[np.float64]

class ProphetPredictionDict(TypedDict):
    """Dictionary containing the decomposed forecast components."""
    yhat: npt.NDArray[np.float64]
    trend: npt.NDArray[np.float64]
    seasonal: npt.NDArray[np.float64]
    holidays: npt.NDArray[np.float64]

class SarimaModel:
    """Fitted SARIMA model instance returned by auto_arima."""

    @property
    def order(self) -> Tuple[int, int, int, int, int, int, int]:
        """The fitted (p, d, q, P, D, Q, m) order."""
        ...

    @property
    def sigma2(self) -> float:
        """Residual variance of the fitted model."""
        ...

    @property
    def std_errors(self) -> Optional[npt.NDArray[np.float64]]:
        """Asymptotic coefficient standard errors, or None if unavailable."""
        ...

    def aic(self) -> float:
        """Akaike Information Criterion of the fitted model."""
        ...

    def residuals(
        self, data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """In-sample innovation residuals (for diagnostics)."""
        ...

    def forecast(
        self,
        data: npt.NDArray[np.float64],
        steps: int,
    ) -> npt.NDArray[np.float64]:
        """Point forecast `steps` periods ahead."""
        ...

    def forecast_with_intervals(
        self,
        data: npt.NDArray[np.float64],
        steps: int,
    ) -> ForecastResultDict:
        """Forecast with 80% and 95% confidence bounds."""
        ...

    def forecast_sarimax(
        self,
        data: npt.NDArray[np.float64],
        exog_hist: npt.NDArray[np.float64],
        exog_future: npt.NDArray[np.float64],
        steps: int,
    ) -> ForecastResultDict:
        """SARIMAX forecast (for a model fitted via `sarimax`)."""
        ...

def auto_arima(
    data: npt.NDArray[np.float64],
    max_p: Optional[int] = ...,
    max_q: Optional[int] = ...,
    max_d: Optional[int] = ...,
    seasonal_period: Optional[int] = ...,
    mle: bool = ...,
) -> SarimaModel:
    """Automatically fits the optimal SARIMA model to a 1D NumPy array.

    Set `mle=True` for exact maximum-likelihood estimation (slower, more
    statistically efficient); the default uses Conditional Sum of Squares.
    """
    ...

def sarimax(
    data: npt.NDArray[np.float64],
    exog: npt.NDArray[np.float64],
    p: int,
    d: int,
    q: int,
) -> SarimaModel:
    """Fits a fixed-order SARIMAX model with exogenous regressors."""
    ...

class Prophet:
    """Prophet-style structural decomposition (trend + seasonality + holidays)."""

    def __init__(
        self,
        n_changepoints: int = 25,
        changepoint_prior_scale: float = 0.05,
    ) -> None: ...

    def set_multiplicative(self) -> None:
        """Switch to multiplicative seasonality."""
        ...

    def add_seasonality(
        self,
        name: str,
        period_days: float,
        fourier_order: int,
        prior_scale: float = 10.0,
    ) -> None:
        """Add a Fourier seasonality component."""
        ...

    def add_holiday(
        self,
        name: str,
        dates: List[str],
        lower_window: int = 0,
        upper_window: int = 0,
    ) -> None:
        """Add a holiday effect over the given ISO dates (with optional windows)."""
        ...

    def fit(self, dates: List[str], y: npt.NDArray[np.float64]) -> None:
        """Fit the model. `dates` are ISO strings ('YYYY-MM-DD')."""
        ...

    def predict(self, dates: List[str]) -> ProphetPredictionDict:
        """Predict decomposed components for the given ISO date strings."""
        ...

    def predict_with_intervals(
        self,
        dates: List[str],
        interval_width: float = 0.95,
        n_samples: int = 1000,
    ) -> ProphetPredictionDict:
        """Predict with Monte-Carlo uncertainty intervals (adds yhat_lower/upper)."""
        ...

    def to_json(self) -> str:
        """Serialize the fitted model to a JSON string."""
        ...
