# chronos_ts.pyi
from typing import TypedDict, Optional
import numpy as np
import numpy.typing as npt

class ForecastResultDict(TypedDict):
    """Dictionary containing point forecasts and confidence interval arrays."""
    mean: npt.NDArray[np.float64]
    lower_80: npt.NDArray[np.float64]
    upper_80: npt.NDArray[np.float64]
    lower_95: npt.NDArray[np.float64]
    upper_95: npt.NDArray[np.float64]

class PySarimaModel:
    """Fitted SARIMA model instance returned by auto_arima."""

    def forecast_with_intervals(
        self,
        data: npt.NDArray[np.float64],
        steps: int
    ) -> ForecastResultDict:
        """
        Generates future point forecasts alongside 80% and 95% confidence bounds.

        Parameters:
            data: Historical time series data as a 1D NumPy array.
            steps: Number of future time steps to forecast.

        Returns:
            A dictionary containing 'mean', 'lower_80', 'upper_80', 'lower_95', and 'upper_95' NumPy arrays.
        """
        ...

def auto_arima(
    data: npt.NDArray[np.float64],
    max_p: Optional[int] = ...,
    max_q: Optional[int] = ...,
    seasonal_period: Optional[int] = ...
) -> PySarimaModel:
    """
    Automatically fits the optimal SARIMA model to a 1D NumPy array.

    Parameters:
        data: Input 1D NumPy floating-point array.
        max_p: Maximum autoregressive (AR) order to evaluate.
        max_q: Maximum moving average (MA) order to evaluate.
        seasonal_period: Length of seasonal cycle (m).

    Returns:
        A fitted PySarimaModel instance.
    """
    ...