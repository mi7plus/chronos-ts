# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Coefficient standard errors** on fitted models (`SarimaModel.std_errors`),
  computed from the objective's numerical Hessian.
- **`SarimaModel::residuals`** exposes in-sample innovation residuals, so the
  `diagnostics` module (ACF/PACF, Ljung-Box, Jarque-Bera) can be run on a fitted
  ARIMA model.
- **Box-Cox / log transform** (`SarimaModel::fit_transformed`); forecasts and
  intervals are automatically back-transformed to the original scale.
- **ARIMA rolling-origin cross-validation** (`arima_cross_validation` →
  `ArimaCvReport` with MAE/RMSE/MAPE).
- **MA invertibility** is now enforced alongside AR stationarity in the fit guard.
- **Input validation**: `fit`/`auto_arima`/cross-validation reject empty or
  non-finite (NaN/inf) input with a clear error.
- Python: `SarimaModel.std_errors` / `SarimaModel.residuals`.
- **Exact maximum-likelihood estimation** (`EstimationMethod::Mle`) via the Kalman
  filter with stationary (Lyapunov) initialization, selectable on `fit_with_method`
  and `AutoArimaOptions.estimation`. This connects the state-space machinery to the
  ARIMA engine as an alternative to the default Conditional Sum of Squares.
- **Intercept / drift term.** ARIMA now fits a mean (`d+D == 0`) or drift
  (`d+D == 1`), so series with a nonzero level no longer forecast toward zero.
- **Correct ARIMA prediction intervals** derived from the model's MA(∞) `psi`-weights
  (with differencing folded in), replacing the `sigma*sqrt(h)` random-walk heuristic.
- **Stationarity guard** on CSS estimates: explosive coefficients are shrunk toward
  a stable region (or zeroed) so forecasts cannot diverge.
- Runnable `examples/` (`arima_forecast`, `prophet_forecast`); a larger, multi-size
  `auto_arima` benchmark plus a Prophet benchmark.
- Python parity: `Prophet.add_holiday` / `Prophet.predict_with_intervals`, a
  `sarimax(...)` fitter with `SarimaModel.forecast_sarimax`, and an `mle=` flag on
  `auto_arima`.
- `Serialize`/`Deserialize` on `ForecastResult`.

### Fixed
- **Seasonal forecasts are integrated correctly.** Forecasts of a model with `D>0`
  now undo the seasonal differencing (previously only non-seasonal `d` was undone,
  leaving seasonal forecasts on the wrong scale).
- `estimate_D`'s documentation now honestly describes the seasonal-strength
  heuristic it implements (it is not an OCSB / Canova-Hansen unit-root test).
- `variance()` returns 0.0 (instead of NaN/inf) for series shorter than two points.
- **ARIMA/SARIMA now actually estimates coefficients.** Previously `SarimaModel::fit`
  left all AR/MA/seasonal coefficients at zero, so every "fit" produced a
  random-walk model and `auto_arima` selected orders on models that were all
  effectively identical. Coefficients are now estimated by Conditional Sum of
  Squares (CSS) using a derivative-free Nelder-Mead search.
- **Forecasts now include moving-average terms.** The forecast recursion reused
  in-sample residuals so MA/seasonal-MA components drive the first `q` (and
  `Q*m`) steps instead of being ignored.
- **Correct higher-order integration.** `integrate_forecast` now anchors each
  un-differencing level on the tail of the corresponding differenced history,
  fixing `d > 1` reconstruction.
- **ADF p-values use the Dickey-Fuller distribution** (tabulated quantiles for the
  constant case) instead of an incorrect Student-t approximation.
- **SARIMAX exogenous handling is correct end-to-end.** `forecast_with_intervals_exog`
  now takes the historical exogenous matrix to reconstruct the regression
  residual series and adds the future exogenous contribution back onto every band.
- **`auto_arima` no longer explores degenerate seasonal orders** when `m <= 1`
  (where seasonal terms alias the non-seasonal lags).

### Changed
- Tightened the public API: `linalg`, `arima_poly`, and `arima_mle` are now
  crate-private implementation details rather than public modules.
- **Removed the Intel MKL / `ndarray-linalg` (LAPACK/BLAS) dependency.** All linear
  algebra is now pure Rust (`src/linalg.rs`), so the crate builds on any platform
  with no C toolchain. This removes the previous hard requirement on a C compiler.
- Python bindings now expose the `Prophet` decomposition model (previously only the
  ARIMA engine was bound) and a richer `SarimaModel` (`order`, `sigma2`, `aic`,
  `forecast`).
- README corrected: accurate feature table, MIT license, real repository links,
  and honest benchmarking instructions (fabricated comparison numbers removed).

### Added
- Accuracy tests for the ARIMA engine (`tests/arima_accuracy_test.rs`) that verify
  coefficient recovery, forecast decay, and SARIMAX behaviour.
- CI now checks formatting (`cargo fmt --check`), lints (`cargo clippy -D warnings`),
  doctests, and compiles the Python extension.

### Removed
- Dead/duplicate modules: `auto_arima.rs` (orphaned duplicate of the live
  `arima::auto_arima`), and `prophet_matrix.rs` / `fitting.rs` (unused
  re-implementations of logic already inlined in `decomposition.rs`).
