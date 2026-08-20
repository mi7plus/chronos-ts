# chronos-ts

[![Crates.io](https://img.shields.io/crates/v/chronos-ts.svg)](https://crates.io/crates/chronos-ts)
[![Documentation](https://docs.rs/chronos-ts/badge.svg)](https://docs.rs/chronos-ts)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/mi7plus/chronos-ts/actions/workflows/ci.yaml/badge.svg)](https://github.com/mi7plus/chronos-ts/actions)

`chronos-ts` is a pure-Rust, parallelized time-series analysis and forecasting
library with optional Python bindings (`pyo3`). It ships two complementary
modelling engines:

* **Auto-ARIMA / SARIMA / SARIMAX** — automatic order selection with real
  coefficient estimation by Conditional Sum of Squares (CSS).
* **Prophet-style decomposition** — additive/multiplicative structural models
  with trend changepoints, Fourier seasonalities, holidays, logistic growth and
  Monte-Carlo uncertainty intervals.

It has **no LAPACK/BLAS/MKL dependency** and needs no C toolchain — all linear
algebra is implemented in Rust, so it builds cleanly on every platform.

---

## Key Features

* **Real ARIMA estimation, two ways.** Coefficients are fitted by fast Conditional
  Sum of Squares (default) or exact Gaussian **maximum likelihood** via the Kalman
  filter (`EstimationMethod::Mle`). `auto_arima` chooses the order by AIC/AICc/BIC
  using a stepwise (or full-grid) search, parallelized with `rayon`. A mean/drift
  term is estimated automatically, and a stationarity guard keeps fits stable.
* **SARIMAX.** Exogenous regressors are supported end-to-end (fit and forecast).
* **Uncertainty everywhere.** Coefficient standard errors, 80%/95% prediction
  intervals from the model's exact MA(∞) forecast variance, and an optional
  Box-Cox/log transform (forecasts auto back-transformed).
* **Diagnostics & validation.** In-sample residuals feed ACF/PACF, Ljung-Box and
  Jarque-Bera tests; `arima_cross_validation` gives rolling-origin MAE/RMSE/MAPE.
* **Statistical safety.** Guards against zero-variance/near-constant inputs; the
  Augmented Dickey-Fuller test maps its statistic through the Dickey-Fuller
  distribution (not a naive t-distribution).
* **Serde persistence.** Full JSON serialization of fitted model state.
* **Python bindings.** `auto_arima`, `SarimaModel` and `Prophet` exposed through
  `pyo3`/`numpy`, PEP 561 typed (`.pyi` stubs + `py.typed`).

---

## Examples & Benchmarks

Runnable examples live in [`examples/`](examples):

```bash
cargo run --example arima_forecast
cargo run --example prophet_forecast
```

Criterion benchmarks (`auto_arima` across several series sizes, plus a Prophet fit):

```bash
cargo bench
```

Results are written to `target/criterion/` (HTML reports enabled). Numbers are
hardware-dependent, so none are quoted here — measure on your own machine.

---

## Rust Usage

```toml
[dependencies]
chronos-ts = "0.1"
ndarray = "0.15"
serde_json = "1.0"
```

### Auto-ARIMA & Probabilistic Forecasting

```rust
use chronos_ts::arima::{auto_arima, AutoArimaOptions};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load your time series.
    let data = Array1::linspace(10.0, 50.0, 100);

    // 2. Configure the search.
    let opts = AutoArimaOptions {
        max_p: 3,
        max_q: 3,
        ..Default::default()
    };

    // 3. Fit the optimal model (coefficients estimated by CSS).
    let model = auto_arima(&data, opts)?;
    println!("Selected order: {:?}", model.order);

    // 4. Forecast 10 steps with 80% and 95% bounds.
    let forecast = model.forecast_with_intervals(&data, 10);
    println!("Point forecast: {:?}", forecast.mean);
    println!("95% interval:   [{:?}, {:?}]", forecast.lower_95, forecast.upper_95);

    Ok(())
}
```

### Prophet-style Decomposition

```rust
use chronos_ts::{ProphetDecomposition, SeasonalityMode};
use chrono::{Duration, NaiveDate};
use ndarray::Array1;

let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
let dates: Vec<NaiveDate> = (0..90).map(|i| start + Duration::days(i)).collect();
let y = Array1::from_shape_fn(90, |i| {
    10.0 + 0.1 * i as f64 + 2.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin()
});

let mut model = ProphetDecomposition::new(10, 0.05);
model.seasonality_mode = SeasonalityMode::Additive;
model.add_seasonality("weekly", 7.0, 3);
model.fit(&dates, &y, None, None).unwrap();

let prediction = model.predict(&dates).unwrap();
println!("yhat: {:?}", prediction.yhat);
```

### Model Serialization

```rust
use chronos_ts::arima::SarimaModel;

let json_repr = serde_json::to_string(&model)?;
let restored: SarimaModel = serde_json::from_str(&json_repr)?;
```

---

## Python Usage

Install (built with [maturin](https://github.com/PyO3/maturin)):

```bash
pip install chronos-ts
```

```python
import numpy as np
import chronos_ts

# --- ARIMA ---
data = np.linspace(10.0, 50.0, 100) + np.random.normal(0, 1, 100)
model = chronos_ts.auto_arima(data, max_p=3, max_q=3)
print("order:", model.order)

res = model.forecast_with_intervals(data, steps=10)
print("mean:", res["mean"])
print("95% upper:", res["upper_95"])

# --- Prophet ---
dates = [f"2023-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(90)]
y = np.array([10.0 + 0.1 * i for i in range(90)])

prophet = chronos_ts.Prophet(n_changepoints=10, changepoint_prior_scale=0.05)
prophet.add_seasonality("weekly", 7.0, 3)
prophet.fit(dates, y)
pred = prophet.predict(dates)
print("yhat:", pred["yhat"])
```

### Feature Flags

| Feature   | Description                                                              |
| --------- | ------------------------------------------------------------------------ |
| `default` | Pure-Rust library (`rlib`). No C toolchain or BLAS/LAPACK required.       |
| `python`  | Builds the `cdylib` and enables the `pyo3`/`numpy` Python extension.      |

Build the Python extension locally:

```bash
maturin develop --features python
```

---

## License

MIT — see [LICENSE](LICENSE).
