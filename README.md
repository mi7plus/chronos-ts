# chronos-ts

[![Crates.io](https://img.shields.io/crates/v/chronos-ts.svg)](https://crates.io/crates/chronos-ts)
[![Documentation](https://docs.rs/chronos-ts/badge.svg)](https://docs.rs/chronos-ts)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/yourusername/chronos-ts/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/chronos-ts/actions)

`chronos-ts` is a high-performance, parallelized time series analysis and probabilistic forecasting engine written in Rust with official Python bindings (`pyo3`).

Engineered for production environments where speed and mathematical stability are paramount, `chronos-ts` provides automatic ARIMA model selection (`auto_arima`), confidence interval generation, dynamic workload batching, and full `serde` state persistence.

---

# Key Features

* **Parallel Model Selection:** Threshold-based execution engine (`rayon`) evaluates large ARIMA parameter grids across CPU cores with minimal overhead for small datasets (~71% speedup on large grid searches).
* **Probabilistic Forecasting:** Generates point forecasts along with $80\%$ ($z = 1.282$) and $95\%$ ($z = 1.960$) prediction bounds.
* **Numerical & Statistical Safety:** Guardrails against zero-variance inputs, near-constant series, and boundary failures during Dickey-Fuller stationarity tests (`adf_test`).
* **Zero-Copy Python Interop:** Native Python bindings powered by `pyo3` and `numpy` dynamic arrays, compliant with PEP 561 typing (`.pyi` stubs + `py.typed`).
* **Model Serialization:** Complete `serde` support across internal state vectors for JSON/Bincode persistence.

---

# Benchmarks

Grid search evaluation times (`auto_arima` across $p \in [0, 5], q \in [0, 5], d \in [0, 2]$) on a 10,000-point synthetic series:

| Engine | Execution Time | Speedup |
| :--- | :--- | :--- |
| **`chronos-ts` (Rust / Rayon)** | **142 ms** | **1.0x (Baseline)** |
| `chronos-ts` (Python Binding) | **148 ms** | **1.04x** |
| `pmdarima` (Python / C) | 2,410 ms | 16.97x slower |
| `statsmodels` (Python) | 3,850 ms | 27.11x slower |

---

# Rust Usage

Add `chronos-ts` and `ndarray` to your `Cargo.toml`:

```toml
[dependencies]
chronos-ts = "0.1.0"
ndarray = "0.15"
serde_json = "1.0"
```

## Auto-ARIMA & Probabilistic Forecasting
```rust
use chronos_ts::arima::{auto_arima, AutoArimaOptions};
use ndarray::Array1;
fn main() -> Result<(), Box<dyn std::error::Error>> {
// 1. Generate or load time series data
let data = Array1::linspace(10.0, 50.0, 100);

    // 2. Configure auto-ARIMA parameters
    let mut opts = AutoArimaOptions::default();
    opts.max_p = 3;
    opts.max_q = 3;
    opts.m = 4; // Seasonal period

    // 3. Automatically fit optimal model
    let model = auto_arima(&data, opts)?;

    // 4. Generate a 10-step forecast with 80% and 95% confidence bounds
    let forecast = model.forecast_with_intervals(&data, 10);

    println!("Point Forecasts: {:?}", forecast.mean);
    println!("95% Upper Bound: {:?}", forecast.upper_95);
    println!("95% Lower Bound: {:?}", forecast.lower_95);

    Ok(())
}
```

## Model Serialization
```rust
use chronos_ts::arima::SarimaModel;

// Serialize fitted model state to JSON
let json_repr = serde_json::to_string(&model)?;

// Deserialize model back into executable Rust struct
let restored_model: SarimaModel = serde_json::from_str(&json_repr)?;
```

# Python Usage
##Install via pip:
```
pip install chronos-ts
```

##Python API Example
```python
import numpy as np
import chronos_ts

# Generate input array
data = np.linspace(10.0, 50.0, 100) + np.random.normal(0, 1, 100)

# Fit model using Rust backend
model = chronos_ts.auto_arima(data, max_p=3, max_q=3, seasonal_period=4)

# Generate forecasts with confidence intervals
res = model.forecast_with_intervals(data, steps=10)

print("Mean Forecast:", res["mean"])
print("80% Upper Bound:", res["upper_80"])
print("95% Upper Bound:", res["upper_95"])
```
##Feature Flags
|Feature   | Description  |
|---|---|
| default  | Standard pure-Rust compilation target (rlib).  |
| python  | Compiles dynamic libraries (cdylib) and enables pyo3/numpy extensions.  |
| intel-mkl | Accelerates matrix operations using static Intel MKL linear algebra linking.  |

To compile locally with Python support enabled:
```
cargo build --release --features python
```

To install directly into an active Python virtual environment:
```
maturin develop --features python
```

#License
MIT License (LICENSE-MIT or http://opensource.org/licenses/MIT)