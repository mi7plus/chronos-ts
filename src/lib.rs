//! # chronos-ts
//!
//! `chronos-ts` is a pure-Rust time-series modeling and forecasting library
//! (no LAPACK/BLAS/MKL dependency, so it builds anywhere).
//!
//! ## Features
//! - **Auto ARIMA / SARIMA / SARIMAX**: automated order selection with real
//!   coefficient estimation by Conditional Sum of Squares (default) or exact
//!   Gaussian maximum likelihood via the Kalman filter
//!   ([`EstimationMethod::Mle`](arima::EstimationMethod)). Includes a mean/drift
//!   term, coefficient standard errors, correct prediction intervals, and an
//!   optional Box-Cox transform.
//! - **Prophet decomposition**: structural time series with trend changepoints,
//!   Fourier seasonalities, holidays, logistic growth, and Monte-Carlo intervals.
//! - **Diagnostics & tuning**: residual ACF/PACF, Ljung-Box, Jarque-Bera, and
//!   cross-validated hyperparameter search.
//!
//! ## ARIMA quick start
//!
//! ```rust
//! use chronos_ts::arima::{auto_arima, AutoArimaOptions};
//! use ndarray::Array1;
//!
//! let series = Array1::from(vec![10.0, 12.0, 15.0, 14.0, 18.0, 20.0, 23.0]);
//! let opts = AutoArimaOptions {
//!     max_p: 2,
//!     max_d: 1,
//!     max_q: 2,
//!     ..Default::default()
//! };
//!
//! let model = auto_arima(&series, opts).expect("Model should fit successfully");
//! let forecast = model.forecast(&series, 3);
//! assert_eq!(forecast.len(), 3);
//! ```
//!
//! ## Prophet quick start
//!
//! ```rust
//! use chronos_ts::{ProphetDecomposition, SeasonalityMode};
//! use chrono::{Duration, NaiveDate};
//! use ndarray::Array1;
//!
//! let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
//! let dates: Vec<NaiveDate> = (0..30).map(|i| start + Duration::days(i)).collect();
//! let y = Array1::from_shape_fn(30, |i| 10.0 + 0.5 * i as f64);
//!
//! let mut model = ProphetDecomposition::new(5, 0.05);
//! model.seasonality_mode = SeasonalityMode::Additive;
//! model.add_seasonality("weekly", 7.0, 3);
//! model.fit(&dates, &y, None, None).unwrap();
//! let prediction = model.predict(&dates).unwrap();
//! assert_eq!(prediction.yhat.len(), 30);
//! ```

// `non_snake_case` is allowed only in the modules that use the standard SARIMA
// uppercase seasonal notation (P, D, Q); it is not enabled crate-wide.

#[cfg(feature = "python")]
pub mod py_bindings;

pub mod arima;
pub mod decomposition;
pub mod diagnostics;
pub mod errors;
pub mod stat_tests;
pub mod statespace;
pub mod statespace_mle;
pub mod tuning;
pub mod utils;
pub mod viz;

// Internal implementation details — not part of the public API.
pub(crate) mod arima_mle;
pub(crate) mod arima_poly;
pub(crate) mod linalg;

// Core public API exports
pub use arima::{
    arima_cross_validation, auto_arima, Arima, ArimaCvReport, ArimaOrder, AutoArimaOptions,
    EstimationMethod, InformationCriterion, SarimaModel, SarimaOrder,
};
pub use decomposition::{
    Holiday, ProphetDecomposition, ProphetPrediction, SeasonalityMode, TrendType,
};
pub use diagnostics::{
    CrossValidationEvaluator, CrossValidationReport, DiagnosticsResult, HorizonMetrics,
    JarqueBeraResult, LjungBoxResult, ResidualDiagnostics,
};
pub use errors::{ChronosError, Result};
pub use stat_tests::{adf_test, estimate_D, estimate_d, AdfTestResult};
pub use statespace::{KalmanFilter, KalmanFilterResult, StateSpaceModel};
pub use statespace_mle::{fit_state_space_mle, MleFitResult, StateSpaceLikelihoodCost};
pub use tuning::{
    AutoTuneResult, AutoTuner, HyperparameterCandidate, HyperparameterGrid, OptimizationMetric,
};
pub use viz::{ChartDataPoint, DecompositionExport, VisualizationExporter};
