//! # chronos-ts
//!
//! `chronos-ts` is a high-performance time-series modeling and forecasting library written in Rust.
//!
//! ## Features
//! - **Auto ARIMA / SARIMA**: Automated order selection and parameters estimation.
//! - **Prophet Decomposition**: Structural time series decomposition.
//!
//! ## Quick Start
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

#![allow(non_snake_case)]

#[cfg(feature = "python")]
pub mod py_bindings;

pub mod arima;
pub mod decomposition;
pub mod diagnostics;
pub mod errors;
pub mod stat_tests;
pub mod statespace;
pub mod statespace_mle;
pub mod utils;
pub mod viz;
pub mod tuning;
pub mod prophet_matrix;  // If matrix builder is separate
pub mod fitting; // Exposes fitting.rs to the rest of the crate

// Core public API exports
pub use arima::{
    auto_arima, Arima, ArimaOrder, AutoArimaOptions, InformationCriterion, SarimaModel,
    SarimaOrder,
};
pub use decomposition::{Holiday, ProphetDecomposition, TrendType, SeasonalityMode, ProphetPrediction};
pub use errors::{ChronosError, Result};
pub use stat_tests::{adf_test, estimate_d, estimate_D, AdfTestResult};
pub use statespace::{KalmanFilter, KalmanFilterResult, StateSpaceModel};
pub use diagnostics::{
    CrossValidationEvaluator, CrossValidationReport, DiagnosticsResult, HorizonMetrics,
    JarqueBeraResult, LjungBoxResult, ResidualDiagnostics,
};
pub use statespace_mle::{fit_state_space_mle, MleFitResult, StateSpaceLikelihoodCost};
pub use tuning::{AutoTuneResult, AutoTuner, HyperparameterCandidate, HyperparameterGrid, OptimizationMetric};
pub use viz::{ChartDataPoint, DecompositionExport, VisualizationExporter};
