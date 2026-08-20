// src/py_bindings.rs
//
// Python bindings for chronos-ts. Exposes both the ARIMA/auto-ARIMA engine and
// the Prophet-style structural decomposition model.

// The `non_local_definitions` warning below is emitted from inside pyo3 0.20's
// `#[pymethods]`/`#[pyclass]` macro expansion on newer compilers, not from our
// own code; silence it here (guarded by `unknown_lints` for older toolchains).
#![allow(unknown_lints)]
#![allow(non_local_definitions)]

use crate::arima::{
    auto_arima as rust_auto_arima, AutoArimaOptions, EstimationMethod,
    SarimaModel as RustSarimaModel, SarimaOrder,
};
use crate::decomposition::{Holiday, ProphetDecomposition, SeasonalityMode};
use chrono::NaiveDate;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Parses a list of ISO-8601 (`YYYY-MM-DD`) date strings into `NaiveDate`s.
fn parse_dates(dates: &[String]) -> PyResult<Vec<NaiveDate>> {
    dates
        .iter()
        .map(|s| {
            NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid date '{s}': {e}"))
            })
        })
        .collect()
}

#[pyclass(name = "SarimaModel")]
pub struct PySarimaModel {
    inner: RustSarimaModel,
}

#[pymethods]
impl PySarimaModel {
    /// The fitted (p, d, q, P, D, Q, m) order.
    #[getter]
    fn order(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        let o = &self.inner.order;
        (o.p, o.d, o.q, o.P, o.D, o.Q, o.m)
    }

    /// Residual variance of the fitted model.
    #[getter]
    fn sigma2(&self) -> f64 {
        self.inner.sigma2
    }

    /// Asymptotic coefficient standard errors `[ar.., ma.., sar.., sma..]`, or None.
    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> Option<&'py numpy::PyArray1<f64>> {
        self.inner.std_errors.as_ref().map(|se| se.to_pyarray(py))
    }

    /// Akaike Information Criterion of the fitted model.
    fn aic(&self) -> f64 {
        self.inner.aic()
    }

    /// In-sample innovation residuals (for diagnostics).
    fn residuals<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
    ) -> &'py numpy::PyArray1<f64> {
        let d = data.as_array().to_owned();
        self.inner.residuals(&d).to_pyarray(py)
    }

    /// Point forecast `steps` periods ahead.
    fn forecast<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        steps: usize,
    ) -> &'py numpy::PyArray1<f64> {
        let history = data.as_array().to_owned();
        self.inner.forecast(&history, steps).to_pyarray(py)
    }

    /// Forecast future values along with 80% and 95% confidence intervals.
    /// Returns a dict of numpy arrays: 'mean', 'lower_80', 'upper_80',
    /// 'lower_95', 'upper_95'.
    fn forecast_with_intervals<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        steps: usize,
    ) -> PyResult<&'py PyDict> {
        let history = data.as_array().to_owned();
        let res = self.inner.forecast_with_intervals(&history, steps);

        let dict = PyDict::new(py);
        dict.set_item("mean", res.mean.to_pyarray(py))?;
        dict.set_item("lower_80", res.lower_80.to_pyarray(py))?;
        dict.set_item("upper_80", res.upper_80.to_pyarray(py))?;
        dict.set_item("lower_95", res.lower_95.to_pyarray(py))?;
        dict.set_item("upper_95", res.upper_95.to_pyarray(py))?;
        Ok(dict)
    }

    /// SARIMAX forecast for a model fitted via `sarimax`. `exog_hist` (aligned with
    /// `data`) and `exog_future` (rows == steps) are 2D arrays of regressors. Returns
    /// the same dict as `forecast_with_intervals`.
    fn forecast_sarimax<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        exog_hist: PyReadonlyArray2<'py, f64>,
        exog_future: PyReadonlyArray2<'py, f64>,
        steps: usize,
    ) -> PyResult<&'py PyDict> {
        let history = data.as_array().to_owned();
        let x_hist = exog_hist.as_array().to_owned();
        let x_fut = exog_future.as_array().to_owned();
        let res = self
            .inner
            .forecast_with_intervals_exog(&history, Some(&x_hist), Some(&x_fut), steps)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("mean", res.mean.to_pyarray(py))?;
        dict.set_item("lower_80", res.lower_80.to_pyarray(py))?;
        dict.set_item("upper_80", res.upper_80.to_pyarray(py))?;
        dict.set_item("lower_95", res.lower_95.to_pyarray(py))?;
        dict.set_item("upper_95", res.upper_95.to_pyarray(py))?;
        Ok(dict)
    }
}

/// Automatically fits a SARIMA model to a NumPy array.
#[pyfunction]
#[pyo3(name = "auto_arima")]
#[pyo3(signature = (data, max_p=None, max_q=None, max_d=None, seasonal_period=None, mle=false))]
pub fn py_auto_arima(
    data: PyReadonlyArray1<'_, f64>,
    max_p: Option<usize>,
    max_q: Option<usize>,
    max_d: Option<usize>,
    seasonal_period: Option<usize>,
    mle: bool,
) -> PyResult<PySarimaModel> {
    let data_array = data.as_array().to_owned();

    let mut opts = AutoArimaOptions::default();
    if let Some(p) = max_p {
        opts.max_p = p;
    }
    if let Some(q) = max_q {
        opts.max_q = q;
    }
    if let Some(d) = max_d {
        opts.max_d = d;
    }
    if let Some(m) = seasonal_period {
        opts.m = m;
    }
    opts.estimation = if mle {
        EstimationMethod::Mle
    } else {
        EstimationMethod::Css
    };

    let model = rust_auto_arima(&data_array, opts)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PySarimaModel { inner: model })
}

/// Fits a fixed-order SARIMAX model `(p, d, q)` with exogenous regressors `exog`
/// (a 2D array aligned with `data`). Use `SarimaModel.forecast_sarimax` to forecast.
#[pyfunction]
#[pyo3(name = "sarimax")]
#[pyo3(signature = (data, exog, p, d, q))]
pub fn py_sarimax(
    data: PyReadonlyArray1<'_, f64>,
    exog: PyReadonlyArray2<'_, f64>,
    p: usize,
    d: usize,
    q: usize,
) -> PyResult<PySarimaModel> {
    let y = data.as_array().to_owned();
    let x = exog.as_array().to_owned();
    let model = RustSarimaModel::fit_with_exog(&y, Some(&x), SarimaOrder::arima(p, d, q))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PySarimaModel { inner: model })
}

/// Prophet-style structural time-series decomposition (trend + seasonality + holidays).
#[pyclass(name = "Prophet")]
pub struct PyProphet {
    inner: ProphetDecomposition,
}

#[pymethods]
impl PyProphet {
    #[new]
    #[pyo3(signature = (n_changepoints=25, changepoint_prior_scale=0.05))]
    fn new(n_changepoints: usize, changepoint_prior_scale: f64) -> Self {
        Self {
            inner: ProphetDecomposition::new(n_changepoints, changepoint_prior_scale),
        }
    }

    /// Enables multiplicative (rather than additive) seasonality.
    fn set_multiplicative(&mut self) {
        self.inner.seasonality_mode = SeasonalityMode::Multiplicative;
    }

    /// Adds a Fourier seasonality component (e.g. weekly: period=7, order=3).
    #[pyo3(signature = (name, period_days, fourier_order, prior_scale=10.0))]
    fn add_seasonality(
        &mut self,
        name: &str,
        period_days: f64,
        fourier_order: usize,
        prior_scale: f64,
    ) {
        self.inner
            .add_seasonality_with_prior(name, period_days, fourier_order, prior_scale);
    }

    /// Adds a holiday effect. `dates` are ISO strings; `lower_window`/`upper_window`
    /// extend the effect to neighbouring days (e.g. -1 and 1 for the day before/after).
    #[pyo3(signature = (name, dates, lower_window=0, upper_window=0))]
    fn add_holiday(
        &mut self,
        name: &str,
        dates: Vec<String>,
        lower_window: i64,
        upper_window: i64,
    ) -> PyResult<()> {
        let parsed = parse_dates(&dates)?;
        self.inner.add_holiday(Holiday {
            name: name.to_string(),
            dates: parsed,
            lower_window,
            upper_window,
        });
        Ok(())
    }

    /// Fits the model. `dates` are ISO strings ("YYYY-MM-DD"); `y` is the target.
    fn fit(&mut self, dates: Vec<String>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let parsed = parse_dates(&dates)?;
        let y_arr = y.as_array().to_owned();
        self.inner
            .fit(&parsed, &y_arr, None, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Predicts for the given ISO date strings, returning a dict of numpy arrays:
    /// 'yhat', 'trend', 'seasonal', 'holidays'.
    fn predict<'py>(&self, py: Python<'py>, dates: Vec<String>) -> PyResult<&'py PyDict> {
        let parsed = parse_dates(&dates)?;
        let pred = self
            .inner
            .predict(&parsed)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("yhat", pred.yhat.to_pyarray(py))?;
        dict.set_item("trend", pred.trend.to_pyarray(py))?;
        dict.set_item("seasonal", pred.seasonal.to_pyarray(py))?;
        dict.set_item("holidays", pred.holidays.to_pyarray(py))?;
        Ok(dict)
    }

    /// Predicts with Monte-Carlo uncertainty intervals. Returns the same keys as
    /// `predict` plus 'yhat_lower' and 'yhat_upper'.
    #[pyo3(signature = (dates, interval_width=0.95, n_samples=1000))]
    fn predict_with_intervals<'py>(
        &self,
        py: Python<'py>,
        dates: Vec<String>,
        interval_width: f64,
        n_samples: usize,
    ) -> PyResult<&'py PyDict> {
        let parsed = parse_dates(&dates)?;
        let pred = self
            .inner
            .predict_with_intervals(&parsed, interval_width, n_samples)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("yhat", pred.yhat.to_pyarray(py))?;
        dict.set_item("trend", pred.trend.to_pyarray(py))?;
        dict.set_item("seasonal", pred.seasonal.to_pyarray(py))?;
        dict.set_item("holidays", pred.holidays.to_pyarray(py))?;
        if let Some(lower) = pred.yhat_lower {
            dict.set_item("yhat_lower", lower.to_pyarray(py))?;
        }
        if let Some(upper) = pred.yhat_upper {
            dict.set_item("yhat_upper", upper.to_pyarray(py))?;
        }
        Ok(dict)
    }

    /// Serializes the fitted model to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

/// The exposed Python module entrypoint.
#[pymodule]
pub fn chronos_ts(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_auto_arima, m)?)?;
    m.add_function(wrap_pyfunction!(py_sarimax, m)?)?;
    m.add_class::<PySarimaModel>()?;
    m.add_class::<PyProphet>()?;
    Ok(())
}
