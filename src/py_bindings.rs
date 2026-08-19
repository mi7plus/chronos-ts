// src/py_bindings.rs
use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray, PyReadonlyArray1};
use crate::arima::{auto_arima as rust_auto_arima, AutoArimaOptions, SarimaModel as RustSarimaModel};

#[pyclass(name = "PySarimaModel")]
pub struct PySarimaModel {
    inner: RustSarimaModel,
}

#[pymethods]
impl PySarimaModel {
    /// Forecast future values along with 80% and 95% confidence intervals.
    /// Returns a dict containing numpy arrays: 'mean', 'lower_80', 'upper_80', 'lower_95', 'upper_95'
    pub fn forecast_with_intervals<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        steps: usize,
    ) -> PyResult<&'py pyo3::types::PyDict> {
        let data_slice = data.as_array().to_owned();
        let res = self.inner.forecast_with_intervals(&data_slice, steps);

        let dict = pyo3::types::PyDict::new(py);
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
pub fn py_auto_arima<'py>(
    _py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    max_p: Option<usize>,
    max_q: Option<usize>,
    seasonal_period: Option<usize>,
) -> PyResult<PySarimaModel> {
    let data_array = data.as_array().to_owned();

    let mut opts = AutoArimaOptions::default();
    if let Some(p) = max_p { opts.max_p = p; }
    if let Some(q) = max_q { opts.max_q = q; }
    if let Some(m) = seasonal_period { opts.m = m; }

    let model = rust_auto_arima(&data_array, opts)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PySarimaModel { inner: model })
}

/// The exposed Python module entrypoint
#[pymodule]
pub fn chronos_ts(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_auto_arima, m)?)?;
    m.add_class::<PySarimaModel>()?;
    Ok(())
}