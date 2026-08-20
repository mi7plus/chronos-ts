#![allow(non_snake_case)]
use crate::arima_poly;
use crate::errors::{ChronosError, Result};
use crate::linalg;
use crate::stat_tests::{estimate_D, estimate_d};
use crate::utils::{
    box_cox, difference, integrate_forecast, inv_box_cox, seasonal_difference,
    seasonal_integrate_forecast,
};
use argmin::core::{CostFunction, Executor, State};
use argmin::solver::neldermead::NelderMead;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarimaOrder {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    pub P: usize,
    pub D: usize,
    pub Q: usize,
    pub m: usize, // Seasonal period (e.g., 4 = quarterly, 12 = monthly, 1 = non-seasonal)
}

impl SarimaOrder {
    /// Convenience constructor for non-seasonal ARIMA orders
    pub fn arima(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            P: 0,
            D: 0,
            Q: 0,
            m: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    #[serde(with = "crate::utils::serde_array1")]
    pub mean: Array1<f64>,
    #[serde(with = "crate::utils::serde_array1")]
    pub lower_80: Array1<f64>,
    #[serde(with = "crate::utils::serde_array1")]
    pub upper_80: Array1<f64>,
    #[serde(with = "crate::utils::serde_array1")]
    pub lower_95: Array1<f64>,
    #[serde(with = "crate::utils::serde_array1")]
    pub upper_95: Array1<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarimaModel {
    pub order: SarimaOrder,

    #[serde(with = "crate::utils::serde_array1")]
    pub ar_coeffs: Array1<f64>,

    #[serde(with = "crate::utils::serde_array1")]
    pub ma_coeffs: Array1<f64>,

    #[serde(with = "crate::utils::serde_array1")]
    pub sar_coeffs: Array1<f64>,

    #[serde(with = "crate::utils::serde_array1")]
    pub sma_coeffs: Array1<f64>,

    pub sigma2: f64,
    pub log_likelihood: f64,

    /// Mean of the (differenced) series, added back during forecasting. Estimated
    /// only when total differencing `d + D <= 1` (a mean for `d+D == 0`, a drift
    /// for `d+D == 1`); otherwise `0.0`.
    #[serde(default)]
    pub intercept: f64,

    /// Asymptotic standard errors of the estimated coefficients, in the order
    /// `[ar.., ma.., sar.., sma..]`. `None` if they could not be computed.
    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub std_errors: Option<Array1<f64>>,

    /// Box-Cox transform parameter applied to the data before fitting (`Some(0.0)`
    /// is a log transform); forecasts are back-transformed. `None` means no transform.
    #[serde(default)]
    pub transform: Option<f64>,

    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub exog_beta: Option<Array1<f64>>,
}

/// Whether a constant/mean term is included for a given order. A mean is fitted
/// when `d + D == 0`, and a drift when `d + D == 1`; higher orders of differencing
/// would imply polynomial drift and are left mean-free (matching common practice).
fn includes_mean(order: &SarimaOrder) -> bool {
    order.d + order.D <= 1
}

/// Type aliases allowing callers to use Arima and Sarima interchangeably
pub type Arima = SarimaModel;
pub type ArimaOrder = SarimaOrder;

impl SarimaModel {
    /// Primary constructor for fitting SARIMA/ARIMA models (Conditional Sum of
    /// Squares estimation).
    pub fn fit(data: &Array1<f64>, order: SarimaOrder) -> Result<Self> {
        Self::fit_with_method(data, order, EstimationMethod::Css)
    }

    /// Fits a SARIMA/ARIMA model with an explicit estimation method.
    ///
    /// - [`EstimationMethod::Css`] — fast Conditional Sum of Squares (default).
    /// - [`EstimationMethod::Mle`] — exact Gaussian maximum likelihood via the
    ///   Kalman filter; slower but statistically more efficient on short series.
    ///   Falls back to CSS if the likelihood optimization fails.
    pub fn fit_with_method(
        data: &Array1<f64>,
        order: SarimaOrder,
        method: EstimationMethod,
    ) -> Result<Self> {
        Self::fit_full(data, order, method, None)
    }

    /// Fits a model to Box-Cox-transformed data (`transform` = lambda; `Some(0.0)`
    /// is a log transform). Requires strictly positive input; forecasts produced by
    /// the returned model are automatically back-transformed to the original scale.
    pub fn fit_transformed(
        data: &Array1<f64>,
        order: SarimaOrder,
        method: EstimationMethod,
        transform: Option<f64>,
    ) -> Result<Self> {
        Self::fit_full(data, order, method, transform)
    }

    fn fit_full(
        data: &Array1<f64>,
        order: SarimaOrder,
        method: EstimationMethod,
        transform: Option<f64>,
    ) -> Result<Self> {
        validate_series(data)?;

        // Optional Box-Cox transform (requires positive data).
        let work = match transform {
            Some(lambda) => {
                if data.iter().any(|&x| x <= 0.0) {
                    return Err(ChronosError::InvalidParameters(
                        "Box-Cox transform requires strictly positive data.".into(),
                    ));
                }
                box_cox(data, lambda)
            }
            None => data.clone(),
        };

        let min_required =
            order.p + order.q + order.d + (order.P + order.Q + order.D) * order.m.max(1);
        if work.len() <= min_required {
            return Err(ChronosError::InsufficientData {
                required: min_required + 1,
                found: work.len(),
            });
        }

        // 1. Non-seasonal differencing
        let mut transformed = difference(&work, order.d);

        // 2. Seasonal differencing (skipped if m <= 1 or D == 0)
        if order.m > 1 && order.D > 0 {
            transformed = seasonal_difference(&transformed, order.m, order.D);
        }

        let n = transformed.len();

        // Center the (differenced) series by its mean so the ARMA part is fitted on
        // a zero-mean series; the mean is added back when forecasting. This gives the
        // model an intercept (d+D == 0) or a drift term (d+D == 1).
        let intercept = if includes_mean(&order) {
            transformed.mean().unwrap_or(0.0)
        } else {
            0.0
        };
        let centered = if intercept != 0.0 {
            &transformed - intercept
        } else {
            transformed.clone()
        };

        // CSS estimate (also the fallback for a failed MLE run).
        let css = |order: &SarimaOrder| {
            let (ar, ma, sar, sma) = estimate_css(&centered, order);
            let (_residuals, sse) = Self::compute_residuals(&centered, &ar, &ma, &sar, &sma, order);
            let sigma2 = (sse / (n as f64)).max(1e-8);
            let log_like = -0.5 * (n as f64) * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0);
            (ar, ma, sar, sma, sigma2, log_like)
        };

        let (ar_coeffs, ma_coeffs, sar_coeffs, sma_coeffs, sigma2, log_like) = match method {
            EstimationMethod::Css => css(&order),
            EstimationMethod::Mle => match crate::arima_mle::estimate_mle(&centered, &order) {
                Some(est) => (
                    est.ar,
                    est.ma,
                    est.sar,
                    est.sma,
                    est.sigma2.max(1e-8),
                    est.log_likelihood,
                ),
                None => css(&order),
            },
        };

        // Asymptotic coefficient standard errors from the (concentrated) objective's
        // numerical Hessian. Both estimators share the same Gaussian objective form.
        let params = concat_params(&ar_coeffs, &ma_coeffs, &sar_coeffs, &sma_coeffs);
        let std_errors = hessian_std_errors(
            |p| {
                let (ar, ma, sar, sma) = split_params(p, &order);
                let (_res, sse) = Self::compute_residuals(&centered, &ar, &ma, &sar, &sma, &order);
                0.5 * (n as f64) * sse.max(1e-12).ln()
            },
            &params,
        );

        Ok(Self {
            order,
            ar_coeffs,
            ma_coeffs,
            sar_coeffs,
            sma_coeffs,
            sigma2,
            log_likelihood: log_like,
            intercept,
            std_errors,
            transform,
            exog_beta: None,
        })
    }

    /// Returns the model's in-sample one-step innovation residuals (on the
    /// differenced, mean-centered scale the ARMA part was fitted on). These feed the
    /// residual diagnostics in [`crate::diagnostics`].
    pub fn residuals(&self, data: &Array1<f64>) -> Array1<f64> {
        let work = match self.transform {
            Some(lambda) => box_cox(data, lambda),
            None => data.clone(),
        };
        let mut w = difference(&work, self.order.d);
        if self.order.m > 1 && self.order.D > 0 {
            w = seasonal_difference(&w, self.order.m, self.order.D);
        }
        let centered = if self.intercept != 0.0 {
            &w - self.intercept
        } else {
            w
        };
        let (residuals, _sse) = Self::compute_residuals(
            &centered,
            &self.ar_coeffs,
            &self.ma_coeffs,
            &self.sar_coeffs,
            &self.sma_coeffs,
            &self.order,
        );
        residuals
    }

    /// Fits a SARIMAX model. If `exog` is provided, it first removes the linear trend
    /// contributed by X and then fits the SARIMA parameters on the regression residuals.
    pub fn fit_with_exog(
        y: &Array1<f64>,
        exog: Option<&Array2<f64>>,
        order: SarimaOrder,
    ) -> Result<Self> {
        // Fixed: Changed return type to crate Result<Self>
        let (y_residuals, exog_beta) = if let Some(x) = exog {
            if x.nrows() != y.len() {
                return Err(ChronosError::ConvergenceFailure(
                    "Exogenous matrix row count must match target array length.".into(),
                ));
            }
            let beta = fit_ols(x, y)?;
            let residuals = y - &x.dot(&beta);
            (residuals, Some(beta))
        } else {
            (y.clone(), None)
        };

        // Fit standard SARIMA on residual time series
        let mut model = Self::fit(&y_residuals, order)?;
        model.exog_beta = exog_beta;

        Ok(model)
    }

    /// Forecasts a SARIMAX model fitted via [`Self::fit_with_exog`].
    ///
    /// The SARIMA dynamics were estimated on the regression residuals
    /// `eta = y - X * beta`, so forecasting requires the historical exogenous
    /// matrix `exog_hist` to reconstruct that residual series, then adds the
    /// future exogenous contribution `X_fut * beta` back onto every band.
    ///
    /// For a model fitted without exogenous variables this reduces to
    /// [`Self::forecast_with_intervals`]; `exog_hist`/`exog_future` are ignored.
    pub fn forecast_with_intervals_exog(
        &self,
        data: &Array1<f64>,
        exog_hist: Option<&Array2<f64>>,
        exog_future: Option<&Array2<f64>>,
        steps: usize,
    ) -> Result<ForecastResult> {
        // Reconstruct the residual series the SARIMA part was actually fitted on.
        let residual_series = match (&self.exog_beta, exog_hist) {
            (Some(beta), Some(x_hist)) => {
                if x_hist.nrows() != data.len() {
                    return Err(ChronosError::ConvergenceFailure(
                        "Historical exogenous matrix rows must match the data length.".into(),
                    ));
                }
                if x_hist.ncols() != beta.len() {
                    return Err(ChronosError::ConvergenceFailure(
                        "Historical exogenous matrix columns must match fitted beta length.".into(),
                    ));
                }
                data - &x_hist.dot(beta)
            }
            (Some(_), None) => {
                return Err(ChronosError::ConvergenceFailure(
                    "Model was fitted with exogenous features; exog_hist is required to forecast."
                        .into(),
                ));
            }
            (None, _) => data.clone(),
        };

        // Base SARIMA point forecast and prediction intervals on the residual series.
        let mut forecast = self.forecast_with_intervals(&residual_series, steps);

        // Add the future exogenous contribution: y_fut = eta_fut + X_fut * beta.
        if let (Some(beta), Some(x_fut)) = (&self.exog_beta, exog_future) {
            if x_fut.nrows() != steps {
                return Err(ChronosError::ConvergenceFailure(
                    "Future exogenous matrix rows must equal requested forecast steps.".into(),
                ));
            }
            if x_fut.ncols() != beta.len() {
                return Err(ChronosError::ConvergenceFailure(
                    "Future exogenous matrix columns must match fitted beta length.".into(),
                ));
            }
            let exog_impact = x_fut.dot(beta);

            forecast.mean += &exog_impact;
            forecast.lower_80 += &exog_impact;
            forecast.upper_80 += &exog_impact;
            forecast.lower_95 += &exog_impact;
            forecast.upper_95 += &exog_impact;
        } else if self.exog_beta.is_some() {
            return Err(ChronosError::ConvergenceFailure(
                "Model was fitted with exogenous features; exog_future is required to forecast."
                    .into(),
            ));
        }

        Ok(forecast)
    }

    pub fn forecast_with_intervals(&self, data: &Array1<f64>, steps: usize) -> ForecastResult {
        // Forecast and build intervals on the working (possibly Box-Cox) scale.
        let work = match self.transform {
            Some(lambda) => box_cox(data, lambda),
            None => data.clone(),
        };
        let mean_w = self.forecast_core(&work, steps);

        let mut lower_80 = Array1::zeros(steps);
        let mut upper_80 = Array1::zeros(steps);
        let mut lower_95 = Array1::zeros(steps);
        let mut upper_95 = Array1::zeros(steps);

        // Proper ARIMA h-step forecast variance: sigma^2 * sum_{j<h} psi_j^2, where
        // the psi-weights are the MA(inf) representation of the full model with the
        // differencing operators folded into the AR side (so the intervals are on
        // the integrated working scale).
        let ar_poly = arima_poly::ar_polynomial(
            &self.order,
            &self.ar_coeffs.to_vec(),
            &self.sar_coeffs.to_vec(),
            true,
        );
        let ma_poly = arima_poly::ma_polynomial(
            &self.order,
            &self.ma_coeffs.to_vec(),
            &self.sma_coeffs.to_vec(),
        );
        let psi = arima_poly::psi_weights(&ar_poly, &ma_poly, steps.saturating_sub(1));

        let mut var_accum = 0.0;
        for h in 0..steps {
            var_accum += self.sigma2 * psi[h] * psi[h];
            let se = var_accum.sqrt();
            lower_80[h] = mean_w[h] - 1.282 * se;
            upper_80[h] = mean_w[h] + 1.282 * se;
            lower_95[h] = mean_w[h] - 1.960 * se;
            upper_95[h] = mean_w[h] + 1.960 * se;
        }

        // Back-transform mean and bounds. The Box-Cox inverse is monotone increasing,
        // so it preserves the lower <= mean <= upper ordering.
        match self.transform {
            Some(lambda) => ForecastResult {
                mean: inv_box_cox(&mean_w, lambda),
                lower_80: inv_box_cox(&lower_80, lambda),
                upper_80: inv_box_cox(&upper_80, lambda),
                lower_95: inv_box_cox(&lower_95, lambda),
                upper_95: inv_box_cox(&upper_95, lambda),
            },
            None => ForecastResult {
                mean: mean_w,
                lower_80,
                upper_80,
                lower_95,
                upper_95,
            },
        }
    }

    /// Convenience shortcut for non-seasonal ARIMA
    pub fn fit_arima(data: &Array1<f64>, p: usize, d: usize, q: usize) -> Result<Self> {
        Self::fit(data, SarimaOrder::arima(p, d, q))
    }

    /// Returns true if the fitted model is purely non-seasonal
    pub fn is_pure_arima(&self) -> bool {
        self.order.P == 0 && self.order.D == 0 && self.order.Q == 0
    }

    /// Recursively computes multiplicative SARIMA residuals:
    /// phi(B) Phi(B^m) (1-B)^d (1-B^m)^D Y_t = theta(B) Theta(B^m) epsilon_t
    fn compute_residuals(
        data: &Array1<f64>,
        ar: &Array1<f64>,
        ma: &Array1<f64>,
        sar: &Array1<f64>,
        sma: &Array1<f64>,
        order: &SarimaOrder,
    ) -> (Array1<f64>, f64) {
        let n = data.len();
        let mut residuals = Array1::zeros(n);
        let mut sse = 0.0;
        let m = order.m.max(1);

        for t in 0..n {
            let mut pred = 0.0;

            // Non-seasonal AR
            for i in 0..order.p {
                if t > i {
                    pred += ar[i] * data[t - i - 1];
                }
            }

            // Seasonal AR
            for i in 0..order.P {
                if t >= (i + 1) * m {
                    pred += sar[i] * data[t - (i + 1) * m];
                }
            }

            // Non-seasonal MA
            for j in 0..order.q {
                if t > j {
                    pred += ma[j] * residuals[t - j - 1];
                }
            }

            // Seasonal MA
            for j in 0..order.Q {
                if t >= (j + 1) * m {
                    pred += sma[j] * residuals[t - (j + 1) * m];
                }
            }

            let res = data[t] - pred;
            residuals[t] = res;

            let burn_in = order.p + order.q + (order.P + order.Q) * m;
            if t >= burn_in {
                sse += res.powi(2);
            }
        }

        (residuals, sse)
    }

    /// Number of estimated parameters: ARMA/seasonal coefficients, plus the mean
    /// (when included) plus the innovation variance.
    fn num_params(&self) -> f64 {
        let mean = usize::from(includes_mean(&self.order));
        (self.order.p + self.order.q + self.order.P + self.order.Q + mean + 1) as f64
    }

    pub fn aic(&self) -> f64 {
        let k = self.num_params();
        2.0 * k - 2.0 * self.log_likelihood
    }

    pub fn aicc(&self, n: usize) -> f64 {
        let k = self.num_params();
        let aic = self.aic();
        if (n as f64 - k - 1.0) <= 0.0 {
            return f64::INFINITY;
        }
        aic + (2.0 * k * (k + 1.0)) / (n as f64 - k - 1.0)
    }

    pub fn bic(&self, n: usize) -> f64 {
        let k = self.num_params();
        k * (n as f64).ln() - 2.0 * self.log_likelihood
    }

    /// Point forecast `steps` periods ahead. If the model was fitted with a Box-Cox
    /// transform, the forecast is automatically back-transformed to the original scale.
    pub fn forecast(&self, history: &Array1<f64>, steps: usize) -> Array1<f64> {
        let work = match self.transform {
            Some(lambda) => box_cox(history, lambda),
            None => history.clone(),
        };
        let fc = self.forecast_core(&work, steps);
        match self.transform {
            Some(lambda) => inv_box_cox(&fc, lambda),
            None => fc,
        }
    }

    /// Point forecast on the model's working scale (Box-Cox already applied, if any).
    fn forecast_core(&self, history: &Array1<f64>, steps: usize) -> Array1<f64> {
        // z = non-seasonally differenced history; w = z after seasonal differencing.
        let z_hist = difference(history, self.order.d);
        let seasonal = self.order.m > 1 && self.order.D > 0;
        let w_hist = if seasonal {
            seasonal_difference(&z_hist, self.order.m, self.order.D)
        } else {
            z_hist.clone()
        };

        // The ARMA part was fitted on the mean-centered w series.
        let centered = if self.intercept != 0.0 {
            &w_hist - self.intercept
        } else {
            w_hist.clone()
        };

        // Reconstruct in-sample residuals so MA/seasonal-MA terms drive the first
        // `q` (and `Q*m`) forecast steps; future innovations have expectation zero.
        let (residuals, _sse) = Self::compute_residuals(
            &centered,
            &self.ar_coeffs,
            &self.ma_coeffs,
            &self.sar_coeffs,
            &self.sma_coeffs,
            &self.order,
        );

        let m = self.order.m.max(1);
        let mut vals = centered.to_vec();
        let mut res = residuals.to_vec();
        let mut w_forecasts = Vec::with_capacity(steps);

        for _ in 0..steps {
            let t = vals.len();
            let mut f_val = 0.0;

            for i in 0..self.order.p {
                if t > i {
                    f_val += self.ar_coeffs[i] * vals[t - 1 - i];
                }
            }
            for i in 0..self.order.P {
                if t >= (i + 1) * m {
                    f_val += self.sar_coeffs[i] * vals[t - (i + 1) * m];
                }
            }
            for j in 0..self.order.q {
                if t > j {
                    f_val += self.ma_coeffs[j] * res[t - 1 - j];
                }
            }
            for j in 0..self.order.Q {
                if t >= (j + 1) * m {
                    f_val += self.sma_coeffs[j] * res[t - (j + 1) * m];
                }
            }

            vals.push(f_val);
            res.push(0.0); // expected value of a future innovation is zero
                           // Add the mean back to move from the centered scale to the w scale.
            w_forecasts.push(f_val + self.intercept);
        }

        // Undo seasonal differencing, then non-seasonal differencing.
        let w_forecasts = Array1::from(w_forecasts);
        let z_forecasts = if seasonal {
            seasonal_integrate_forecast(&w_forecasts, &z_hist, self.order.m, self.order.D)
        } else {
            w_forecasts
        };
        integrate_forecast(&z_forecasts, history, self.order.d)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InformationCriterion {
    Aic,
    Aicc,
    Bic,
}

/// How ARIMA coefficients are estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EstimationMethod {
    /// Conditional Sum of Squares: fast, robust, the default.
    #[default]
    Css,
    /// Exact Gaussian maximum likelihood via the Kalman filter: slower but more
    /// statistically efficient, especially on short series.
    Mle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoArimaOptions {
    pub max_p: usize,
    pub max_q: usize,
    pub max_P: usize,
    pub max_Q: usize,
    pub max_d: usize,
    pub max_D: usize,
    pub m: usize,
    pub criterion: InformationCriterion,
    pub stepwise: bool,
    pub estimation: EstimationMethod,
}

impl Default for AutoArimaOptions {
    fn default() -> Self {
        Self {
            max_p: 5,
            max_q: 5,
            max_P: 2,
            max_Q: 2,
            max_d: 2,
            max_D: 1,
            m: 1,
            criterion: InformationCriterion::Aicc,
            stepwise: true,
            estimation: EstimationMethod::Css,
        }
    }
}

/// Minimum batch size of candidate models required to justify Rayon thread-pool overhead.
const PARALLEL_BATCH_THRESHOLD: usize = 8;

/// Fits linear regression y = X * beta and returns beta coefficients.
fn fit_ols(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
    linalg::lstsq(x, y).map_err(|e| {
        ChronosError::ConvergenceFailure(format!(
            "Failed to solve OLS for exogenous variables: {}",
            e
        ))
    })
}

/// Splits a flat CSS parameter vector into (AR, MA, seasonal-AR, seasonal-MA) blocks.
/// Rejects empty series and series containing non-finite (NaN/inf) values.
fn validate_series(data: &Array1<f64>) -> Result<()> {
    if data.is_empty() {
        return Err(ChronosError::InvalidParameters(
            "Input series is empty.".into(),
        ));
    }
    if data.iter().any(|x| !x.is_finite()) {
        return Err(ChronosError::InvalidParameters(
            "Input series contains non-finite (NaN/inf) values.".into(),
        ));
    }
    Ok(())
}

/// Concatenates coefficient blocks into a single parameter vector `[ar, ma, sar, sma]`.
fn concat_params(
    ar: &Array1<f64>,
    ma: &Array1<f64>,
    sar: &Array1<f64>,
    sma: &Array1<f64>,
) -> Vec<f64> {
    ar.iter()
        .chain(ma.iter())
        .chain(sar.iter())
        .chain(sma.iter())
        .copied()
        .collect()
}

/// Asymptotic coefficient standard errors from a scalar objective (a negative
/// log-likelihood) evaluated at its optimum `x`: the covariance is the inverse of
/// the numerical Hessian; the standard errors are the square roots of its diagonal.
/// Returns `None` if the Hessian is singular or yields negative variances.
pub(crate) fn hessian_std_errors<F: Fn(&[f64]) -> f64>(f: F, x: &[f64]) -> Option<Array1<f64>> {
    let k = x.len();
    if k == 0 {
        return Some(Array1::zeros(0));
    }

    let h = 1e-4;
    let f0 = f(x);
    let mut hess = Array2::<f64>::zeros((k, k));
    let mut probe = x.to_vec();

    for i in 0..k {
        for j in i..k {
            let value = if i == j {
                probe[i] = x[i] + h;
                let fp = f(&probe);
                probe[i] = x[i] - h;
                let fm = f(&probe);
                probe[i] = x[i];
                (fp - 2.0 * f0 + fm) / (h * h)
            } else {
                probe[i] = x[i] + h;
                probe[j] = x[j] + h;
                let fpp = f(&probe);
                probe[j] = x[j] - h;
                let fpm = f(&probe);
                probe[i] = x[i] - h;
                let fmm = f(&probe);
                probe[j] = x[j] + h;
                let fmp = f(&probe);
                probe[i] = x[i];
                probe[j] = x[j];
                (fpp - fpm - fmp + fmm) / (4.0 * h * h)
            };
            hess[[i, j]] = value;
            hess[[j, i]] = value;
        }
    }

    let cov = linalg::inv(&hess).ok()?;
    let mut se = Array1::zeros(k);
    for i in 0..k {
        let var = cov[[i, i]];
        if !var.is_finite() || var < 0.0 {
            return None;
        }
        se[i] = var.sqrt();
    }
    Some(se)
}

pub(crate) fn split_params(
    params: &[f64],
    order: &SarimaOrder,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut idx = 0;
    let ar = Array1::from(params[idx..idx + order.p].to_vec());
    idx += order.p;
    let ma = Array1::from(params[idx..idx + order.q].to_vec());
    idx += order.q;
    let sar = Array1::from(params[idx..idx + order.P].to_vec());
    idx += order.P;
    let sma = Array1::from(params[idx..idx + order.Q].to_vec());
    (ar, ma, sar, sma)
}

/// argmin cost wrapping the Conditional Sum of Squares objective for a fixed order.
struct CssCost<'a> {
    data: &'a Array1<f64>,
    order: SarimaOrder,
}

impl<'a> CostFunction for CssCost<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let (ar, ma, sar, sma) = split_params(params, &self.order);
        let (_res, sse) =
            SarimaModel::compute_residuals(self.data, &ar, &ma, &sar, &sma, &self.order);
        if sse.is_finite() {
            Ok(sse)
        } else {
            Ok(f64::INFINITY)
        }
    }
}

/// Estimates SARIMA coefficients by Conditional Sum of Squares.
///
/// Uses a derivative-free Nelder-Mead search. The CSS objective is smooth for
/// pure-AR models but strongly nonlinear once moving-average terms are present
/// (residuals recurse through the coefficients), where gradient/line-search
/// methods stall at the origin; Nelder-Mead is robust across both cases.
///
/// Returns zero-length arrays for any block whose order is zero, and falls back
/// to zero coefficients if the optimizer fails (e.g. degenerate/constant input)
/// so that model fitting never panics.
fn estimate_css(
    data: &Array1<f64>,
    order: &SarimaOrder,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let k = order.p + order.q + order.P + order.Q;
    let fallback = || {
        (
            Array1::zeros(order.p),
            Array1::zeros(order.q),
            Array1::zeros(order.P),
            Array1::zeros(order.Q),
        )
    };

    // Pure white-noise / random-walk model: nothing to estimate.
    if k == 0 {
        return fallback();
    }

    // Initial simplex: the origin plus one perturbed vertex per parameter.
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(k + 1);
    simplex.push(vec![0.0; k]);
    for i in 0..k {
        let mut vertex = vec![0.0; k];
        vertex[i] = 0.3;
        simplex.push(vertex);
    }

    let cost = CssCost {
        data,
        order: *order,
    };

    let solver = match NelderMead::new(simplex).with_sd_tolerance(1e-9) {
        Ok(s) => s,
        Err(_) => return fallback(),
    };

    let outcome = Executor::new(cost, solver)
        .configure(|state| state.max_iters(1000))
        .run();

    match outcome {
        Ok(exec) => match exec.state.get_best_param() {
            Some(best) => {
                let (ar, ma, sar, sma) = split_params(best, order);
                stabilize(order, ar, ma, sar, sma)
            }
            None => fallback(),
        },
        Err(_) => fallback(),
    }
}

/// Keeps an estimated coefficient set stationary/stable. If the fitted AR part is
/// explosive, coefficients are shrunk geometrically toward zero until the
/// impulse-response is bounded; if that fails, they are zeroed. This is a safety
/// net against pathological CSS optima on difficult data — well-behaved fits pass
/// through unchanged.
fn stabilize(
    order: &SarimaOrder,
    mut ar: Array1<f64>,
    mut ma: Array1<f64>,
    mut sar: Array1<f64>,
    mut sma: Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let is_stable = |ar: &Array1<f64>, ma: &Array1<f64>, sar: &Array1<f64>, sma: &Array1<f64>| {
        let ar_poly = arima_poly::ar_polynomial(order, &ar.to_vec(), &sar.to_vec(), false);
        let ma_poly = arima_poly::ma_polynomial(order, &ma.to_vec(), &sma.to_vec());
        // Require both AR stationarity and MA invertibility.
        arima_poly::is_stable(&ar_poly, &ma_poly) && arima_poly::is_invertible(&ma_poly)
    };

    if is_stable(&ar, &ma, &sar, &sma) {
        return (ar, ma, sar, sma);
    }

    for _ in 0..40 {
        ar.mapv_inplace(|c| c * 0.9);
        ma.mapv_inplace(|c| c * 0.9);
        sar.mapv_inplace(|c| c * 0.9);
        sma.mapv_inplace(|c| c * 0.9);
        if is_stable(&ar, &ma, &sar, &sma) {
            return (ar, ma, sar, sma);
        }
    }

    // Give up: zero coefficients are trivially stable.
    (
        Array1::zeros(order.p),
        Array1::zeros(order.q),
        Array1::zeros(order.P),
        Array1::zeros(order.Q),
    )
}

/// Aggregate accuracy from an ARIMA rolling-origin cross-validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArimaCvReport {
    pub mae: f64,
    pub rmse: f64,
    pub mape: f64,
    pub n_windows: usize,
    pub horizon: usize,
}

/// Rolling-origin (expanding-window) cross-validation for a fixed ARIMA order.
///
/// Starting from an initial training window, the model is refit and forecast
/// `horizon` steps ahead, the origin advanced by `step`, and forecast errors
/// accumulated across all windows. Returns MAE / RMSE / MAPE over every
/// (window, step-ahead) pair.
pub fn arima_cross_validation(
    data: &Array1<f64>,
    order: SarimaOrder,
    method: EstimationMethod,
    initial: usize,
    horizon: usize,
    step: usize,
) -> Result<ArimaCvReport> {
    validate_series(data)?;
    let n = data.len();
    let step = step.max(1);

    if horizon == 0 {
        return Err(ChronosError::InvalidParameters(
            "Cross-validation horizon must be at least 1.".into(),
        ));
    }
    if initial + horizon > n {
        return Err(ChronosError::InsufficientData {
            required: initial + horizon,
            found: n,
        });
    }

    let mut abs_sum = 0.0;
    let mut sq_sum = 0.0;
    let mut pct_sum = 0.0;
    let mut pct_count = 0usize;
    let mut count = 0usize;
    let mut n_windows = 0usize;

    let mut train_end = initial;
    while train_end + horizon <= n {
        let train = data.slice(ndarray::s![..train_end]).to_owned();
        if let Ok(model) = SarimaModel::fit_with_method(&train, order, method) {
            let fc = model.forecast(&train, horizon);
            for h in 0..horizon {
                let actual = data[train_end + h];
                let err = fc[h] - actual;
                abs_sum += err.abs();
                sq_sum += err * err;
                if actual.abs() > 1e-8 {
                    pct_sum += (err / actual).abs();
                    pct_count += 1;
                }
                count += 1;
            }
            n_windows += 1;
        }
        train_end += step;
    }

    if count == 0 {
        return Err(ChronosError::ConvergenceFailure(
            "Cross-validation produced no valid forecasts.".into(),
        ));
    }

    let denom = count as f64;
    Ok(ArimaCvReport {
        mae: abs_sum / denom,
        rmse: (sq_sum / denom).sqrt(),
        mape: if pct_count > 0 {
            100.0 * pct_sum / pct_count as f64
        } else {
            f64::NAN
        },
        n_windows,
        horizon,
    })
}

/// Executes automatic model selection using statistical unit-root tests and stepwise IC optimization
pub fn auto_arima(data: &Array1<f64>, opts: AutoArimaOptions) -> Result<SarimaModel> {
    validate_series(data)?;
    let n = data.len();

    // 1. Determine d and D using stationarity tests
    let d = estimate_d(data, opts.max_d, 0.05);
    let D = estimate_D(data, opts.m, opts.max_D);

    // Seasonal orders are only meaningful with a real seasonal period. When
    // m <= 1 the seasonal P/D/Q terms operate at the same lags as the
    // non-seasonal ones (aliasing/degenerate), so disable them entirely.
    let seasonal = opts.m > 1;
    let max_P = if seasonal { opts.max_P } else { 0 };
    let max_Q = if seasonal { opts.max_Q } else { 0 };
    let seed_seasonal = usize::from(seasonal);

    let get_ic = |model: &SarimaModel| match opts.criterion {
        InformationCriterion::Aic => model.aic(),
        InformationCriterion::Aicc => model.aicc(n),
        InformationCriterion::Bic => model.bic(n),
    };

    let method = opts.estimation;

    // Helper closure to evaluate a batch of orders sequentially
    let eval_sequential = |orders: &[SarimaOrder]| {
        orders
            .iter()
            .filter_map(|&order| {
                SarimaModel::fit_with_method(data, order, method)
                    .ok()
                    .map(|model| (get_ic(&model), model))
            })
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    };

    // Helper closure to evaluate a batch of orders in parallel via Rayon
    let eval_parallel = |orders: &[SarimaOrder]| {
        orders
            .par_iter()
            .filter_map(|&order| {
                SarimaModel::fit_with_method(data, order, method)
                    .ok()
                    .map(|model| (get_ic(&model), model))
            })
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    };

    // Smart evaluator that applies Strategy 2
    let eval_orders = |orders: &[SarimaOrder]| {
        if orders.len() >= PARALLEL_BATCH_THRESHOLD {
            eval_parallel(orders)
        } else {
            eval_sequential(orders)
        }
    };

    // -------------------------------------------------------------
    // Full Grid Search Mode
    // -------------------------------------------------------------
    if !opts.stepwise {
        let mut orders = Vec::new();
        for p in 0..=opts.max_p {
            for q in 0..=opts.max_q {
                for P in 0..=max_P {
                    for Q in 0..=max_Q {
                        orders.push(SarimaOrder {
                            p,
                            d,
                            q,
                            P,
                            D,
                            Q,
                            m: opts.m,
                        });
                    }
                }
            }
        }

        return eval_orders(&orders)
            .map(|(_, model)| model)
            .ok_or_else(|| ChronosError::ConvergenceFailure("Failed grid model fitting".into()));
    }

    // -------------------------------------------------------------
    // Stepwise Heuristic Optimization Mode
    // -------------------------------------------------------------
    let seed_orders = vec![
        SarimaOrder {
            p: 2,
            d,
            q: 2,
            P: seed_seasonal,
            D,
            Q: seed_seasonal,
            m: opts.m,
        },
        SarimaOrder {
            p: 0,
            d,
            q: 0,
            P: 0,
            D,
            Q: 0,
            m: opts.m,
        },
        SarimaOrder {
            p: 1,
            d,
            q: 0,
            P: seed_seasonal,
            D,
            Q: 0,
            m: opts.m,
        },
        SarimaOrder {
            p: 0,
            d,
            q: 1,
            P: 0,
            D,
            Q: seed_seasonal,
            m: opts.m,
        },
    ];

    let (mut best_score, mut current_model) = eval_orders(&seed_orders).ok_or_else(|| {
        ChronosError::ConvergenceFailure("Failed fitting initial seed models".into())
    })?;

    let mut improved = true;
    while improved {
        improved = false;
        let curr_o = current_model.order;
        let mut candidates = Vec::new();

        if curr_o.p < opts.max_p {
            candidates.push(SarimaOrder {
                p: curr_o.p + 1,
                ..curr_o
            });
        }
        if curr_o.p > 0 {
            candidates.push(SarimaOrder {
                p: curr_o.p - 1,
                ..curr_o
            });
        }
        if curr_o.q < opts.max_q {
            candidates.push(SarimaOrder {
                q: curr_o.q + 1,
                ..curr_o
            });
        }
        if curr_o.q > 0 {
            candidates.push(SarimaOrder {
                q: curr_o.q - 1,
                ..curr_o
            });
        }

        // Seasonal perturbations (only when a seasonal period is configured).
        if curr_o.P < max_P {
            candidates.push(SarimaOrder {
                P: curr_o.P + 1,
                ..curr_o
            });
        }
        if curr_o.P > 0 {
            candidates.push(SarimaOrder {
                P: curr_o.P - 1,
                ..curr_o
            });
        }
        if curr_o.Q < max_Q {
            candidates.push(SarimaOrder {
                Q: curr_o.Q + 1,
                ..curr_o
            });
        }
        if curr_o.Q > 0 {
            candidates.push(SarimaOrder {
                Q: curr_o.Q - 1,
                ..curr_o
            });
        }

        if let Some((cand_score, cand_model)) = eval_orders(&candidates) {
            if cand_score < best_score {
                best_score = cand_score;
                current_model = cand_model;
                improved = true;
            }
        }
    }

    Ok(current_model)
}
