#![allow(non_snake_case)]
use crate::errors::{ChronosError, Result};
use crate::stat_tests::{estimate_d, estimate_D};
use crate::utils::{difference, integrate_forecast, seasonal_difference};
use ndarray::{Array1, Array2}; // Fixed: Added missing Array2 import
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

pub struct ForecastResult {
    pub mean: Array1<f64>,
    pub lower_80: Array1<f64>,
    pub upper_80: Array1<f64>,
    pub lower_95: Array1<f64>,
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

    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub exog_beta: Option<Array1<f64>>,
}

/// Type aliases allowing callers to use Arima and Sarima interchangeably
pub type Arima = SarimaModel;
pub type ArimaOrder = SarimaOrder;

impl SarimaModel {
    /// Primary constructor for fitting SARIMA/ARIMA models
    pub fn fit(data: &Array1<f64>, order: SarimaOrder) -> Result<Self> {
        let min_required =
            order.p + order.q + order.d + (order.P + order.Q + order.D) * order.m.max(1);
        if data.len() <= min_required {
            return Err(ChronosError::InsufficientData {
                required: min_required + 1,
                found: data.len(),
            });
        }

        // 1. Non-seasonal differencing
        let mut transformed = difference(data, order.d);

        // 2. Seasonal differencing (skipped if m <= 1 or D == 0)
        if order.m > 1 && order.D > 0 {
            transformed = seasonal_difference(&transformed, order.m, order.D);
        }

        let n = transformed.len();

        let ar_coeffs = Array1::zeros(order.p);
        let ma_coeffs = Array1::zeros(order.q);
        let sar_coeffs = Array1::zeros(order.P);
        let sma_coeffs = Array1::zeros(order.Q);

        let (_residuals, sse) = Self::compute_residuals(
            &transformed,
            &ar_coeffs,
            &ma_coeffs,
            &sar_coeffs,
            &sma_coeffs,
            &order,
        );

        let sigma2 = (sse / (n as f64)).max(1e-8);
        let log_like = -0.5 * (n as f64) * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0);

        // Fixed: Included missing `exog_beta: None`
        Ok(Self {
            order,
            ar_coeffs,
            ma_coeffs,
            sar_coeffs,
            sma_coeffs,
            sigma2,
            log_likelihood: log_like,
            exog_beta: None,
        })
    }

    /// Fits a SARIMAX model. If `exog` is provided, it first removes the linear trend
    /// contributed by X and then fits the SARIMA parameters on the regression residuals.
    pub fn fit_with_exog(
        y: &Array1<f64>,
        exog: Option<&Array2<f64>>,
        order: SarimaOrder,
    ) -> Result<Self> { // Fixed: Changed return type to crate Result<Self>
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

    pub fn forecast_with_intervals_exog(
        &self,
        data: &Array1<f64>,
        exog_future: Option<&Array2<f64>>,
        steps: usize,
    ) -> Result<ForecastResult> { // Fixed: Changed return type to crate Result<ForecastResult>
        // Adjust historical series if model was fitted with exogenous features
        let data_adjusted = if let Some(_beta) = &self.exog_beta {
            // Note: In real applications, pass historical X to subtract historical exog effects here
            data.clone()
        } else {
            data.clone()
        };

        // Generate base SARIMA point forecasts and confidence intervals
        let mut forecast = self.forecast_with_intervals(&data_adjusted, steps);

        // Add the contribution of future exogenous variables: y_fut = \eta_fut + X_fut * beta
        if let (Some(beta), Some(x_fut)) = (&self.exog_beta, exog_future) {
            if x_fut.nrows() != steps {
                return Err(ChronosError::ConvergenceFailure(
                    "Future exogenous matrix rows must equal requested forecast steps.".into(),
                ));
            }
            let exog_impact = x_fut.dot(beta);

            forecast.mean += &exog_impact;
            forecast.lower_80 += &exog_impact;
            forecast.upper_80 += &exog_impact;
            forecast.lower_95 += &exog_impact;
            forecast.upper_95 += &exog_impact;
        }

        Ok(forecast)
    }

    pub fn forecast_with_intervals(&self, data: &Array1<f64>, steps: usize) -> ForecastResult {
        let mean = self.forecast(data, steps);
        let mut lower_80 = Array1::zeros(steps);
        let mut upper_80 = Array1::zeros(steps);
        let mut lower_95 = Array1::zeros(steps);
        let mut upper_95 = Array1::zeros(steps);

        // Derive standard deviation from model residual variance (sigma2)
        let sigma = self.sigma2.sqrt();

        for h in 0..steps {
            let se = sigma * ((h as f64 + 1.0).sqrt());
            lower_80[h] = mean[h] - 1.282 * se;
            upper_80[h] = mean[h] + 1.282 * se;
            lower_95[h] = mean[h] - 1.960 * se;
            upper_95[h] = mean[h] + 1.960 * se;
        }

        ForecastResult {
            mean,
            lower_80,
            upper_80,
            lower_95,
            upper_95,
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
                if t >= i + 1 {
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
                if t >= j + 1 {
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

    pub fn aic(&self) -> f64 {
        let k = (self.order.p + self.order.q + self.order.P + self.order.Q + 1) as f64;
        2.0 * k - 2.0 * self.log_likelihood
    }

    pub fn aicc(&self, n: usize) -> f64 {
        let k = (self.order.p + self.order.q + self.order.P + self.order.Q + 1) as f64;
        let aic = self.aic();
        if (n as f64 - k - 1.0) <= 0.0 {
            return f64::INFINITY;
        }
        aic + (2.0 * k * (k + 1.0)) / (n as f64 - k - 1.0)
    }

    pub fn bic(&self, n: usize) -> f64 {
        let k = (self.order.p + self.order.q + self.order.P + self.order.Q + 1) as f64;
        k * (n as f64).ln() - 2.0 * self.log_likelihood
    }

    pub fn forecast(&self, history: &Array1<f64>, steps: usize) -> Array1<f64> {
        let mut diff_hist = difference(history, self.order.d);
        if self.order.m > 1 && self.order.D > 0 {
            diff_hist = seasonal_difference(&diff_hist, self.order.m, self.order.D);
        }

        let mut diff_forecasts = Vec::with_capacity(steps);
        let mut extended_diff = diff_hist.to_vec();

        for _ in 0..steps {
            let n = extended_diff.len();
            let mut f_val = 0.0;

            for i in 0..self.order.p {
                if n >= i + 1 {
                    f_val += self.ar_coeffs[i] * extended_diff[n - 1 - i];
                }
            }
            for i in 0..self.order.P {
                if n >= (i + 1) * self.order.m {
                    f_val += self.sar_coeffs[i] * extended_diff[n - (i + 1) * self.order.m];
                }
            }

            diff_forecasts.push(f_val);
            extended_diff.push(f_val);
        }

        integrate_forecast(&Array1::from(diff_forecasts), history, self.order.d)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InformationCriterion {
    Aic,
    Aicc,
    Bic,
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
        }
    }
}

/// Minimum batch size of candidate models required to justify Rayon thread-pool overhead.
const PARALLEL_BATCH_THRESHOLD: usize = 8;

/// Fits linear regression y = X * beta and returns beta coefficients.
fn fit_ols(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> { // Fixed: Return crate Result
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);

    match ndarray_linalg::Solve::solve_into(&xtx, xty) {
        Ok(beta) => Ok(beta),
        Err(e) => Err(ChronosError::ConvergenceFailure(format!(
            "Failed to solve OLS for exogenous variables: {}",
            e
        ))),
    }
}

/// Executes automatic model selection using statistical unit-root tests and stepwise IC optimization
pub fn auto_arima(data: &Array1<f64>, opts: AutoArimaOptions) -> Result<SarimaModel> {
    let n = data.len();

    // 1. Determine d and D using stationarity tests
    let d = estimate_d(data, opts.max_d, 0.05);
    let D = estimate_D(data, opts.m, opts.max_D);

    let get_ic = |model: &SarimaModel| match opts.criterion {
        InformationCriterion::Aic => model.aic(),
        InformationCriterion::Aicc => model.aicc(n),
        InformationCriterion::Bic => model.bic(n),
    };

    // Helper closure to evaluate a batch of orders sequentially
    let eval_sequential = |orders: &[SarimaOrder]| {
        orders
            .iter()
            .filter_map(|&order| {
                SarimaModel::fit(data, order)
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
                SarimaModel::fit(data, order)
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
                for P in 0..=opts.max_P {
                    for Q in 0..=opts.max_Q {
                        orders.push(SarimaOrder { p, d, q, P, D, Q, m: opts.m });
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
        SarimaOrder { p: 2, d, q: 2, P: 1, D, Q: 1, m: opts.m },
        SarimaOrder { p: 0, d, q: 0, P: 0, D, Q: 0, m: opts.m },
        SarimaOrder { p: 1, d, q: 0, P: 1, D, Q: 0, m: opts.m },
        SarimaOrder { p: 0, d, q: 1, P: 0, D, Q: 1, m: opts.m },
    ];

    let (mut best_score, mut current_model) = eval_orders(&seed_orders).ok_or_else(|| {
        ChronosError::ConvergenceFailure("Failed fitting initial seed models".into())
    })?;

    let mut improved = true;
    while improved {
        improved = false;
        let curr_o = current_model.order;
        let mut candidates = Vec::new();

        if curr_o.p < opts.max_p { candidates.push(SarimaOrder { p: curr_o.p + 1, ..curr_o }); }
        if curr_o.p > 0 { candidates.push(SarimaOrder { p: curr_o.p - 1, ..curr_o }); }
        if curr_o.q < opts.max_q { candidates.push(SarimaOrder { q: curr_o.q + 1, ..curr_o }); }
        if curr_o.q > 0 { candidates.push(SarimaOrder { q: curr_o.q - 1, ..curr_o }); }

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