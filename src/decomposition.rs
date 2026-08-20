use crate::errors::{ChronosError, Result};
use crate::linalg;
use chrono::{Datelike, NaiveDate};
use ndarray::{s, Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalitySpec {
    pub name: String,
    pub period_days: f64,
    pub fourier_order: usize,
    pub prior_scale: f64, // Added field
}

impl Default for SeasonalitySpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            period_days: 365.25,
            fourier_order: 3,
            prior_scale: 10.0, // Matching Prophet default seasonality prior scale
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProphetPrediction {
    /// Combined forecast: trend + seasonality + holidays
    #[serde(with = "crate::utils::serde_array1")]
    pub yhat: Array1<f64>,
    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub yhat_lower: Option<Array1<f64>>,
    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub yhat_upper: Option<Array1<f64>>,

    /// Isolated g(t) trend signal
    #[serde(with = "crate::utils::serde_array1")]
    pub trend: Array1<f64>,
    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub trend_lower: Option<Array1<f64>>,
    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub trend_upper: Option<Array1<f64>>,

    /// Aggregated seasonal effects (sum of all Fourier orders)
    #[serde(with = "crate::utils::serde_array1")]
    pub seasonal: Array1<f64>,

    /// Breakdown of individual seasonality terms by name (e.g., "yearly", "weekly")
    pub seasonalities: HashMap<String, Array1<f64>>,
    pub seasonalities_lower: HashMap<String, Array1<f64>>,
    pub seasonalities_upper: HashMap<String, Array1<f64>>,

    /// Aggregated holiday adjustments
    #[serde(with = "crate::utils::serde_array1")]
    pub holidays: Array1<f64>,
}

impl ProphetPrediction {
    /// Returns the number of prediction points
    pub fn len(&self) -> usize {
        self.yhat.len()
    }

    /// Returns true if the prediction vector is empty
    pub fn is_empty(&self) -> bool {
        self.yhat.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SeasonalityMode {
    #[default]
    Additive,
    Multiplicative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrendType {
    #[default]
    Linear,
    Logistic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Holiday {
    pub name: String,
    pub dates: Vec<NaiveDate>,
    pub lower_window: i64,
    pub upper_window: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProphetDecomposition {
    pub trend_type: TrendType,
    pub seasonality_mode: SeasonalityMode, // Additive or Multiplicative

    /// Upper capacity bound per observation date C(t)
    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub cap: Option<Array1<f64>>,

    /// Optional lower capacity floor per observation date F(t) (default: 0.0)
    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub floor: Option<Array1<f64>>,

    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub capacities: Option<Array1<f64>>,
    pub n_changepoints: usize,
    pub changepoint_range: f64,
    pub changepoint_prior_scale: f64,
    pub holiday_prior_scale: f64,
    pub seasonalities: Vec<SeasonalitySpec>,
    pub holidays: Vec<Holiday>,

    // Training time normalization state
    pub t0_days: Option<f64>,
    pub total_days: Option<f64>,

    // Fitted Parameters
    pub changepoints: Option<Vec<f64>>,
    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub delta: Option<Array1<f64>>,
    pub k: Option<f64>,
    pub m: Option<f64>,
    #[serde(default, with = "crate::utils::serde_opt_array1")]
    pub beta: Option<Array1<f64>>,
}

impl ProphetDecomposition {
    pub fn new(n_changepoints: usize, changepoint_prior_scale: f64) -> Self {
        Self {
            trend_type: TrendType::Linear,
            capacities: None,
            cap: None,
            floor: None,
            n_changepoints,
            changepoint_range: 0.8,
            changepoint_prior_scale,
            holiday_prior_scale: 10.0,
            seasonality_mode: SeasonalityMode::Additive,
            seasonalities: Vec::new(),
            holidays: Vec::new(),
            t0_days: None,
            total_days: None,
            changepoints: None,
            delta: None,
            k: None,
            m: None,
            beta: None,
        }
    }

    /// Serializes the model state (including fitted coefficients) to a JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            ChronosError::InvalidParameters(format!("Failed to serialize model: {}", e))
        })
    }

    /// Deserializes a ProphetDecomposition model from a JSON string.
    pub fn from_json(json_str: &str) -> Result<Self> {
        serde_json::from_str(json_str).map_err(|e| {
            ChronosError::InvalidParameters(format!("Failed to deserialize model: {}", e))
        })
    }

    /// Saves the model state to a JSON file on disk.
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let json_data = self.to_json()?;
        std::fs::write(path, json_data).map_err(|e| {
            ChronosError::InvalidParameters(format!("Failed to write model file: {}", e))
        })
    }

    /// Loads a model state from a JSON file on disk.
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let json_data = std::fs::read_to_string(path).map_err(|e| {
            ChronosError::InvalidParameters(format!("Failed to read model file: {}", e))
        })?;
        Self::from_json(&json_data)
    }

    /// Computes the logistic trend g(t) handling rate delta adjustments and offset continuity
    #[allow(clippy::too_many_arguments)]
    fn evaluate_logistic_trend(
        &self,
        t_norm: &Array1<f64>,
        cps: &[f64],
        k: f64,
        m: f64,
        delta: &Array1<f64>,
        cap: &Array1<f64>,
        floor: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        let n = t_norm.len();
        let mut trend = Array1::<f64>::zeros(n);
        let default_floor = Array1::<f64>::zeros(n);
        let f = floor.unwrap_or(&default_floor);

        for i in 0..n {
            let t = t_norm[i];
            let mut rate = k;
            let mut gamma = 0.0;

            // Accumulate rate changes and offset adjustments at changepoints
            for (j, &s_j) in cps.iter().enumerate() {
                if t >= s_j {
                    let d = delta[j];
                    rate += d;
                    // Gamma ensures trend continuity across rate changes at s_j
                    gamma += (s_j - m - gamma / rate) * (1.0 - (rate - d) / rate);
                }
            }

            let c_i = cap[i];
            let f_i = f[i];
            let net_cap = (c_i - f_i).max(1e-5);

            // Logistic curve calculation: Floor + (Cap - Floor) / (1 + exp(-(k * (t - (m + gamma)))))
            let exp_term = (-rate * (t - (m + gamma))).exp();
            trend[i] = f_i + net_cap / (1.0 + exp_term);
        }

        trend
    }

    /// Adds a seasonal component with a default prior scale of 10.0
    pub fn add_seasonality(&mut self, name: &str, period_days: f64, fourier_order: usize) {
        self.add_seasonality_with_prior(name, period_days, fourier_order, 10.0);
    }

    /// Adds a seasonal component with a custom prior scale for independent regularization
    pub fn add_seasonality_with_prior(
        &mut self,
        name: &str,
        period_days: f64,
        fourier_order: usize,
        prior_scale: f64,
    ) {
        self.seasonalities.push(SeasonalitySpec {
            name: name.to_string(),
            period_days,
            fourier_order,
            prior_scale,
        });
    }

    pub fn add_holiday(&mut self, holiday: Holiday) {
        self.holidays.push(holiday);
    }

    /// Normalizes dates relative to the training period's t0 and scale horizon
    fn normalize_time(&self, dates: &[NaiveDate]) -> Result<Array1<f64>> {
        let t0 = self.t0_days.ok_or_else(|| {
            ChronosError::InvalidParameters(
                "Model must be fit before normalizing prediction dates".into(),
            )
        })?;
        let total = self.total_days.unwrap_or(1.0);

        Ok(Array1::from_vec(
            dates
                .iter()
                .map(|d| (d.num_days_from_ce() as f64 - t0) / total)
                .collect(),
        ))
    }

    fn build_changepoint_matrix(&self, t_norm: &Array1<f64>, changepoints: &[f64]) -> Array2<f64> {
        let n = t_norm.len();
        let s_len = changepoints.len();
        let mut a = Array2::<f64>::zeros((n, s_len));

        for i in 0..n {
            for j in 0..s_len {
                if t_norm[i] >= changepoints[j] {
                    a[[i, j]] = 1.0;
                }
            }
        }
        a
    }

    fn build_seasonal_and_holiday_matrix(&self, dates: &[NaiveDate]) -> Array2<f64> {
        let n = dates.len();
        let total_fourier_cols: usize =
            self.seasonalities.iter().map(|s| s.fourier_order * 2).sum();
        let num_holidays = self.holidays.len();
        let cols = total_fourier_cols + num_holidays;

        if cols == 0 {
            return Array2::zeros((n, 0));
        }

        let mut x = Array2::<f64>::zeros((n, cols));
        let t0 = self
            .t0_days
            .unwrap_or_else(|| dates[0].num_days_from_ce() as f64);

        let mut col_offset = 0;

        for spec in &self.seasonalities {
            for i in 0..n {
                let t_days = dates[i].num_days_from_ce() as f64 - t0;
                for j in 0..spec.fourier_order {
                    let n_term = (j + 1) as f64;
                    let arg = 2.0 * std::f64::consts::PI * n_term * t_days / spec.period_days;
                    x[[i, col_offset + 2 * j]] = arg.sin();
                    x[[i, col_offset + 2 * j + 1]] = arg.cos();
                }
            }
            col_offset += spec.fourier_order * 2;
        }

        for (h_idx, holiday) in self.holidays.iter().enumerate() {
            for i in 0..n {
                let mut active = 0.0;
                for h_date in &holiday.dates {
                    let diff = (dates[i] - *h_date).num_days();
                    if diff >= holiday.lower_window && diff <= holiday.upper_window {
                        active = 1.0;
                        break;
                    }
                }
                x[[i, col_offset + h_idx]] = active;
            }
        }

        x
    }

    /// Predicts yhat and computes percentile-based uncertainty intervals via parallel Monte Carlo simulation
    pub fn predict_with_intervals(
        &self,
        dates: &[NaiveDate],
        interval_width: f64,
        n_samples: usize,
    ) -> Result<ProphetPrediction> {
        let n_obs = dates.len();
        let base_pred = self.predict(dates)?;

        if n_samples == 0 {
            return Ok(base_pred);
        }

        // 1. Retrieve delta vector
        let delta = self.delta.as_ref().ok_or_else(|| {
            ChronosError::InvalidParameters(
                "Model must be fitted before predicting intervals".into(),
            )
        })?;

        let abs_mean_delta = delta.mapv(|d| d.abs()).mean().unwrap_or(0.01);
        let n_historical = delta.len();
        let changepoint_prob = (self.n_changepoints as f64) / (n_historical as f64).max(1.0);
        let b = abs_mean_delta.max(1e-5);
        let seasonality_mode = self.seasonality_mode;

        // 2. Parallel Monte Carlo Sample Draws using Rayon
        let samples: Vec<(Vec<f64>, Vec<f64>)> = (0..n_samples)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let noise_dist = Normal::new(0.0, 0.01).unwrap();

                let mut trend_draws = vec![0.0; n_obs];
                let mut yhat_draws = vec![0.0; n_obs];
                let mut sampled_slope_change = 0.0;

                for t in 0..n_obs {
                    if rng.gen_bool(changepoint_prob.min(1.0)) {
                        // Inverse transform sampling for Laplace(0, b)
                        let u: f64 = rng.gen_range(-0.5..0.5);
                        let laplace_sample = -b * u.signum() * (1.0 - 2.0 * u.abs()).ln();
                        sampled_slope_change += laplace_sample;
                    }

                    let trend_draw = base_pred.trend[t] + sampled_slope_change * (t as f64);
                    trend_draws[t] = trend_draw;

                    let noise = noise_dist.sample(&mut rng);
                    let yhat_draw = match seasonality_mode {
                        SeasonalityMode::Additive => {
                            trend_draw + base_pred.seasonal[t] + base_pred.holidays[t] + noise
                        }
                        SeasonalityMode::Multiplicative => {
                            trend_draw * (1.0 + base_pred.seasonal[t] + base_pred.holidays[t])
                                + noise
                        }
                    };

                    yhat_draws[t] = yhat_draw;
                }

                (trend_draws, yhat_draws)
            })
            .collect();

        // 3. Compute percentile bounds across parallel draws
        let alpha = (1.0 - interval_width) / 2.0;
        let lower_idx = ((alpha * n_samples as f64).floor() as usize).min(n_samples - 1);
        let upper_idx = (((1.0 - alpha) * n_samples as f64).ceil() as usize).min(n_samples - 1);

        let mut trend_lower = vec![0.0; n_obs];
        let mut trend_upper = vec![0.0; n_obs];
        let mut yhat_lower = vec![0.0; n_obs];
        let mut yhat_upper = vec![0.0; n_obs];

        // Parallel percentile extraction per timestamp t
        let bounds: Vec<(f64, f64, f64, f64)> = (0..n_obs)
            .into_par_iter()
            .map(|t| {
                let mut t_col: Vec<f64> = samples.iter().map(|s| s.0[t]).collect();
                t_col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let mut y_col: Vec<f64> = samples.iter().map(|s| s.1[t]).collect();
                y_col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                (
                    t_col[lower_idx],
                    t_col[upper_idx],
                    y_col[lower_idx],
                    y_col[upper_idx],
                )
            })
            .collect();

        for (t, (tl, tu, yl, yu)) in bounds.into_iter().enumerate() {
            trend_lower[t] = tl;
            trend_upper[t] = tu;
            yhat_lower[t] = yl;
            yhat_upper[t] = yu;
        }

        // 4. Return ProphetPrediction with interval options
        let mut res = base_pred;
        res.trend_lower = Some(Array1::from_vec(trend_lower));
        res.trend_upper = Some(Array1::from_vec(trend_upper));
        res.yhat_lower = Some(Array1::from_vec(yhat_lower));
        res.yhat_upper = Some(Array1::from_vec(yhat_upper));

        Ok(res)
    }

    pub fn fit(
        &mut self,
        dates: &[NaiveDate],
        y: &Array1<f64>,
        cap: Option<&Array1<f64>>,
        floor: Option<&Array1<f64>>,
    ) -> Result<()> {
        let n = dates.len();
        if n < 2 {
            return Err(ChronosError::InsufficientData {
                required: 2,
                found: n,
            });
        }

        self.cap = cap.cloned();
        self.floor = floor.cloned();

        // Store training scale parameters
        let t0 = dates[0].num_days_from_ce() as f64;
        let t_end = dates.last().unwrap().num_days_from_ce() as f64;
        let total = (t_end - t0).max(1.0);

        self.t0_days = Some(t0);
        self.total_days = Some(total);

        let t_norm = self.normalize_time(dates)?;

        // Handle Logistic logit transformation or Linear pass-through
        let y_target = match self.trend_type {
            TrendType::Linear => y.clone(),
            TrendType::Logistic => {
                let cap_arr = self.cap.as_ref().ok_or_else(|| {
                    ChronosError::InvalidParameters("Logistic growth requires cap values".into())
                })?;
                let default_floor = Array1::<f64>::zeros(n);
                let f = self.floor.as_ref().unwrap_or(&default_floor);

                let mut y_transformed = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let net_cap = (cap_arr[i] - f[i]).max(1e-5);
                    let p = ((y[i] - f[i]) / net_cap).clamp(1e-4, 1.0 - 1e-4);
                    y_transformed[i] = (p / (1.0 - p)).ln();
                }
                y_transformed
            }
        };

        let max_cp_t = self.changepoint_range;
        let mut cps = Vec::with_capacity(self.n_changepoints);
        for i in 1..=self.n_changepoints {
            cps.push((i as f64 / (self.n_changepoints + 1) as f64) * max_cp_t);
        }

        let a_cp = self.build_changepoint_matrix(&t_norm, &cps);
        let x_seasonal = self.build_seasonal_and_holiday_matrix(dates);

        let n_seasonal_cols = x_seasonal.ncols();
        let total_cols = 2 + self.n_changepoints + n_seasonal_cols;

        let mut x = Array2::<f64>::zeros((n, total_cols));

        let is_multiplicative = self.seasonality_mode == SeasonalityMode::Multiplicative;

        // Baseline trend endpoints from raw y
        let y_start = y[0];
        let y_end = y[n - 1];

        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = t_norm[i];

            for j in 0..self.n_changepoints {
                if a_cp[[i, j]] > 0.0 {
                    x[[i, 2 + j]] = (t_norm[i] - cps[j]) * a_cp[[i, j]];
                }
            }

            // Scale seasonal columns by the actual empirical magnitude of y(t)
            let trend_scale = if is_multiplicative {
                (y_start + (y_end - y_start) * t_norm[i]).max(1e-3)
            } else {
                1.0
            };

            // Scale seasonal/holiday features
            for j in 0..n_seasonal_cols {
                x[[i, 2 + self.n_changepoints + j]] = x_seasonal[[i, j]] * trend_scale;
            }
        }

        let mut xtx = x.t().dot(&x);

        // 1. Changepoint Regularization
        let lambda_cp = 1.0 / (self.changepoint_prior_scale.powi(2)).max(1e-5);
        for j in 0..self.n_changepoints {
            xtx[[2 + j, 2 + j]] += lambda_cp;
        }

        // 2. Per-Seasonality Fourier Regularization
        let mut col_offset = 2 + self.n_changepoints;

        for spec in &self.seasonalities {
            let fourier_cols = spec.fourier_order * 2;
            let lambda_spec = 1.0 / (spec.prior_scale.powi(2)).max(1e-5);

            for col in col_offset..(col_offset + fourier_cols) {
                xtx[[col, col]] += lambda_spec;
            }

            col_offset += fourier_cols;
        }

        // 3. Holiday Regularization
        let lambda_holiday = 1.0 / (self.holiday_prior_scale.powi(2)).max(1e-5);
        for h_idx in 0..self.holidays.len() {
            let col = col_offset + h_idx;
            xtx[[col, col]] += lambda_holiday;
        }

        // Solve the (regularized, positive-definite) Ridge normal equations.
        let xty = x.t().dot(&y_target);
        let coeffs = linalg::solve(&xtx, &xty).map_err(ChronosError::LinalgError)?;

        self.m = Some(coeffs[0]);
        self.k = Some(coeffs[1]);
        self.delta = Some(coeffs.slice(s![2..2 + self.n_changepoints]).to_owned());

        if n_seasonal_cols > 0 {
            self.beta = Some(coeffs.slice(s![2 + self.n_changepoints..]).to_owned());
        }

        self.changepoints = Some(cps);

        Ok(())
    }

    pub fn predict(&self, dates: &[NaiveDate]) -> Result<ProphetPrediction> {
        let cps = self.changepoints.as_ref().ok_or_else(|| {
            ChronosError::InvalidParameters("Model must be fit before calling predict".into())
        })?;
        let delta = self.delta.as_ref().unwrap();
        let k = self.k.unwrap();
        let m = self.m.unwrap();

        let n = dates.len();
        let t_norm = self.normalize_time(dates)?;
        let a_cp = self.build_changepoint_matrix(&t_norm, cps);

        // 1. Evaluate Trend Signal g(t)
        let trend = match self.trend_type {
            TrendType::Linear => {
                let mut tr = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let mut rate = k;
                    let mut offset = m;
                    for j in 0..self.n_changepoints {
                        if a_cp[[i, j]] > 0.0 {
                            rate += delta[j];
                            offset -= cps[j] * delta[j];
                        }
                    }
                    tr[i] = rate * t_norm[i] + offset;
                }
                tr
            }
            TrendType::Logistic => {
                let cap = self.cap.as_ref().ok_or_else(|| {
                    ChronosError::InvalidParameters("Logistic growth requires cap values".into())
                })?;
                self.evaluate_logistic_trend(&t_norm, cps, k, m, delta, cap, self.floor.as_ref())
            }
        };

        // 2. Evaluate Individual Seasonalities and Holidays
        let mut seasonal_total = Array1::<f64>::zeros(n);
        let mut holiday_total = Array1::<f64>::zeros(n);
        let mut seasonalities_map = HashMap::new();

        if let Some(ref beta) = self.beta {
            let t0 = self
                .t0_days
                .unwrap_or_else(|| dates[0].num_days_from_ce() as f64);
            let mut col_offset = 0;

            // Extract each registered seasonality independently
            for spec in &self.seasonalities {
                let mut spec_component = Array1::<f64>::zeros(n);
                let fourier_cols = spec.fourier_order * 2;

                for i in 0..n {
                    let t_days = dates[i].num_days_from_ce() as f64 - t0;
                    let mut val = 0.0;
                    for j in 0..spec.fourier_order {
                        let n_term = (j + 1) as f64;
                        let arg = 2.0 * std::f64::consts::PI * n_term * t_days / spec.period_days;

                        let sin_coef = beta[col_offset + 2 * j];
                        let cos_coef = beta[col_offset + 2 * j + 1];

                        val += sin_coef * arg.sin() + cos_coef * arg.cos();
                    }
                    spec_component[i] = val;
                }

                seasonal_total = &seasonal_total + &spec_component;
                seasonalities_map.insert(spec.name.clone(), spec_component);
                col_offset += fourier_cols;
            }

            // Extract holiday features
            for (h_idx, holiday) in self.holidays.iter().enumerate() {
                let coef = beta[col_offset + h_idx];
                for i in 0..n {
                    for h_date in &holiday.dates {
                        let diff = (dates[i] - *h_date).num_days();
                        if diff >= holiday.lower_window && diff <= holiday.upper_window {
                            holiday_total[i] += coef;
                            break;
                        }
                    }
                }
            }
        }

        // 3. Compute combined yhat forecast based on seasonality mode
        let yhat = match self.seasonality_mode {
            SeasonalityMode::Additive => &trend + &seasonal_total + &holiday_total,
            SeasonalityMode::Multiplicative => {
                // seasonal_total is the fitted fractional ratio S(t)
                &trend * (1.0 + &seasonal_total + &holiday_total)
            }
        };

        Ok(ProphetPrediction {
            yhat,
            yhat_lower: None,
            yhat_upper: None,
            trend,
            trend_lower: None,
            trend_upper: None,
            seasonal: seasonal_total,
            seasonalities: seasonalities_map,
            seasonalities_lower: HashMap::new(),
            seasonalities_upper: HashMap::new(),
            holidays: holiday_total,
        })
    }
}
