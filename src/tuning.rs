use chrono::NaiveDate;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::errors::{ChronosError, Result};
use crate::decomposition::{ProphetDecomposition, SeasonalityMode};

/// Hyperparameter grid target parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterGrid {
    pub changepoint_prior_scales: Vec<f64>,
    pub seasonality_prior_scales: Vec<f64>,
    pub holidays_prior_scales: Vec<f64>,
    pub seasonality_modes: Vec<SeasonalityMode>,
}

impl Default for HyperparameterGrid {
    fn default() -> Self {
        Self {
            changepoint_prior_scales: vec![0.001, 0.01, 0.05, 0.1, 0.5],
            seasonality_prior_scales: vec![0.01, 0.1, 1.0, 10.0],
            holidays_prior_scales: vec![0.01, 0.1, 1.0, 10.0],
            seasonality_modes: vec![SeasonalityMode::Additive, SeasonalityMode::Multiplicative],
        }
    }
}

/// A specific candidate set of hyperparameter configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterCandidate {
    pub changepoint_prior_scale: f64,
    pub seasonality_prior_scale: f64,
    pub holidays_prior_scale: f64,
    pub seasonality_mode: SeasonalityMode,
}

/// Cross-validation metric choice
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationMetric {
    MAE,
    RMSE,
    MAPE,
}

/// Result returned from auto-tuning evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneResult {
    pub best_params: HyperparameterCandidate,
    pub best_score: f64,
    pub metric: OptimizationMetric,
    pub all_evaluated_scores: Vec<(HyperparameterCandidate, f64)>,
}

/// AutoTuner handles parameter search over rolling time-series cutoffs
pub struct AutoTuner {
    grid: HyperparameterGrid,
    metric: OptimizationMetric,
    initial_window_pct: f64,
    horizon_days: usize,
    step_days: usize,
}

impl AutoTuner {
    pub fn new(grid: HyperparameterGrid) -> Self {
        Self {
            grid,
            metric: OptimizationMetric::MAE,
            initial_window_pct: 0.6,
            horizon_days: 30,
            step_days: 15,
        }
    }

    pub fn with_metric(mut self, metric: OptimizationMetric) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_validation_split(
        mut self,
        initial_window_pct: f64,
        horizon_days: usize,
        step_days: usize,
    ) -> Self {
        self.initial_window_pct = initial_window_pct.clamp(0.1, 0.9);
        self.horizon_days = horizon_days.max(1);
        self.step_days = step_days.max(1);
        self
    }

    /// Run grid search optimization across historical data cuts
    pub fn fit_and_tune(
        &self,
        base_model: &ProphetDecomposition,
        dates: &[NaiveDate],
        values: &Array1<f64>,
    ) -> Result<AutoTuneResult> {
        let n = dates.len();
        if n < 30 || values.len() != n {
            return Err(ChronosError::InvalidParameters(
                "Auto-tuning requires at least 30 data points".into(),
            ));
        }

        let candidates = self.generate_candidates();
        let cutoffs = self.generate_cross_validation_cutoffs(dates)?;

        if cutoffs.is_empty() {
            return Err(ChronosError::InvalidParameters(
                "Dataset duration is too short for the configured horizon and initial window".into(),
            ));
        }

        let mut evaluated_scores = Vec::new();
        let mut best_score = f64::INFINITY;
        let mut best_candidate = candidates[0].clone();

        for candidate in candidates {
            let mut fold_scores = Vec::new();

            for (train_end_idx, val_end_idx) in &cutoffs {
                let train_dates = &dates[..*train_end_idx];
                let train_vals = values.slice(ndarray::s![..*train_end_idx]).to_owned();

                let val_dates = &dates[*train_end_idx..*val_end_idx];
                let val_vals = values.slice(ndarray::s![*train_end_idx..*val_end_idx]).to_owned();

                // Clone base template and apply hyperparameter candidate
                let mut tuned_model = base_model.clone();
                tuned_model.changepoint_prior_scale = candidate.changepoint_prior_scale;
                tuned_model.holiday_prior_scale = candidate.holidays_prior_scale;
                tuned_model.seasonality_mode = candidate.seasonality_mode;

                // Update seasonality prior scales across registered seasonality specs
                for spec in &mut tuned_model.seasonalities {
                    spec.prior_scale = candidate.seasonality_prior_scale;
                }

                // Fit fold with 4 parameters: dates, y, cap, floor
                if tuned_model.fit(train_dates, &train_vals, None, None).is_err() {
                    continue;
                }

                // Predict validation step
                if let Ok(pred) = tuned_model.predict(val_dates) {
                    let score = evaluate_metric(&pred.yhat, &val_vals, self.metric);
                    if !score.is_nan() && !score.is_infinite() {
                        fold_scores.push(score);
                    }
                }
            }

            if fold_scores.is_empty() {
                continue;
            }

            // Average score across cross-validation windows
            let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            evaluated_scores.push((candidate.clone(), mean_score));

            if mean_score < best_score {
                best_score = mean_score;
                best_candidate = candidate;
            }
        }

        if evaluated_scores.is_empty() {
            return Err(ChronosError::InvalidParameters(
                "All candidate evaluations failed during tuning".into(),
            ));
        }

        Ok(AutoTuneResult {
            best_params: best_candidate,
            best_score,
            metric: self.metric,
            all_evaluated_scores: evaluated_scores,
        })
    }

    fn generate_candidates(&self) -> Vec<HyperparameterCandidate> {
        let mut candidates = Vec::new();
        for &cp in &self.grid.changepoint_prior_scales {
            for &s in &self.grid.seasonality_prior_scales {
                for &h in &self.grid.holidays_prior_scales {
                    for &m in &self.grid.seasonality_modes {
                        candidates.push(HyperparameterCandidate {
                            changepoint_prior_scale: cp,
                            seasonality_prior_scale: s,
                            holidays_prior_scale: h,
                            seasonality_mode: m,
                        });
                    }
                }
            }
        }
        candidates
    }

    fn generate_cross_validation_cutoffs(
        &self,
        dates: &[NaiveDate],
    ) -> Result<Vec<(usize, usize)>> {
        let n = dates.len();
        let initial_train_size = ((n as f64) * self.initial_window_pct).floor() as usize;

        let mut cutoffs = Vec::new();
        let mut current_train_end = initial_train_size;

        while current_train_end + self.horizon_days <= n {
            let val_end = current_train_end + self.horizon_days;
            cutoffs.push((current_train_end, val_end));
            current_train_end += self.step_days;
        }

        Ok(cutoffs)
    }
}

/// Helper function to evaluate accuracy error metrics
fn evaluate_metric(pred: &Array1<f64>, actual: &Array1<f64>, metric: OptimizationMetric) -> f64 {
    let n = pred.len();
    if n == 0 || actual.len() != n {
        return f64::NAN;
    }

    match metric {
        OptimizationMetric::MAE => {
            (pred - actual).mapv(|x| x.abs()).sum() / (n as f64)
        }
        OptimizationMetric::RMSE => {
            ((pred - actual).mapv(|x| x * x).sum() / (n as f64)).sqrt()
        }
        OptimizationMetric::MAPE => {
            let sum_err: f64 = pred
                .iter()
                .zip(actual.iter())
                .map(|(&p, &a)| {
                    if a.abs() < 1e-8 {
                        0.0
                    } else {
                        ((p - a) / a).abs()
                    }
                })
                .sum();
            (sum_err / (n as f64)) * 100.0
        }
    }
}