use crate::decomposition::ProphetDecomposition;
use crate::errors::{ChronosError, Result};
use chrono::NaiveDate;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub struct LjungBoxResult {
    pub q_stat: f64,
    pub p_value: f64,
    pub lags: usize,
}

pub struct JarqueBeraResult {
    pub jb_stat: f64,
    pub p_value: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

pub struct DiagnosticsResult {
    pub acf: Array1<f64>,
    pub pacf: Array1<f64>,
    pub ljung_box: LjungBoxResult,
    pub jarque_bera: JarqueBeraResult,
}

pub struct ResidualDiagnostics;

impl ResidualDiagnostics {
    /// Sample Autocorrelation Function (ACF) up to `max_lag`
    pub fn acf(residuals: &Array1<f64>, max_lag: usize) -> Result<Array1<f64>> {
        let n = residuals.len();
        if n <= max_lag {
            return Err(ChronosError::InsufficientData {
                required: max_lag + 1,
                found: n,
            });
        }

        let mean = residuals.mean().unwrap_or(0.0);
        let var = residuals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();

        if var == 0.0 {
            return Err(ChronosError::InvalidParameters(
                "Zero variance in residual series".into(),
            ));
        }

        let mut acf_vals = Vec::with_capacity(max_lag + 1);
        acf_vals.push(1.0); // Lag 0 is always 1.0

        for lag in 1..=max_lag {
            let mut cov = 0.0;
            for t in lag..n {
                cov += (residuals[t] - mean) * (residuals[t - lag] - mean);
            }
            acf_vals.push(cov / var);
        }

        Ok(Array1::from_vec(acf_vals))
    }

    /// Partial Autocorrelation Function (PACF) using Levinson-Durbin Recursion
    pub fn pacf(residuals: &Array1<f64>, max_lag: usize) -> Result<Array1<f64>> {
        let acf_vals = Self::acf(residuals, max_lag)?;
        let mut pacf_vals = Vec::with_capacity(max_lag + 1);
        pacf_vals.push(1.0);

        if max_lag == 0 {
            return Ok(Array1::from_vec(pacf_vals));
        }

        pacf_vals.push(acf_vals[1]);

        let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];
        phi[1][1] = acf_vals[1];

        for k in 2..=max_lag {
            let mut num = acf_vals[k];
            let mut den = 1.0;

            for j in 1..k {
                num -= phi[k - 1][j] * acf_vals[k - j];
                den -= phi[k - 1][j] * acf_vals[j];
            }

            let phi_kk = num / den;
            phi[k][k] = phi_kk;
            pacf_vals.push(phi_kk);

            for j in 1..k {
                phi[k][j] = phi[k - 1][j] - phi_kk * phi[k - 1][k - j];
            }
        }

        Ok(Array1::from_vec(pacf_vals))
    }

    /// Ljung-Box Q-Test for Autocorrelation
    pub fn ljung_box(residuals: &Array1<f64>, lags: usize) -> Result<LjungBoxResult> {
        let n = residuals.len() as f64;
        let acf_vals = Self::acf(residuals, lags)?;

        let mut q_stat = 0.0;
        for k in 1..=lags {
            let r_k = acf_vals[k];
            q_stat += (r_k * r_k) / (n - k as f64);
        }
        q_stat *= n * (n + 2.0);

        // Chi-square survival function approximation (1 degree of freedom per lag)
        let p_value = chi2_sf(q_stat, lags as f64);

        Ok(LjungBoxResult {
            q_stat,
            p_value,
            lags,
        })
    }

    /// Jarque-Bera Test for Normality (Skewness & Excess Kurtosis)
    pub fn jarque_bera(residuals: &Array1<f64>) -> Result<JarqueBeraResult> {
        let n = residuals.len() as f64;
        if n < 4.0 {
            return Err(ChronosError::InsufficientData {
                required: 4,
                found: n as usize,
            });
        }

        let mean = residuals.mean().unwrap_or(0.0);
        let m2 = residuals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3 = residuals.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
        let m4 = residuals.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;

        if m2 == 0.0 {
            return Err(ChronosError::InvalidParameters(
                "Zero residual variance".into(),
            ));
        }

        let skewness = m3 / m2.powf(1.5);
        let kurtosis = m4 / m2.powi(2);
        let excess_kurtosis = kurtosis - 3.0;

        let jb_stat = (n / 6.0) * (skewness.powi(2) + (0.25 * excess_kurtosis.powi(2)));
        let p_value = chi2_sf(jb_stat, 2.0);

        Ok(JarqueBeraResult {
            jb_stat,
            p_value,
            skewness,
            kurtosis,
        })
    }

    /// Full Diagnostic Pass across residuals
    pub fn evaluate(residuals: &Array1<f64>, max_lags: usize) -> Result<DiagnosticsResult> {
        let acf_vals = Self::acf(residuals, max_lags)?;
        let pacf_vals = Self::pacf(residuals, max_lags)?;
        let lb = Self::ljung_box(residuals, max_lags)?;
        let jb = Self::jarque_bera(residuals)?;

        Ok(DiagnosticsResult {
            acf: acf_vals,
            pacf: pacf_vals,
            ljung_box: lb,
            jarque_bera: jb,
        })
    }
}

/// Lower incomplete gamma function approximation for Chi-Square distribution survival calculation
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let a = df / 2.0;
    let z = x / 2.0;

    // Regularized incomplete gamma upper bound approximation
    let mut sum = 0.0;
    let mut term = 1.0 / a;
    sum += term;
    for i in 1..100 {
        term *= z / (a + i as f64);
        sum += term;
        if term < 1e-10 {
            break;
        }
    }

    let gamma_sf = (-z + a * z.ln() - gamma_log(a)).exp() * sum;
    gamma_sf.clamp(0.0, 1.0)
}

fn gamma_log(a: f64) -> f64 {
    // Lanczos approximation for ln(gamma(a))
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let mut y = a;
    let mut tmp = a + 5.5;
    tmp -= (a + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for c in &coeffs {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / a).ln()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonMetrics {
    pub horizon_step: usize,
    pub mae: f64,
    pub rmse: f64,
    pub mape: f64,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationReport {
    pub total_folds: usize,
    pub horizon_metrics: Vec<HorizonMetrics>,
    pub overall_mae: f64,
    pub overall_rmse: f64,
}

pub struct CrossValidationEvaluator {
    pub horizon: usize,
    pub initial: usize,
    pub step: usize,
}

impl CrossValidationEvaluator {
    pub fn new(horizon: usize, initial: usize, step: usize) -> Self {
        Self {
            horizon,
            initial,
            step,
        }
    }

    /// Computes horizon-level degradation metrics across rolling cross-validation folds
    pub fn evaluate(
        &self,
        model: &ProphetDecomposition,
        dates: &[NaiveDate],
        y: &Array1<f64>,
    ) -> Result<CrossValidationReport> {
        let n = y.len();
        if n < self.initial + self.horizon {
            return Err(ChronosError::InsufficientData {
                required: self.initial + self.horizon,
                found: n,
            });
        }

        let mut horizon_errors: BTreeMap<usize, Vec<(f64, f64)>> = BTreeMap::new();
        let mut fold_count = 0;

        let mut train_end = self.initial;
        while train_end + self.horizon <= n {
            let train_dates = &dates[..train_end];
            let train_y = y.slice(ndarray::s![..train_end]).to_owned();

            let val_dates = &dates[train_end..train_end + self.horizon];
            let val_y = y.slice(ndarray::s![train_end..train_end + self.horizon]);

            let mut fold_model = model.clone();
            fold_model.fit(train_dates, &train_y, None, None)?;

            let pred = fold_model.predict(val_dates)?;

            for h in 0..self.horizon {
                let actual = val_y[h];
                let predicted = pred.yhat[h];
                horizon_errors
                    .entry(h + 1)
                    .or_default()
                    .push((actual, predicted));
            }

            fold_count += 1;
            train_end += self.step;
        }

        if fold_count == 0 {
            return Err(ChronosError::InvalidParameters(
                "No cross-validation folds were executed".into(),
            ));
        }

        let mut horizon_metrics = Vec::with_capacity(self.horizon);
        let mut total_absolute_error = 0.0;
        let mut total_squared_error = 0.0;
        let mut total_points = 0;

        for (h, pairs) in horizon_errors {
            let count = pairs.len();
            let mut sum_abs_err = 0.0;
            let mut sum_sq_err = 0.0;
            let mut sum_pct_err = 0.0;

            for (actual, predicted) in &pairs {
                let err = (actual - predicted).abs();
                sum_abs_err += err;
                sum_sq_err += err * err;
                if actual.abs() > 1e-8 {
                    sum_pct_err += err / actual.abs();
                }

                total_absolute_error += err;
                total_squared_error += err * err;
                total_points += 1;
            }

            let mae = sum_abs_err / (count as f64);
            let rmse = (sum_sq_err / (count as f64)).sqrt();
            let mape = (sum_pct_err / (count as f64)) * 100.0;

            horizon_metrics.push(HorizonMetrics {
                horizon_step: h,
                mae,
                rmse,
                mape,
                sample_count: count,
            });
        }

        let overall_mae = total_absolute_error / (total_points as f64);
        let overall_rmse = (total_squared_error / (total_points as f64)).sqrt();

        Ok(CrossValidationReport {
            total_folds: fold_count,
            horizon_metrics,
            overall_mae,
            overall_rmse,
        })
    }
}
