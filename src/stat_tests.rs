#![allow(non_snake_case)]
use ndarray::{s, Array1, Array2};
use ndarray_linalg::LeastSquaresSvd;
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Augmented Dickey-Fuller (ADF) Test
/// Null Hypothesis (H0): The series has a unit root (is non-stationary).
/// Alternative (H1): The series is stationary.
pub struct AdfTestResult {
    pub stat: f64,
    pub p_value: f64,
    pub used_lags: usize,
}

pub fn adf_test(series: &Array1<f64>, max_lags: Option<usize>) -> AdfTestResult {
    let n = series.len();

    // 1. Minimum length guard: need at least 4-5 points to fit even a zero-lag regression
    if n < 4 {
        return AdfTestResult {
            stat: 0.0,
            p_value: 1.0,
            used_lags: 0,
        };
    }

    // 2. Zero-variance / constant series guard
    let std_dev = series.std(0.0);
    if std_dev.abs() < f64::EPSILON || std_dev.is_nan() {
        return AdfTestResult {
            stat: 0.0,
            p_value: 0.0, // Constant series has no unit root (stationary)
            used_lags: 0,
        };
    }

    let lags = max_lags.unwrap_or_else(|| ((n as f64 - 1.0).powf(1.0 / 3.0)) as usize);

    // Delta Y_t = Y_t - Y_{t-1}
    let dy = &series.slice(s![1..]) - &series.slice(s![..-1]);
    let dy_len = dy.len();

    if dy_len <= lags {
        return AdfTestResult {
            stat: 0.0,
            p_value: 1.0,
            used_lags: lags,
        };
    }

    let effective_n = dy_len - lags;
    let cols = 2 + lags;

    // Ensure degrees of freedom are positive
    if effective_n <= cols {
        return AdfTestResult {
            stat: 0.0,
            p_value: 1.0,
            used_lags: lags,
        };
    }

    // Dependent variable: Delta Y_t from t = lags to dy_len
    let y_dep = dy.slice(s![lags..dy_len]).to_owned();

    // Design Matrix X: [Y_{t-1}, 1 (constant), Delta Y_{t-1}, ..., Delta Y_{t-lags}]
    let mut x = Array2::<f64>::zeros((effective_n, cols));

    for i in 0..effective_n {
        let idx = i + lags;
        x[[i, 0]] = series[idx]; // Level term Y_{t-1}
        x[[i, 1]] = 1.0; // Intercept term

        for j in 0..lags {
            x[[i, 2 + j]] = dy[idx - 1 - j]; // Lagged differences
        }
    }

    // OLS via Singular Value Decomposition
    let sol = match x.least_squares(&y_dep) {
        Ok(s) => s,
        Err(_) => {
            return AdfTestResult {
                stat: 0.0,
                p_value: 1.0,
                used_lags: lags,
            };
        }
    };
    let beta = sol.solution;

    // Compute standard error of gamma (beta[0])
    let residuals = &y_dep - &x.dot(&beta);
    let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
    let df = effective_n - cols;
    let mse = sse / (df as f64);

    let xtx_inv = match (x.t().dot(&x)).least_squares(&Array2::eye(cols)) {
        Ok(s) => s.solution,
        Err(_) => {
            return AdfTestResult {
                stat: 0.0,
                p_value: 1.0,
                used_lags: lags,
            };
        }
    };

    let se_gamma = (mse * xtx_inv[[0, 0]]).sqrt();

    if se_gamma <= 0.0 || se_gamma.is_nan() {
        return AdfTestResult {
            stat: 0.0,
            p_value: 0.0,
            used_lags: lags,
        };
    }

    let t_stat = beta[0] / se_gamma;

    // Safe p-value evaluation with statrs
    let p_value = match StudentsT::new(0.0, 1.0, df as f64) {
        Ok(t_dist) => {
            let p = t_dist.cdf(t_stat);
            if p.is_nan() {
                1.0
            } else {
                p.clamp(0.0, 1.0)
            }
        }
        Err(_) => 1.0,
    };

    AdfTestResult {
        stat: t_stat,
        p_value,
        used_lags: lags,
    }
}

/// Automatically determines required non-seasonal differencing d
pub fn estimate_d(series: &Array1<f64>, max_d: usize, alpha: f64) -> usize {
    let mut current = series.clone();
    let mut d = 0;

    while d < max_d {
        let res = adf_test(&current, None);
        if res.p_value < alpha {
            // Reject H0 -> Series is stationary
            break;
        }
        // Fail to reject H0 -> Need differencing
        if current.len() <= 2 {
            break;
        }
        current = &current.slice(s![1..]) - &current.slice(s![..-1]);
        d += 1;
    }
    d
}

/// Estimates required seasonal differencing D using OCSB / Canova-Hansen surrogate test
pub fn estimate_D(series: &Array1<f64>, m: usize, max_D: usize) -> usize {
    if m <= 1 || series.len() < 2 * m {
        return 0;
    }

    let mut current = series.clone();
    let mut D = 0;

    while D < max_D {
        let n = current.len();
        if n <= 2 * m {
            break;
        }

        // Calculate seasonal strength index
        let m_neg = -(m as isize);
        let seasonal_diff = &current.slice(s![m..]) - &current.slice(s![..m_neg]);
        let var_orig = crate::utils::variance(&current);
        let var_sdiff = crate::utils::variance(&seasonal_diff);

        // Seasonal strength F_s = max(0, 1 - Var(res) / Var(res + seasonal))
        let seasonal_strength = (1.0 - (var_sdiff / var_orig)).max(0.0);

        if seasonal_strength < 0.64 {
            break;
        }

        current = seasonal_diff;
        D += 1;
    }
    D
}