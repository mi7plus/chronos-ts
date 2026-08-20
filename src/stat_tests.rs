#![allow(non_snake_case)]
use crate::linalg;
use ndarray::{s, Array1, Array2};

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

    // OLS via the normal equations (pure Rust, no LAPACK backend required).
    let beta = match linalg::lstsq(&x, &y_dep) {
        Ok(b) => b,
        Err(_) => {
            return AdfTestResult {
                stat: 0.0,
                p_value: 1.0,
                used_lags: lags,
            };
        }
    };

    // Compute standard error of gamma (beta[0])
    let residuals = &y_dep - &x.dot(&beta);
    let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
    let df = effective_n - cols;
    let mse = sse / (df as f64);

    let xtx_inv = match linalg::inv(&x.t().dot(&x)) {
        Ok(m) => m,
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

    // The ADF statistic does NOT follow a Student-t distribution under the null;
    // it follows the (non-standard) Dickey-Fuller distribution. Map the statistic
    // to a p-value using tabulated DF quantiles for the constant-only case.
    let p_value = dickey_fuller_pvalue(t_stat);

    AdfTestResult {
        stat: t_stat,
        p_value,
        used_lags: lags,
    }
}

/// Approximate p-value of the Augmented Dickey-Fuller statistic for the
/// "constant, no trend" regression case.
///
/// The ADF `tau` statistic follows the Dickey-Fuller distribution rather than a
/// Student-t. This uses the well-established asymptotic quantiles of that
/// distribution (Fuller 1976 / MacKinnon) and performs monotone linear
/// interpolation of the CDF, which is accurate around the decision region
/// (~1%-10%) that matters for differencing decisions. Values outside the table
/// are clamped to `[0, 1]`.
fn dickey_fuller_pvalue(tau: f64) -> f64 {
    // (tau quantile, cumulative probability) pairs, ascending in tau.
    // Left tail => small p (reject unit root / stationary).
    const TABLE: [(f64, f64); 8] = [
        (-3.43, 0.01),
        (-3.12, 0.025),
        (-2.86, 0.05),
        (-2.57, 0.10),
        (-0.44, 0.90),
        (-0.07, 0.95),
        (0.23, 0.975),
        (0.60, 0.99),
    ];

    if tau <= TABLE[0].0 {
        return 0.01;
    }
    let last = TABLE[TABLE.len() - 1];
    if tau >= last.0 {
        return 0.99;
    }

    for w in TABLE.windows(2) {
        let (t0, p0) = w[0];
        let (t1, p1) = w[1];
        if tau >= t0 && tau <= t1 {
            let frac = (tau - t0) / (t1 - t0);
            return (p0 + frac * (p1 - p0)).clamp(0.0, 1.0);
        }
    }

    // Unreachable given the bounds checks above, but stay safe.
    1.0
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

/// Estimates the required seasonal differencing order `D` using a seasonal-strength
/// heuristic (not a formal OCSB / Canova-Hansen unit-root test).
///
/// At each step it seasonally differences the series and measures how much variance
/// that removes: `F_s = max(0, 1 - Var(seasonally differenced) / Var(current))`. If
/// `F_s` exceeds a fixed threshold (0.64) the seasonal component is deemed strong
/// enough to warrant another seasonal difference. This mirrors the strength-based
/// rule popularised by Wang, Smith & Hyndman and is cheaper than a full unit-root
/// test, at the cost of some statistical rigor.
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
