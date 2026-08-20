//! Exact Gaussian maximum-likelihood estimation for ARIMA/SARIMA via the Kalman
//! filter.
//!
//! The (already differenced and mean-centered) series is cast into an ARMA
//! state-space model in Harvey companion form. The Kalman filter yields the exact
//! Gaussian log-likelihood; the innovation variance `sigma^2` is concentrated out
//! analytically, so only the AR/MA coefficients are optimized (with Nelder-Mead).
//! This connects the crate's state-space machinery to the ARIMA engine and gives
//! an alternative to the faster Conditional Sum of Squares estimator.

use crate::arima::{split_params, SarimaOrder};
use crate::arima_poly::{ar_polynomial, ma_polynomial};
use crate::linalg;
use crate::statespace::{KalmanFilter, StateSpaceModel};
use argmin::core::{CostFunction, Executor, State};
use argmin::solver::neldermead::NelderMead;
use ndarray::{Array1, Array2};

/// Result of an exact-MLE fit: coefficient blocks plus the estimated innovation
/// variance and the maximized Gaussian log-likelihood.
pub(crate) struct MleEstimate {
    pub ar: Array1<f64>,
    pub ma: Array1<f64>,
    pub sar: Array1<f64>,
    pub sma: Array1<f64>,
    pub sigma2: f64,
    pub log_likelihood: f64,
}

/// Builds an ARMA state-space model (Harvey companion form) from the full AR/MA
/// operator polynomials (seasonal factors already expanded). `sigma^2` is set to 1
/// and concentrated out later; there is no separate observation noise.
fn build_arma_state_space(ar_poly: &[f64], ma_poly: &[f64]) -> StateSpaceModel {
    let p = ar_poly.len().saturating_sub(1);
    let q = ma_poly.len().saturating_sub(1);
    let r = p.max(q + 1).max(1);

    // phi_i = -ar_poly[i]; theta_j = ma_poly[j].
    let mut transition = Array2::zeros((r, r));
    for i in 0..r {
        let phi = if i < p { -ar_poly[i + 1] } else { 0.0 };
        transition[[i, 0]] = phi;
        if i + 1 < r {
            transition[[i, i + 1]] = 1.0;
        }
    }

    let mut selection = Array2::zeros((r, 1));
    selection[[0, 0]] = 1.0;
    for j in 1..r {
        selection[[j, 0]] = if j <= q { ma_poly[j] } else { 0.0 };
    }

    let mut design = Array1::zeros(r);
    design[0] = 1.0;

    StateSpaceModel {
        transition_matrix: transition,
        selection_matrix: selection,
        design_matrix: design,
        state_cov: Array2::from_elem((1, 1), 1.0),
        obs_cov: 0.0,
    }
}

/// Solves the discrete Lyapunov equation `P = T P T' + R Q R'` for the stationary
/// state covariance, used to initialize the Kalman filter exactly. Returns `None`
/// for a non-stationary transition (which doubles as a stationarity constraint).
fn stationary_state_cov(model: &StateSpaceModel) -> Option<Array2<f64>> {
    let t = &model.transition_matrix;
    let r = t.nrows();
    let rqr = model
        .selection_matrix
        .dot(&model.state_cov)
        .dot(&model.selection_matrix.t());

    // vec(P) = (I - T (x) T)^{-1} vec(R Q R'), where (x) is the Kronecker product.
    let n2 = r * r;
    let mut a = Array2::<f64>::eye(n2);
    for i in 0..r {
        for j in 0..r {
            for k in 0..r {
                for l in 0..r {
                    a[[i * r + j, k * r + l]] -= t[[i, k]] * t[[j, l]];
                }
            }
        }
    }
    let mut b = Array1::<f64>::zeros(n2);
    for i in 0..r {
        for j in 0..r {
            b[i * r + j] = rqr[[i, j]];
        }
    }

    let sol = linalg::solve(&a, &b).ok()?;
    let mut p = Array2::zeros((r, r));
    for i in 0..r {
        for j in 0..r {
            p[[i, j]] = sol[i * r + j];
        }
    }
    // A valid stationary covariance must have non-negative variances.
    if (0..r).any(|i| p[[i, i]] < -1e-6 || !p[[i, i]].is_finite()) {
        return None;
    }
    Some(p)
}

/// Runs the Kalman filter for a candidate coefficient vector and returns
/// `(concentrated sigma^2, sum(ln f_t), n)`, or `None` if the filter is degenerate
/// or the candidate is non-stationary.
fn filter_stats(w: &Array1<f64>, order: &SarimaOrder, params: &[f64]) -> Option<(f64, f64, usize)> {
    let (ar, ma, sar, sma) = split_params(params, order);
    let ar_poly = ar_polynomial(order, &ar.to_vec(), &sar.to_vec(), false);
    let ma_poly = ma_polynomial(order, &ma.to_vec(), &sma.to_vec());
    let model = build_arma_state_space(&ar_poly, &ma_poly);

    // Exact stationary initialization; reject non-stationary candidates.
    let p0 = stationary_state_cov(&model)?;
    let a0 = Array1::zeros(model.design_matrix.len());

    let filter = KalmanFilter::new(&model);
    let res = filter.filter(w, Some(a0), Some(p0)).ok()?;
    let n = res.steps.len();
    if n == 0 {
        return None;
    }

    let mut sum_ln_f = 0.0;
    let mut sum_v2f = 0.0;
    for step in &res.steps {
        if !step.f.is_finite() || step.f <= 0.0 {
            return None;
        }
        sum_ln_f += step.f.ln();
        sum_v2f += step.v * step.v / step.f;
    }
    let sigma2 = (sum_v2f / n as f64).max(1e-12);
    Some((sigma2, sum_ln_f, n))
}

/// Concentrated negative log-likelihood (constants dropped) for the optimizer.
struct MleCost<'a> {
    w: &'a Array1<f64>,
    order: SarimaOrder,
}

impl<'a> CostFunction for MleCost<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        match filter_stats(self.w, &self.order, params) {
            Some((sigma2, sum_ln_f, n)) => Ok(0.5 * (n as f64 * sigma2.ln() + sum_ln_f)),
            None => Ok(f64::INFINITY),
        }
    }
}

/// Full Gaussian log-likelihood at the given parameters (with all constants).
fn full_log_likelihood(w: &Array1<f64>, order: &SarimaOrder, params: &[f64]) -> Option<(f64, f64)> {
    let (sigma2, sum_ln_f, n) = filter_stats(w, order, params)?;
    let n = n as f64;
    // Because sum(v^2/f) == n * sigma2 at the concentrated estimate, the last term
    // collapses to n.
    let ll = -0.5 * (n * (2.0 * std::f64::consts::PI).ln() + n * sigma2.ln() + sum_ln_f + n);
    Some((sigma2, ll))
}

/// Estimates SARIMA coefficients on the (differenced, centered) series `w` by exact
/// Gaussian maximum likelihood.
pub(crate) fn estimate_mle(w: &Array1<f64>, order: &SarimaOrder) -> Option<MleEstimate> {
    let k = order.p + order.q + order.P + order.Q;

    let best_params: Vec<f64> = if k == 0 {
        Vec::new()
    } else {
        let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(k + 1);
        simplex.push(vec![0.0; k]);
        for i in 0..k {
            let mut vertex = vec![0.0; k];
            vertex[i] = 0.3;
            simplex.push(vertex);
        }

        let cost = MleCost { w, order: *order };
        let solver = NelderMead::new(simplex).with_sd_tolerance(1e-9).ok()?;
        let outcome = Executor::new(cost, solver)
            .configure(|state| state.max_iters(1000))
            .run()
            .ok()?;
        outcome.state.get_best_param()?.clone()
    };

    let (sigma2, log_likelihood) = full_log_likelihood(w, order, &best_params)?;
    let (ar, ma, sar, sma) = split_params(&best_params, order);

    Some(MleEstimate {
        ar,
        ma,
        sar,
        sma,
        sigma2,
        log_likelihood,
    })
}
