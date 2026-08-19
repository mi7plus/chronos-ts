use crate::errors::{ChronosError, Result};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Inverse;
use serde::{Deserialize, Serialize};

/// State-space specification:
/// Y_t = Z * alpha_t + eps_t,        eps_t ~ N(0, H)
/// alpha_{t+1} = T * alpha_t + R * eta_t,  eta_t ~ N(0, Q)
#[derive(Serialize, Deserialize)]
pub struct StateSpaceModel {
    #[serde(with = "crate::utils::serde_array2")]
    pub transition_matrix: Array2<f64>, // T (m x m)
    #[serde(with = "crate::utils::serde_array2")]
    pub selection_matrix: Array2<f64>,  // R (m x r)
    #[serde(with = "crate::utils::serde_array1")]
    pub design_matrix: Array1<f64>,     // Z (m)
    #[serde(with = "crate::utils::serde_array2")]
    pub state_cov: Array2<f64>,         // Q (r x r)
    pub obs_cov: f64,                   // H (scalar)
}

#[derive(Serialize, Deserialize)]
pub struct FilterStepResult {
    #[serde(with = "crate::utils::serde_array1")]
    pub a_prior: Array1<f64>,  // a_{t|t-1}
    #[serde(with = "crate::utils::serde_array2")]
    pub p_prior: Array2<f64>,  // P_{t|t-1}
    #[serde(with = "crate::utils::serde_array1")]
    pub a_post: Array1<f64>,   // a_{t|t}
    #[serde(with = "crate::utils::serde_array2")]
    pub p_post: Array2<f64>,   // P_{t|t}
    pub v: f64,                // Innovation v_t
    pub f: f64,                // Innovation variance F_t
    #[serde(with = "crate::utils::serde_array1")]
    pub k: Array1<f64>,        // Kalman gain K_t
}

pub struct KalmanFilterResult {
    pub steps: Vec<FilterStepResult>,
    pub log_likelihood: f64,
}

pub struct SmootherResult {
    pub smoothed_states: Vec<Array1<f64>>, // a_{t|N}
    pub smoothed_covs: Vec<Array2<f64>>,   // P_{t|N}
}

pub struct KalmanFilter<'a> {
    pub model: &'a StateSpaceModel,
}

impl StateSpaceModel {
    pub fn local_linear_trend(sigma_obs: f64, sigma_level: f64, sigma_trend: f64) -> Self {
        use ndarray::{array, Array2};

        // 1D observation vector (Z): [1.0, 0.0]
        let design_matrix = array![1.0, 0.0];

        // 2D transition matrix (T): [[1.0, 1.0], [0.0, 1.0]]
        let transition_matrix = array![[1.0, 1.0], [0.0, 1.0]];

        // Selection matrix (R): Identity matrix mapping state error terms
        let selection_matrix = Array2::eye(2);

        // State covariance matrix (Q)
        let state_cov = array![
            [sigma_level.powi(2), 0.0],
            [0.0, sigma_trend.powi(2)]
        ];

        // Observation covariance variance scalar (sigma^2_obs)
        let obs_cov = sigma_obs.powi(2);

        Self {
            transition_matrix,
            selection_matrix,
            design_matrix,
            state_cov,
            obs_cov,
        }
    }
}

impl<'a> KalmanFilter<'a> {
    pub fn new(model: &'a StateSpaceModel) -> Self {
        Self { model }
    }

    /// Performs exact Forward Kalman Filtering with optional Diffuse State Initialization
    pub fn filter(
        &self,
        observations: &Array1<f64>,
        a0: Option<Array1<f64>>,
        p0: Option<Array2<f64>>,
    ) -> Result<KalmanFilterResult> {
        let n = observations.len();
        let m = self.model.design_matrix.len();

        // 1. Initialize states: If None, apply Diffuse Initializer (large covariance scalar kappa * I)
        let mut a = a0.unwrap_or_else(|| Array1::zeros(m));
        let mut p = p0.unwrap_or_else(|| {
            let kappa = 1e6; // Diffuse initialization variance
            Array2::eye(m) * kappa
        });

        let mut steps = Vec::with_capacity(n);
        let mut log_like = 0.0;

        let t = &self.model.transition_matrix;
        let r = &self.model.selection_matrix;
        let z = &self.model.design_matrix;
        let q = &self.model.state_cov;
        let h = self.model.obs_cov;

        let rqr = r.dot(q).dot(&r.t());

        for t_idx in 0..n {
            let y_t = observations[t_idx];

            let a_prior = a.clone();
            let p_prior = p.clone();

            // Measurement Prediction & Innovation
            let y_hat = z.dot(&a_prior);
            let v_t = y_t - y_hat;

            // Innovation Variance F_t = Z * P * Z^T + H
            let f_t = z.dot(&p_prior.dot(z)) + h;

            if f_t <= 0.0 {
                return Err(ChronosError::ConvergenceFailure(
                    "Non-positive innovation variance encountered during filter pass".to_string(),
                ));
            }

            // Kalman Gain K_t = P * Z^T / F_t
            let k_t = p_prior.dot(z) / f_t;

            // Posterior Updating
            let a_post = &a_prior + &(&k_t * v_t);

            // P_{t|t} = P_{t|t-1} - K_t * Z * P_{t|t-1}
            let z_row = z.clone().insert_axis(Axis(0));
            let k_col = k_t.clone().insert_axis(Axis(1));
            let p_post = &p_prior - &k_col.dot(&z_row.dot(&p_prior));

            // Replace LN_2_PI with std::f64::consts::TAU.ln()
            let ln_2_pi = std::f64::consts::TAU.ln(); // ln(2 * PI)
            log_like -= 0.5 * (ln_2_pi + f_t.ln() + (v_t * v_t) / f_t);

            steps.push(FilterStepResult {
                a_prior: a_prior.clone(),
                p_prior: p_prior.clone(),
                a_post: a_post.clone(),
                p_post: p_post.clone(),
                v: v_t,
                f: f_t,
                k: k_t,
            });

            // Time Update (Predict next state)
            a = t.dot(&a_post);
            p = t.dot(&p_post).dot(&t.t()) + &rqr;
        }

        Ok(KalmanFilterResult {
            steps,
            log_likelihood: log_like,
        })
    }

    /// Rauch-Tung-Striebel (RTS) Backward Smoother Pass
    /// Computes conditional state distributions given ALL observations: a_{t|N} and P_{t|N}
    pub fn smooth(&self, filter_result: &KalmanFilterResult) -> Result<SmootherResult> {
        let n = filter_result.steps.len();
        if n == 0 {
            return Err(ChronosError::InsufficientData {
                required: 1,
                found: 0,
            });
        }

        let m = self.model.design_matrix.len();
        let t = &self.model.transition_matrix;

        let mut smoothed_states = vec![Array1::zeros(m); n];
        let mut smoothed_covs = vec![Array2::zeros((m, m)); n];

        // Terminal state at t = N is identical to the filtered state
        smoothed_states[n - 1] = filter_result.steps[n - 1].a_post.clone();
        smoothed_covs[n - 1] = filter_result.steps[n - 1].p_post.clone();

        // Backward recursion from t = N-2 down to 0
        for t_idx in (0..n - 1).rev() {
            let step_curr = &filter_result.steps[t_idx];
            let step_next = &filter_result.steps[t_idx + 1];

            // Predicted covariance for next step: P_{t+1|t} = T * P_{t|t} * T^T + R*Q*R^T
            let p_next_pred = &step_next.p_prior;

            // Smoother Gain C_t = P_{t|t} * T^T * [P_{t+1|t}]^-1
            let p_next_inv = p_next_pred.inv()?;
            let c_t = step_curr.p_post.dot(&t.t()).dot(&p_next_inv);

            // Smoothed State: a_{t|N} = a_{t|t} + C_t * (a_{t+1|N} - a_{t+1|t})
            let a_diff = &smoothed_states[t_idx + 1] - &step_next.a_prior;
            smoothed_states[t_idx] = &step_curr.a_post + &c_t.dot(&a_diff);

            // Smoothed Covariance: P_{t|N} = P_{t|t} + C_t * (P_{t+1|N} - P_{t+1|t}) * C_t^T
            let p_diff = &smoothed_covs[t_idx + 1] - p_next_pred;
            smoothed_covs[t_idx] = &step_curr.p_post + &c_t.dot(&p_diff).dot(&c_t.t());
        }

        Ok(SmootherResult {
            smoothed_states,
            smoothed_covs,
        })
    }
}