use ndarray::{Array1, Array2};
use ndarray_linalg::Solve; // Trait providing matrix linear solvers

pub struct ProphetRidgeFitter {
    pub l2_penalty: f64, // Lambda parameter
}

/// Solves (X^T * X + lambda * I) * beta = X^T * y without LAPACK/BLAS dependencies
pub fn fit_ridge_pure_rust(
    x: &Array2<f64>,
    y: &Array1<f64>,
    l2_penalty: f64,
) -> Result<Array1<f64>, String> {
    let (_n, p) = x.dim();

    let xt_x = x.t().dot(x);
    let mut penalty_mat = Array2::<f64>::eye(p) * l2_penalty;
    penalty_mat[[0, 0]] = 0.0; // Don't penalize bias/intercept

    let mut a = xt_x + penalty_mat;
    let mut b = x.t().dot(y);

    // Forward Gauss-Jordan Elimination with Pivoting
    for i in 0..p {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..p {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows in A and B
        for col in 0..p {
            let tmp = a[[i, col]];
            a[[i, col]] = a[[max_row, col]];
            a[[max_row, col]] = tmp;
        }
        let tmp_b = b[i];
        b[i] = b[max_row];
        b[max_row] = tmp_b;

        if a[[i, i]].abs() < 1e-12 {
            return Err("Singular matrix encountered during Ridge solving".to_string());
        }

        // Pivot normalization
        let pivot = a[[i, i]];
        for col in 0..p {
            a[[i, col]] /= pivot;
        }
        b[i] /= pivot;

        // Eliminate column entries
        for k in 0..p {
            if k != i {
                let factor = a[[k, i]];
                for col in 0..p {
                    a[[k, col]] -= factor * a[[i, col]];
                }
                b[k] -= factor * b[i];
            }
        }
    }

    Ok(b)
}

impl ProphetRidgeFitter {
    pub fn new(l2_penalty: f64) -> Self {
        Self { l2_penalty }
    }

    /// Fits the parameters using Ridge Regression: (X^T * X + lambda * I)^(-1) * X^T * y
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (N, P) [intercept, time, changepoints, Fourier terms]
    /// * `y` - Target vector of shape (N)
    ///
    /// # Returns
    /// * Vector of fitted coefficients of shape (P)
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        let (_n, p) = x.dim();

        // 1. Compute X^T * X  --> shape (P, P)
        let xt_x = x.t().dot(x);

        // 2. Build L2 Penalty Matrix (lambda * I)
        // Note: Commonly, we do NOT penalize the intercept term (column 0)
        let mut penalty_mat = Array2::<f64>::eye(p) * self.l2_penalty;
        penalty_mat[[0, 0]] = 0.0; // Keep intercept unregularized

        let xt_x_reg = xt_x + penalty_mat;

        // 3. Compute X^T * y --> shape (P)
        let xt_y = x.t().dot(y);

        // 4. Solve the linear system (X^T * X + lambda * I) * beta = X^T * y
        xt_x_reg
            .solve_into(xt_y)
            .map_err(|e| format!("Failed to solve Ridge Regression system: {}", e))
    }

    /// Computes predictions y_hat given fitted coefficients
    pub fn predict(&self, x: &Array2<f64>, beta: &Array1<f64>) -> Array1<f64> {
        x.dot(beta)
    }
}