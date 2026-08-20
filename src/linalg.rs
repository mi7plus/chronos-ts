//! Minimal, dependency-free linear algebra.
//!
//! `chronos-ts` only needs to solve small, dense linear systems (OLS/Ridge normal
//! equations, Kalman smoother gains). Rather than link a full LAPACK/BLAS backend
//! (which forces a C toolchain and platform-specific builds), we provide small,
//! self-contained routines based on Gaussian elimination with partial pivoting.
//! All problems solved here are tiny (a handful of parameters), so the O(n^3)
//! cost is irrelevant.

use ndarray::{Array1, Array2};

/// Solves the dense linear system `a * x = b` via Gaussian elimination with
/// partial pivoting. `a` must be square and non-singular.
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(format!(
            "solve: matrix must be square, got {}x{}",
            n,
            a.ncols()
        ));
    }
    if b.len() != n {
        return Err(format!(
            "solve: rhs length {} does not match matrix dimension {}",
            b.len(),
            n
        ));
    }

    // Work on owned copies so the caller's data is untouched.
    let mut m = a.clone();
    let mut rhs = b.clone();

    for col in 0..n {
        // Partial pivot: find the row with the largest magnitude in this column.
        let mut pivot_row = col;
        let mut pivot_val = m[[col, col]].abs();
        for r in (col + 1)..n {
            let v = m[[r, col]].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }

        if pivot_val < 1e-12 {
            return Err("solve: matrix is singular or near-singular".to_string());
        }

        if pivot_row != col {
            swap_rows(&mut m, col, pivot_row);
            rhs.swap(col, pivot_row);
        }

        // Eliminate entries below the pivot.
        let pivot = m[[col, col]];
        for r in (col + 1)..n {
            let factor = m[[r, col]] / pivot;
            if factor != 0.0 {
                for c in col..n {
                    let sub = factor * m[[col, c]];
                    m[[r, c]] -= sub;
                }
                rhs[r] -= factor * rhs[col];
            }
        }
    }

    // Back substitution.
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut acc = rhs[i];
        for j in (i + 1)..n {
            acc -= m[[i, j]] * x[j];
        }
        x[i] = acc / m[[i, i]];
    }

    Ok(x)
}

/// Inverts a square matrix via Gauss-Jordan elimination with partial pivoting.
pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(format!(
            "inv: matrix must be square, got {}x{}",
            n,
            a.ncols()
        ));
    }

    let mut m = a.clone();
    let mut out = Array2::<f64>::eye(n);

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = m[[col, col]].abs();
        for r in (col + 1)..n {
            let v = m[[r, col]].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }

        if pivot_val < 1e-12 {
            return Err("inv: matrix is singular or near-singular".to_string());
        }

        if pivot_row != col {
            swap_rows(&mut m, col, pivot_row);
            swap_rows(&mut out, col, pivot_row);
        }

        // Normalize the pivot row.
        let pivot = m[[col, col]];
        for c in 0..n {
            m[[col, c]] /= pivot;
            out[[col, c]] /= pivot;
        }

        // Eliminate the pivot column from every other row.
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[[r, col]];
            if factor != 0.0 {
                for c in 0..n {
                    let mm = factor * m[[col, c]];
                    m[[r, c]] -= mm;
                    let oo = factor * out[[col, c]];
                    out[[r, c]] -= oo;
                }
            }
        }
    }

    Ok(out)
}

/// Solves the ordinary least squares problem `min ||X b - y||` for a tall matrix
/// `X` (n x p, n >= p) via the normal equations `(XᵀX) b = Xᵀy`.
///
/// Returns the coefficient vector `b`. For rank-deficient or ill-conditioned
/// systems callers should prefer adding regularization to `XᵀX` before solving.
pub fn lstsq(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    solve(&xtx, &xty)
}

fn swap_rows(m: &mut Array2<f64>, a: usize, b: usize) {
    if a == b {
        return;
    }
    let ncols = m.ncols();
    for c in 0..ncols {
        let tmp = m[[a, c]];
        m[[a, c]] = m[[b, c]];
        m[[b, c]] = tmp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn solves_simple_system() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![3.0, 5.0];
        let x = solve(&a, &b).unwrap();
        // Expected solution: x = 0.8, y = 1.4
        assert!((x[0] - 0.8).abs() < 1e-10);
        assert!((x[1] - 1.4).abs() < 1e-10);
    }

    #[test]
    fn inverts_matrix() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let ainv = inv(&a).unwrap();
        let prod = a.dot(&ainv);
        let eye = Array2::<f64>::eye(2);
        assert!((&prod - &eye).mapv(f64::abs).sum() < 1e-10);
    }

    #[test]
    fn detects_singular() {
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        let b = array![1.0, 2.0];
        assert!(solve(&a, &b).is_err());
    }

    #[test]
    fn lstsq_recovers_line() {
        // y = 2 + 3x with an intercept column.
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let y = array![2.0, 5.0, 8.0, 11.0];
        let b = lstsq(&x, &y).unwrap();
        assert!((b[0] - 2.0).abs() < 1e-9);
        assert!((b[1] - 3.0).abs() < 1e-9);
    }
}
