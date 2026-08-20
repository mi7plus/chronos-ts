//! Lag-polynomial helpers for ARIMA: building the combined AR/MA operators
//! (including differencing), deriving MA(∞) `psi`-weights for forecast-variance
//! calculations, and a stationarity guard used to keep CSS estimates well-behaved.
//!
//! Polynomials are represented as coefficient vectors in the backshift operator
//! `B`, with index `l` holding the coefficient of `B^l` and index `0` the
//! constant term (always `1` for the operators built here).

use crate::arima::SarimaOrder;

/// Multiplies two polynomials given as coefficient vectors.
pub fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// Builds `1 + sign * (c0 B + c1 B^2 + ...)` — coefficients placed at consecutive lags.
fn lag_poly(coeffs: &[f64], sign: f64) -> Vec<f64> {
    let mut out = vec![0.0; coeffs.len() + 1];
    out[0] = 1.0;
    for (k, &c) in coeffs.iter().enumerate() {
        out[k + 1] = sign * c;
    }
    out
}

/// Builds `1 + sign * (c0 B^m + c1 B^{2m} + ...)` — coefficients placed at seasonal lags.
fn seasonal_lag_poly(coeffs: &[f64], m: usize, sign: f64) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![1.0];
    }
    let m = m.max(1);
    let mut out = vec![0.0; coeffs.len() * m + 1];
    out[0] = 1.0;
    for (k, &c) in coeffs.iter().enumerate() {
        out[(k + 1) * m] = sign * c;
    }
    out
}

/// The combined autoregressive operator `phi(B) * Phi(B^m)`, optionally folding in
/// the differencing operators `(1-B)^d (1-B^m)^D`.
///
/// Returns the polynomial with `[0] == 1`.
pub fn ar_polynomial(order: &SarimaOrder, ar: &[f64], sar: &[f64], fold_diff: bool) -> Vec<f64> {
    let ns = lag_poly(ar, -1.0);
    let s = seasonal_lag_poly(sar, order.m, -1.0);
    let mut poly = poly_mul(&ns, &s);

    if fold_diff {
        for _ in 0..order.d {
            poly = poly_mul(&poly, &[1.0, -1.0]);
        }
        if order.m > 1 {
            // (1 - B^m): coefficient 1 at lag 0 and -1 at lag m.
            let mut sdiff = vec![0.0; order.m + 1];
            sdiff[0] = 1.0;
            sdiff[order.m] = -1.0;
            for _ in 0..order.D {
                poly = poly_mul(&poly, &sdiff);
            }
        }
    }

    poly
}

/// The combined moving-average operator `theta(B) * Theta(B^m)`.
///
/// Uses the crate's `+theta` sign convention (see `compute_residuals`). Returns the
/// polynomial with `[0] == 1`.
pub fn ma_polynomial(order: &SarimaOrder, ma: &[f64], sma: &[f64]) -> Vec<f64> {
    let ns = lag_poly(ma, 1.0);
    let s = seasonal_lag_poly(sma, order.m, 1.0);
    poly_mul(&ns, &s)
}

/// Computes the MA(∞) representation `psi`-weights of an ARMA model from its AR and
/// MA operator polynomials, up to and including lag `horizon` (so `horizon + 1`
/// weights, with `psi[0] == 1`).
pub fn psi_weights(ar_poly: &[f64], ma_poly: &[f64], horizon: usize) -> Vec<f64> {
    let mut psi = vec![0.0; horizon + 1];
    psi[0] = 1.0;
    for j in 1..=horizon {
        let mut val = ma_poly.get(j).copied().unwrap_or(0.0);
        for l in 1..ar_poly.len() {
            if j >= l {
                val -= ar_poly[l] * psi[j - l];
            }
        }
        psi[j] = val;
    }
    psi
}

/// Returns true if the (stationary part of the) ARMA model is stable, i.e. its
/// impulse-response `psi`-weights stay bounded and decay rather than explode.
///
/// This is a pragmatic guard: an explosive AR polynomial produces `psi`-weights
/// that grow without bound, which would make forecasts and prediction intervals
/// diverge. `ar_poly`/`ma_poly` must be the operators WITHOUT any differencing
/// folded in (integrated series are expected to have non-decaying weights).
pub fn is_stable(ar_poly: &[f64], ma_poly: &[f64]) -> bool {
    const HORIZON: usize = 200;
    let psi = psi_weights(ar_poly, ma_poly, HORIZON);
    let max_abs = psi.iter().copied().map(f64::abs).fold(0.0, f64::max);
    max_abs.is_finite() && max_abs < 1.0e3 && psi[HORIZON].abs() < 1.0
}

/// Returns true if the moving-average operator is invertible, i.e. the AR(∞)
/// (`pi`-weight) expansion `1 / theta(B)` stays bounded and decays. A non-invertible
/// MA has an unstable inverse and an ill-defined residual recursion.
pub fn is_invertible(ma_poly: &[f64]) -> bool {
    // The pi-weights of 1/theta(B) are the psi-weights of the AR-only model whose
    // AR operator is theta(B): treat `ma_poly` as the AR side and unit MA.
    is_stable(ma_poly, &[1.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(p: usize, d: usize, q: usize) -> SarimaOrder {
        SarimaOrder {
            p,
            d,
            q,
            P: 0,
            D: 0,
            Q: 0,
            m: 1,
        }
    }

    #[test]
    fn poly_mul_basic() {
        // (1 - B)(1 - B) = 1 - 2B + B^2
        let p = poly_mul(&[1.0, -1.0], &[1.0, -1.0]);
        assert_eq!(p, vec![1.0, -2.0, 1.0]);
    }

    #[test]
    fn random_walk_psi_are_ones() {
        // ARIMA(0,1,0): AR operator folded with (1-B) => psi_j == 1.
        let ar = ar_polynomial(&order(0, 1, 0), &[], &[], true);
        let ma = ma_polynomial(&order(0, 1, 0), &[], &[]);
        let psi = psi_weights(&ar, &ma, 5);
        for w in psi {
            assert!((w - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn ar1_psi_are_powers() {
        // AR(1) phi=0.5 => psi_j = 0.5^j.
        let ar = ar_polynomial(&order(1, 0, 0), &[0.5], &[], false);
        let ma = ma_polynomial(&order(1, 0, 0), &[], &[]);
        let psi = psi_weights(&ar, &ma, 6);
        for (j, w) in psi.iter().enumerate() {
            assert!((w - 0.5_f64.powi(j as i32)).abs() < 1e-12);
        }
    }

    #[test]
    fn stability_detects_explosive_ar() {
        let stable_ar = ar_polynomial(&order(1, 0, 0), &[0.9], &[], false);
        let ma = ma_polynomial(&order(1, 0, 0), &[], &[]);
        assert!(is_stable(&stable_ar, &ma));

        let explosive_ar = ar_polynomial(&order(1, 0, 0), &[1.05], &[], false);
        assert!(!is_stable(&explosive_ar, &ma));
    }
}
