//! Accuracy tests for the ARIMA estimator. Unlike the smoke tests in
//! `integration_test.rs`, these assert that fitting actually recovers known
//! parameters and produces sensible forecasts (regression guard against the
//! historical "coefficients left at zero" bug).

use chronos_ts::arima::{
    arima_cross_validation, auto_arima, AutoArimaOptions, EstimationMethod, SarimaModel,
    SarimaOrder,
};
use chronos_ts::diagnostics::ResidualDiagnostics;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Simulates a zero-mean AR(1) process: x[t] = phi * x[t-1] + eps.
fn simulate_ar1(phi: f64, n: usize, seed: u64) -> Array1<f64> {
    simulate_ar1_mean(phi, 0.0, n, seed)
}

/// Simulates an AR(1) process around a nonzero mean: x[t] = mu + phi*(x[t-1]-mu) + eps.
fn simulate_ar1_mean(phi: f64, mu: f64, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let mut x = mu;
    let mut out = Vec::with_capacity(n);
    for _ in 0..200 {
        x = mu + phi * (x - mu) + noise.sample(&mut rng);
    }
    for _ in 0..n {
        x = mu + phi * (x - mu) + noise.sample(&mut rng);
        out.push(x);
    }
    Array1::from(out)
}

#[test]
fn css_recovers_ar1_coefficient() {
    let phi = 0.7;
    let series = simulate_ar1(phi, 500, 42);

    let model = SarimaModel::fit(&series, SarimaOrder::arima(1, 0, 0)).expect("fit failed");

    let est = model.ar_coeffs[0];
    println!("AR(1): true phi = {phi}, estimated = {est:.4}");

    assert!(
        (est - phi).abs() < 0.1,
        "CSS failed to recover AR(1) coefficient: expected ~{phi}, got {est:.4}"
    );
}

#[test]
fn ar1_forecast_decays_toward_mean() {
    let phi = 0.7;
    let series = simulate_ar1(phi, 500, 7);
    let model = SarimaModel::fit(&series, SarimaOrder::arima(1, 0, 0)).expect("fit failed");

    let fc = model.forecast(&series, 20);
    assert_eq!(fc.len(), 20);

    // A stationary zero-mean AR(1) forecast must decay in magnitude toward 0.
    assert!(
        fc[19].abs() <= fc[0].abs() + 1e-9,
        "AR(1) forecast should decay: first = {}, last = {}",
        fc[0],
        fc[19]
    );
    // And each step should be phi times the previous (geometric decay).
    let ratio = fc[1] / fc[0];
    assert!(
        (ratio - phi).abs() < 0.1,
        "forecast decay ratio {ratio:.4} should be close to phi {phi}"
    );
}

#[test]
fn css_beats_zero_coefficients_in_sample() {
    // The old stub left coefficients at zero, giving SSE = sum of squares of the
    // series. A real fit must reduce residual variance well below that.
    let phi = 0.8;
    let series = simulate_ar1(phi, 400, 99);

    let fitted = SarimaModel::fit(&series, SarimaOrder::arima(1, 0, 0)).expect("fit failed");
    let total_var: f64 = series.iter().map(|v| v * v).sum::<f64>() / series.len() as f64;

    println!(
        "sigma2 = {:.4}, series 2nd moment = {:.4}",
        fitted.sigma2, total_var
    );
    assert!(
        fitted.sigma2 < 0.85 * total_var,
        "fitted residual variance ({:.4}) should be well below the series variance ({:.4})",
        fitted.sigma2,
        total_var
    );
}

#[test]
fn auto_arima_selects_ar_component_for_ar_data() {
    let series = simulate_ar1(0.75, 300, 123);
    let opts = AutoArimaOptions {
        max_p: 3,
        max_d: 1,
        max_q: 3,
        ..Default::default()
    };
    let model = auto_arima(&series, opts).expect("auto_arima failed");
    // Data is stationary AR(1): differencing should not be required.
    assert_eq!(
        model.order.d, 0,
        "auto_arima over-differenced stationary data"
    );
    // Some autoregressive/moving-average structure should be picked up.
    assert!(
        model.order.p + model.order.q >= 1,
        "auto_arima selected a white-noise model for autocorrelated data"
    );
}

#[test]
fn sarimax_recovers_exog_effect_and_forecasts() {
    // y_t = 5 * x_t + AR(1) noise.
    let n = 300;
    let steps = 10;
    let mut rng = StdRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let mut eta = 0.0;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for t in 0..n {
        eta = 0.5 * eta + noise.sample(&mut rng);
        let xt = (t as f64 * 0.1).sin();
        x.push(xt);
        y.push(5.0 * xt + eta);
    }
    let y_arr = Array1::from(y);
    let x_hist = Array2::from_shape_vec((n, 1), x).unwrap();

    let model = SarimaModel::fit_with_exog(&y_arr, Some(&x_hist), SarimaOrder::arima(1, 0, 0))
        .expect("SARIMAX fit failed");

    let beta = model.exog_beta.as_ref().expect("exog_beta missing");
    println!("SARIMAX exog beta = {:.4} (true 5.0)", beta[0]);
    assert!(
        (beta[0] - 5.0).abs() < 0.5,
        "exogenous coefficient {:.4} should be close to 5.0",
        beta[0]
    );

    let x_fut_vec: Vec<f64> = (n..n + steps).map(|t| (t as f64 * 0.1).sin()).collect();
    let x_fut = Array2::from_shape_vec((steps, 1), x_fut_vec.clone()).unwrap();

    let fc = model
        .forecast_with_intervals_exog(&y_arr, Some(&x_hist), Some(&x_fut), steps)
        .expect("SARIMAX forecast failed");

    // Point forecast should track 5 * x_fut (the noise has zero mean).
    for (i, (&fm, &xf)) in fc.mean.iter().zip(x_fut_vec.iter()).enumerate() {
        let expected = 5.0 * xf;
        assert!(
            (fm - expected).abs() < 3.0,
            "step {i}: forecast {fm:.3} far from exog signal {expected:.3}"
        );
    }

    // Forecasting without the required historical exog matrix must error, not lie.
    assert!(model
        .forecast_with_intervals_exog(&y_arr, None, Some(&x_fut), steps)
        .is_err());
}

/// Simulates an MA(1) process: x[t] = eps[t] + theta * eps[t-1].
fn simulate_ma1(theta: f64, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let mut prev = 0.0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let cur = noise.sample(&mut rng);
        out.push(cur + theta * prev);
        prev = cur;
    }
    Array1::from(out)
}

#[test]
fn exact_mle_recovers_ar_and_ma_coefficients() {
    // AR(1)
    let ar_series = simulate_ar1_mean(0.7, 20.0, 400, 1);
    let mle_ar = SarimaModel::fit_with_method(
        &ar_series,
        SarimaOrder::arima(1, 0, 0),
        EstimationMethod::Mle,
    )
    .expect("MLE AR fit failed");
    println!(
        "MLE AR(1): ar={:.4}, intercept={:.3}, ll={:.2}",
        mle_ar.ar_coeffs[0], mle_ar.intercept, mle_ar.log_likelihood
    );
    assert!(mle_ar.log_likelihood.is_finite());
    assert!(
        (mle_ar.ar_coeffs[0] - 0.7).abs() < 0.1,
        "MLE AR coefficient {:.4} should be close to 0.7",
        mle_ar.ar_coeffs[0]
    );
    assert!((mle_ar.intercept - 20.0).abs() < 2.0);

    // MA(1)
    let ma_series = simulate_ma1(0.5, 400, 2);
    let mle_ma = SarimaModel::fit_with_method(
        &ma_series,
        SarimaOrder::arima(0, 0, 1),
        EstimationMethod::Mle,
    )
    .expect("MLE MA fit failed");
    println!(
        "MLE MA(1): ma={:.4}, ll={:.2}",
        mle_ma.ma_coeffs[0], mle_ma.log_likelihood
    );
    assert!(
        (mle_ma.ma_coeffs[0] - 0.5).abs() < 0.15,
        "MLE MA coefficient {:.4} should be close to 0.5",
        mle_ma.ma_coeffs[0]
    );
}

#[test]
fn auto_arima_mle_matches_ar_data() {
    let series = simulate_ar1_mean(0.75, 5.0, 300, 44);
    let opts = AutoArimaOptions {
        max_p: 3,
        max_d: 1,
        max_q: 3,
        estimation: EstimationMethod::Mle,
        ..Default::default()
    };
    let model = auto_arima(&series, opts).expect("auto_arima (MLE) failed");
    assert_eq!(model.order.d, 0);
    assert!(model.order.p + model.order.q >= 1);
    // Forecast should be finite and near the series mean far out.
    let fc = model.forecast(&series, 30);
    assert!(fc.iter().all(|v| v.is_finite()));
    assert!((fc[29] - 5.0).abs() < 5.0);
}

#[test]
fn ar1_with_nonzero_mean_forecasts_toward_mean() {
    // A stationary AR(1) around mean 50 must forecast toward 50, not toward 0.
    let mu = 50.0;
    let series = simulate_ar1_mean(0.6, mu, 500, 5);
    let model = SarimaModel::fit(&series, SarimaOrder::arima(1, 0, 0)).expect("fit failed");

    let fc = model.forecast(&series, 40);
    println!(
        "nonzero-mean AR(1): intercept={:.3}, long-horizon forecast={:.3}",
        model.intercept, fc[39]
    );
    assert!(
        (fc[39] - mu).abs() < 5.0,
        "long-horizon forecast {:.3} should converge to the mean {mu}",
        fc[39]
    );
}

#[test]
fn arima_011_recovers_drift_and_extrapolates() {
    // Random walk with drift 2.0 -> ARIMA(0,1,0) should learn the drift and
    // extrapolate a straight line (exercises the d>=1 integration path).
    let n = 300;
    let drift = 2.0;
    let mut rng = StdRng::seed_from_u64(3);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let mut x = 0.0;
    let mut series = Vec::with_capacity(n);
    for _ in 0..n {
        x += drift + noise.sample(&mut rng);
        series.push(x);
    }
    let s = Array1::from(series);

    let model = SarimaModel::fit(&s, SarimaOrder::arima(0, 1, 0)).expect("fit failed");
    assert!(
        (model.intercept - drift).abs() < 0.5,
        "estimated drift {:.3} should be close to {drift}",
        model.intercept
    );

    let fc = model.forecast(&s, 10);
    let last = s[n - 1];
    assert!((fc[0] - (last + drift)).abs() < 1.5, "first step off");
    assert!(
        (fc[9] - (last + 10.0 * drift)).abs() < 5.0,
        "10-step extrapolation off: {:.3}",
        fc[9]
    );
}

#[test]
fn seasonal_forecast_reproduces_pattern() {
    // Exact seasonal pattern + linear trend, modelled as (0,0,0)(0,1,0)_4.
    // Seasonal differencing collapses it to a constant drift of m * trend_slope,
    // and the forecast must integrate back to the original seasonal scale.
    let m = 4;
    let pattern = [10.0, 14.0, 8.0, 12.0];
    let slope = 0.5;
    let n = 80;
    let series: Vec<f64> = (0..n).map(|t| pattern[t % m] + slope * t as f64).collect();
    let s = Array1::from(series);

    let order = SarimaOrder {
        p: 0,
        d: 0,
        q: 0,
        P: 0,
        D: 1,
        Q: 0,
        m,
    };
    let model = SarimaModel::fit(&s, order).expect("seasonal fit failed");

    let fc = model.forecast(&s, m);
    for (i, &f) in fc.iter().enumerate() {
        let expected = pattern[(n + i) % m] + slope * (n + i) as f64;
        assert!(
            (f - expected).abs() < 0.5,
            "seasonal step {i}: forecast {f:.3} vs expected {expected:.3}"
        );
    }
}

#[test]
fn prediction_intervals_widen_with_horizon() {
    let series = simulate_ar1(0.5, 300, 21);
    let model = SarimaModel::fit(&series, SarimaOrder::arima(1, 0, 0)).expect("fit failed");
    let fc = model.forecast_with_intervals(&series, 15);

    // Bands must bracket the mean and be non-decreasing in width with the horizon.
    let width = |h: usize| fc.upper_95[h] - fc.lower_95[h];
    for h in 0..15 {
        assert!(fc.lower_95[h] <= fc.mean[h] && fc.mean[h] <= fc.upper_95[h]);
    }
    assert!(
        width(14) > width(0),
        "95% interval should widen with horizon: {:.3} -> {:.3}",
        width(0),
        width(14)
    );
    // Stationary AR(1) forecast SE converges, so growth must be sub-linear
    // (a random-walk sigma*sqrt(h) would give width(14) ~ sqrt(15) * width(0)).
    assert!(
        width(14) < 3.0 * width(0),
        "stationary AR(1) interval should saturate, not grow like a random walk"
    );
}

#[test]
fn residuals_of_correct_model_are_approximately_white() {
    // A correctly specified AR(1) fit should leave near-white residuals: the
    // Ljung-Box test should NOT reject whiteness at the 5% level.
    let series = simulate_ar1(0.6, 500, 314);
    let model = SarimaModel::fit(&series, SarimaOrder::arima(1, 0, 0)).expect("fit failed");

    let residuals = model.residuals(&series);
    assert_eq!(residuals.len(), series.len());

    let lb = ResidualDiagnostics::ljung_box(&residuals, 10).expect("ljung-box failed");
    println!(
        "Ljung-Box Q={:.3}, p={:.3} on AR(1) residuals",
        lb.q_stat, lb.p_value
    );
    assert!(
        lb.p_value > 0.05,
        "residuals of a correctly specified model should look white (p={:.3})",
        lb.p_value
    );
}

#[test]
fn coefficient_standard_errors_are_reported_and_reasonable() {
    let series = simulate_ar1(0.7, 500, 271);
    let model = SarimaModel::fit(&series, SarimaOrder::arima(1, 0, 0)).expect("fit failed");

    let se = model
        .std_errors
        .as_ref()
        .expect("std errors should be present");
    assert_eq!(se.len(), 1);
    // Asymptotic SE of an AR(1) coefficient is ~ sqrt((1 - phi^2)/n).
    let expected = ((1.0_f64 - 0.7 * 0.7) / 500.0).sqrt();
    println!(
        "AR(1) coeff SE: got {:.4}, expected ~{:.4}",
        se[0], expected
    );
    assert!(se[0] > 0.0 && se[0].is_finite());
    assert!(
        (se[0] - expected).abs() < 0.02,
        "standard error {:.4} not close to asymptotic value {:.4}",
        se[0],
        expected
    );
}

#[test]
fn box_cox_transform_forecasts_on_original_scale() {
    // Exponential-growth series: fitting on the log scale then back-transforming
    // should keep forecasts positive and near the true continuation.
    let n = 120;
    let series = Array1::from_shape_fn(n, |i| (0.03 * i as f64).exp() * 10.0);

    let model = SarimaModel::fit_transformed(
        &series,
        SarimaOrder::arima(1, 1, 0),
        EstimationMethod::Css,
        Some(0.0), // log transform
    )
    .expect("transformed fit failed");
    assert_eq!(model.transform, Some(0.0));

    let fc = model.forecast(&series, 10);
    assert!(fc.iter().all(|v| v.is_finite() && *v > 0.0));
    // Series is increasing; the first forecast should exceed the last observation.
    assert!(fc[0] > series[n - 1] * 0.98);
}

#[test]
fn fit_rejects_non_finite_input() {
    let bad = Array1::from(vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    assert!(SarimaModel::fit(&bad, SarimaOrder::arima(1, 0, 0)).is_err());
    assert!(auto_arima(&bad, AutoArimaOptions::default()).is_err());

    let empty = Array1::<f64>::zeros(0);
    assert!(auto_arima(&empty, AutoArimaOptions::default()).is_err());
}

#[test]
fn arima_cross_validation_reports_sane_metrics() {
    let series = simulate_ar1_mean(0.6, 30.0, 200, 88);
    let report = arima_cross_validation(
        &series,
        SarimaOrder::arima(1, 0, 0),
        EstimationMethod::Css,
        120,
        5,
        10,
    )
    .expect("cross-validation failed");

    println!(
        "ARIMA CV: MAE={:.3}, RMSE={:.3}, windows={}",
        report.mae, report.rmse, report.n_windows
    );
    assert!(report.n_windows >= 3);
    assert!(report.mae > 0.0 && report.mae.is_finite());
    assert!(report.rmse >= report.mae); // RMSE >= MAE always holds
}
