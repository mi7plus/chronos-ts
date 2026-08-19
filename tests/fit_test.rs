use chrono::{Duration, NaiveDate};
use chronos_ts::{ProphetDecomposition, SeasonalityMode, TrendType};
use ndarray::Array1;

#[test]
fn test_prophet_fit_and_predict_future() {
    // 1. Generate 100 days of synthetic time series data
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let total_days = 100;
    let train_days = 80;

    let dates: Vec<NaiveDate> = (0..total_days)
        .map(|i| start_date + Duration::days(i))
        .collect();

    // Ground truth: Intercept (10.0) + Linear Trend (0.1 * t) + Weekly Seasonality (2.0 * sin(2pi * t / 7))
    let y_all: Vec<f64> = (0..total_days)
        .map(|i| {
            let day = i as f64;
            let trend = 10.0 + 0.1 * day;
            let seasonality = 2.0 * (2.0 * std::f64::consts::PI * day / 7.0).sin();
            trend + seasonality
        })
        .collect();

    let train_dates = &dates[..train_days];
    let train_y = Array1::from_vec(y_all[..train_days].to_vec());

    let test_dates = &dates[train_days..];
    let actual_future_y = Array1::from_vec(y_all[train_days..].to_vec());

    // 2. Configure model with 7-day Fourier seasonality
    let mut model = ProphetDecomposition::new(5, 0.05);
    model.add_seasonality("weekly", 7.0, 3);

    // 3. Fit model on 80 training days
    model.fit(train_dates, &train_y, None, None).expect("Model fit failed");

    // 4. Predict 20 out-of-sample future days
    let prediction = model.predict(test_dates).expect("Prediction failed");

    assert_eq!(prediction.yhat.len(), test_dates.len());
    assert_eq!(prediction.trend.len(), test_dates.len());
    assert_eq!(prediction.seasonal.len(), test_dates.len());

    // Verify named seasonality breakdown
    let weekly = prediction
        .seasonalities
        .get("weekly")
        .expect("Missing 'weekly' seasonality in breakdown");
    assert_eq!(weekly.len(), test_dates.len());

    // 5. Assert Mean Absolute Error (MAE) on total forecast yhat
    let errors: Array1<f64> = &prediction.yhat - &actual_future_y;
    let mae = errors.mapv(|e| e.abs()).mean().unwrap();

    println!("Future Prediction MAE: {:.4}", mae);

    assert!(
        mae < 0.25,
        "Future prediction MAE ({:.4}) exceeded maximum tolerance of 0.25",
        mae
    );
}

/// Unit Test: Verifies that predict() applies the multiplicative formula correctly.
#[test]
fn test_multiplicative_prediction_math() {
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..10).map(|i| start_date + Duration::days(i)).collect();
    let train_y = Array1::from_vec(vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]);

    let mut model = ProphetDecomposition::new(2, 0.05);
    model.seasonality_mode = SeasonalityMode::Multiplicative;
    model.add_seasonality("weekly", 7.0, 2);

    model.fit(&dates, &train_y, None, None).expect("Model fit failed");
    let pred = model.predict(&dates).expect("Prediction failed");

    // Check row-by-row that yhat == trend * (1.0 + seasonal + holidays)
    for i in 0..dates.len() {
        let expected_yhat = pred.trend[i] * (1.0 + pred.seasonal[i] + pred.holidays[i]);
        let diff = (pred.yhat[i] - expected_yhat).abs();
        assert!(
            diff < 1e-6,
            "Multiplicative math failed at index {}: expected {}, got {}",
            i, expected_yhat, pred.yhat[i]
        );
    }
}

/// Integration Test: Fits a synthetic dataset with expanding seasonal amplitude.
#[test]
fn test_multiplicative_fit_accuracy() {
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let total_days = 100;
    let train_days = 80;

    let dates: Vec<NaiveDate> = (0..total_days)
        .map(|i| start_date + Duration::days(i))
        .collect();

    // Ground truth multiplicative model: y(t) = trend(t) * (1 + 0.1 * sin(...))
    let y_all: Vec<f64> = (0..total_days)
        .map(|i| {
            let day = i as f64;
            let trend = 100.0 + 2.0 * day;
            let seasonal_ratio = 0.10 * (2.0 * std::f64::consts::PI * day / 7.0).sin();
            trend * (1.0 + seasonal_ratio)
        })
        .collect();

    let train_dates = &dates[..train_days];
    let train_y = Array1::from_vec(y_all[..train_days].to_vec());
    let test_dates = &dates[train_days..];
    let actual_future_y = Array1::from_vec(y_all[train_days..].to_vec());

    let mut mult_model = ProphetDecomposition::new(5, 0.05);
    mult_model.seasonality_mode = SeasonalityMode::Multiplicative;
    mult_model.add_seasonality("weekly", 7.0, 3);

    mult_model.fit(train_dates, &train_y, None, None).expect("Multiplicative fit failed");
    let mult_pred = mult_model.predict(test_dates).expect("Multiplicative predict failed");

    let errors = &mult_pred.yhat - &actual_future_y;
    let mult_mae = errors.mapv(|e| e.abs()).mean().unwrap();

    println!("Multiplicative Seasonality MAE: {:.4}", mult_mae);

    assert!(
        mult_mae < 5.0,
        "Multiplicative prediction MAE ({:.4}) exceeded tolerance",
        mult_mae
    );
}

#[test]
fn test_logistic_growth_trend() {
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..50).map(|i| start_date + Duration::days(i)).collect();

    let cap = Array1::from_elem(50, 100.0);
    let y = Array1::from_vec(
        (0..50).map(|i| 100.0 / (1.0 + (-0.1 * (i as f64 - 25.0)).exp())).collect()
    );

    let mut model = ProphetDecomposition::new(3, 0.05);
    model.trend_type = TrendType::Logistic;

    model.fit(&dates, &y, Some(&cap), None).expect("Logistic fit failed");
    let pred = model.predict(&dates).expect("Logistic predict failed");

    for &y_hat in pred.yhat.iter() {
        assert!(y_hat <= 100.0);
    }
}

#[test]
fn test_holiday_prior_scale_regularization() {
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..30).map(|i| start_date + Duration::days(i)).collect();
    let y = Array1::from_elem(30, 100.0);

    let holiday = chronos_ts::decomposition::Holiday {
        name: "New Year".to_string(),
        dates: vec![start_date],
        lower_window: 0,
        upper_window: 0,
    };

    let mut model = ProphetDecomposition::new(2, 0.05);
    model.add_holiday(holiday);

    // Heavily penalize holiday prior scale to force beta to zero
    model.holiday_prior_scale = 1e-4;

    model.fit(&dates, &y, None, None).expect("Fit failed");
    let pred = model.predict(&dates).expect("Predict failed");

    // Verify holiday component is strongly regularized
    for &h_effect in pred.holidays.iter() {
        assert!(h_effect.abs() < 1e-3, "Holiday effect was not regularized properly: {}", h_effect);
    }
}

#[test]
fn test_model_serialization_roundtrip() {
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..30).map(|i| start_date + Duration::days(i)).collect();
    let y = Array1::from_vec((0..30).map(|i| 10.0 + 0.5 * (i as f64)).collect());

    // 1. Train original model
    let mut original_model = ProphetDecomposition::new(3, 0.05);
    original_model.add_seasonality("weekly", 7.0, 3);
    original_model
        .fit(&dates, &y, None, None)
        .expect("Fit failed");

    let orig_pred = original_model.predict(&dates).expect("Predict failed");

    // 2. Round-trip through JSON
    let json_str = original_model.to_json().expect("Serialization failed");
    let loaded_model = ProphetDecomposition::from_json(&json_str).expect("Deserialization failed");

    // 3. Predict with loaded model and compare results
    let loaded_pred = loaded_model.predict(&dates).expect("Predict failed");

    let diff = (&orig_pred.yhat - &loaded_pred.yhat)
        .mapv(|e| e.abs())
        .sum();

    assert!(diff < 1e-12, "Deserialized model predictions deviated: {}", diff);
}

#[test]
fn test_independent_seasonality_prior_scales() {
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..60).map(|i| start_date + Duration::days(i)).collect();

    // Signal with strong weekly component
    let y = Array1::from_vec(
        (0..60)
            .map(|i| 100.0 + 10.0 * (2.0 * std::f64::consts::PI * (i as f64) / 7.0).sin())
            .collect(),
    );

    // Model with two seasonalities: one heavily penalized (1e-4), one flexible (10.0)
    let mut model = ProphetDecomposition::new(2, 0.05);
    model.add_seasonality_with_prior("restricted_weekly", 7.0, 3, 1e-4);
    model.add_seasonality_with_prior("free_weekly", 7.0, 3, 10.0);

    model.fit(&dates, &y, None, None).expect("Fit failed");
    let pred = model.predict(&dates).expect("Predict failed");

    // Use .iter() and fold to get maximum absolute amplitude safely on f64 arrays
    let restricted_amplitude = pred
        .seasonalities
        .get("restricted_weekly")
        .unwrap()
        .iter()
        .fold(0.0f64, |acc, &v| acc.max(v.abs()));

    let free_amplitude = pred
        .seasonalities
        .get("free_weekly")
        .unwrap()
        .iter()
        .fold(0.0f64, |acc, &v| acc.max(v.abs()));

    assert!(
        restricted_amplitude < 1e-2,
        "Restricted seasonality was not heavily suppressed: {}",
        restricted_amplitude
    );
    assert!(
        free_amplitude > 1.0,
        "Free seasonality failed to capture signal: {}",
        free_amplitude
    );
}

#[test]
fn test_uncertainty_interval_generation() {
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..30).map(|i| start_date + Duration::days(i)).collect();
    let y = Array1::from_vec((0..30).map(|i| 10.0 + 0.5 * (i as f64)).collect());

    let mut model = ProphetDecomposition::new(3, 0.05);
    model.fit(&dates, &y, None, None).expect("Fit failed");

    let pred = model
        .predict_with_intervals(&dates, 0.95, 200)
        .expect("Interval prediction failed");

    let yhat_lower = pred.yhat_lower.unwrap();
    let yhat_upper = pred.yhat_upper.unwrap();

    for i in 0..30 {
        assert!(
            yhat_lower[i] <= pred.yhat[i],
            "Lower bound should be <= yhat at index {}",
            i
        );
        assert!(
            pred.yhat[i] <= yhat_upper[i],
            "Upper bound should be >= yhat at index {}",
            i
        );
    }
}
