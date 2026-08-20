use chrono::NaiveDate;
use chronos_ts::{auto_arima, AutoArimaOptions, Holiday, ProphetDecomposition};
use ndarray::Array1;

#[test]
fn test_constant_series() {
    let series = Array1::from_elem(50, 10.0);
    let opts = AutoArimaOptions {
        max_p: 1,
        max_d: 1,
        max_q: 1,
        ..Default::default()
    };

    // Should return Ok or an explicit Err without panicking
    let res = auto_arima(&series, opts);
    assert!(
        res.is_ok() || res.is_err(),
        "Must handle constant series gracefully without panicking"
    );
}

#[test]
fn test_small_sample_size() {
    // Minimal viable sample for ARIMA order estimation is typically n >= 8
    let series = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let opts = AutoArimaOptions {
        max_p: 1,
        max_d: 1,
        max_q: 1,
        ..Default::default()
    };

    let model = auto_arima(&series, opts);
    assert!(
        model.is_ok(),
        "Model fitting should work on 8+ observations"
    );
}

#[test]
fn test_prophet_decomposition_edge_cases() {
    let series = Array1::from_elem(30, 5.0);

    // Generate dates matching the length of the series
    let start_date = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..30)
        .map(|i| start_date + chrono::Duration::days(i))
        .collect();

    let mut decomp = ProphetDecomposition::new(2, 0.05);
    let res = decomp.fit(&dates, &series, None, None);

    assert!(
        res.is_ok(),
        "Prophet decomposition should handle uniform series"
    );
}

#[test]
fn test_auto_arima_fit() {
    // Generate simple AR(1) target series
    let mut data = Vec::new();
    let mut val = 10.0;
    for i in 0..100 {
        val = 0.7 * val + (i as f64 % 3.0) * 0.1;
        data.push(val);
    }
    let series = Array1::from(data);

    // With this:
    let opts = AutoArimaOptions {
        max_p: 2,
        max_d: 1,
        max_q: 2,
        ..Default::default()
    };
    let model = auto_arima(&series, opts).expect("Auto ARIMA model should fit");
    assert!(model.order.p <= 2);

    let forecast = model.forecast(&series, 5);
    assert_eq!(forecast.len(), 5);
}

#[test]
fn test_auto_arima_pipeline() {
    let series = Array1::from_vec(vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]);

    let opts = AutoArimaOptions {
        max_p: 2,
        max_d: 1,
        max_q: 2,
        ..Default::default() // Populates criterion, stepwise, and other defaults
    };

    let model = auto_arima(&series, opts).expect("Auto ARIMA model should fit");
    let forecast_result = model.forecast(&series, 3);
    assert_eq!(forecast_result.len(), 3);
}

#[test]
fn test_prophet_decomposition() {
    let mut decomp = ProphetDecomposition::new(2, 0.05);
    let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
    let dates: Vec<NaiveDate> = (0..30).map(|i| start + chrono::Duration::days(i)).collect();

    let y = Array1::from_vec((0..30).map(|i| (i as f64) * 0.5 + 2.0).collect());

    decomp.add_holiday(Holiday {
        name: "NewYear".to_string(),
        dates: vec![start],
        lower_window: 0,
        upper_window: 1,
    });

    decomp.fit(&dates, &y, None, None).unwrap();
    let preds = decomp.predict(&dates).unwrap();

    assert_eq!(preds.yhat.len(), 30);
}
