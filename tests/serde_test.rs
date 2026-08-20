use chronos_ts::arima::{SarimaModel, SarimaOrder};
use chronos_ts::decomposition::{ProphetDecomposition, SeasonalitySpec, TrendType};
use ndarray::array;

fn create_dummy_model(exog_beta: Option<ndarray::Array1<f64>>) -> SarimaModel {
    SarimaModel {
        order: SarimaOrder {
            p: 1,
            d: 0,
            q: 1,
            P: 0,
            D: 0,
            Q: 0,
            m: 1, // Fixed: Changed 's' to 'm'
        },
        ar_coeffs: array![0.5],
        ma_coeffs: array![-0.2],
        sar_coeffs: array![],
        sma_coeffs: array![],
        sigma2: 1.25,
        log_likelihood: -42.5,
        intercept: 0.0,
        std_errors: None,
        transform: None,
        exog_beta,
    }
}

#[test]
fn test_serde_without_exog_beta() {
    let model = create_dummy_model(None);

    let json_data = serde_json::to_string(&model).expect("Serialization failed");
    let deserialized: SarimaModel =
        serde_json::from_str(&json_data).expect("Deserialization failed");

    assert_eq!(deserialized.order.p, model.order.p);
    assert_eq!(deserialized.ar_coeffs, model.ar_coeffs);
    assert_eq!(deserialized.sigma2, model.sigma2);
    assert!(deserialized.exog_beta.is_none());
}

#[test]
fn test_serde_with_exog_beta() {
    let beta_values = array![1.5, -0.8, 3.2];
    let model = create_dummy_model(Some(beta_values.clone()));

    let json_data = serde_json::to_string(&model).expect("Serialization failed");
    let deserialized: SarimaModel =
        serde_json::from_str(&json_data).expect("Deserialization failed");

    assert_eq!(deserialized.order.p, model.order.p);
    assert_eq!(deserialized.ar_coeffs, model.ar_coeffs);
    assert_eq!(deserialized.sigma2, model.sigma2);

    let deserialized_beta = deserialized
        .exog_beta
        .expect("exog_beta should be present after deserialization");
    assert_eq!(deserialized_beta, beta_values);
}

#[test]
fn test_serde_backwards_compatibility() {
    let legacy_json = r#"{
        "order": {"p": 1, "d": 0, "q": 1, "P": 0, "D": 0, "Q": 0, "m": 1},
        "ar_coeffs": [0.5],
        "ma_coeffs": [-0.2],
        "sar_coeffs": [],
        "sma_coeffs": [],
        "sigma2": 1.25,
        "log_likelihood": -42.5
    }"#;

    let deserialized: SarimaModel =
        serde_json::from_str(legacy_json).expect("Deserialization of legacy model failed");

    assert!(deserialized.exog_beta.is_none());
    assert_eq!(deserialized.ar_coeffs, array![0.5]);
}

#[test]
fn test_prophet_decomposition_serde() {
    let original = ProphetDecomposition {
        trend_type: TrendType::Linear,
        seasonalities: vec![SeasonalitySpec {
            name: "yearly".to_string(),
            fourier_order: 3,
            period_days: 365.25,
            prior_scale: 10.0,
        }],
        ..Default::default() // Automatically fills in beta, capacities, changepoint_prior_scale, etc.
    };

    // Serialize to JSON string
    let json = serde_json::to_string_pretty(&original).expect("Failed to serialize");
    println!("Serialized JSON:\n{}", json);

    // Deserialize back into struct
    let deserialized: ProphetDecomposition =
        serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(
        original.seasonalities.len(),
        deserialized.seasonalities.len()
    );
}
