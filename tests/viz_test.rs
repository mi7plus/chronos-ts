use chrono::NaiveDate;
use chronos_ts::{ProphetDecomposition, VisualizationExporter};
use ndarray::Array1;

#[test]
fn test_visualization_export() {
    let start_date = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
    let n = 30;
    let mut dates = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    for i in 0..n {
        let date = start_date + chrono::Duration::days(i as i64);
        dates.push(date);
        values.push(10.0 + (i as f64) * 0.5);
    }

    let y = Array1::from_vec(values);
    let mut model = ProphetDecomposition::new(10, 0.05);
    model.fit(&dates, &y, None, None).unwrap();

    let pred = model.predict_with_intervals(&dates, 0.95, 50).unwrap();

    // Verify JSON export
    let json_res = VisualizationExporter::to_json(&dates, &pred).unwrap();
    assert!(json_res.contains("yhat_upper"));
    assert!(json_res.contains("yhat_lower"));

    // Verify HTML export
    let html_res =
        VisualizationExporter::to_html_string(&dates, &pred, "Chronos Forecast").unwrap();
    assert!(html_res.contains("<!DOCTYPE html>"));
    assert!(html_res.contains("Plotly.newPlot"));
}
