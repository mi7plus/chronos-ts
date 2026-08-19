// src/viz.rs

use crate::decomposition::ProphetPrediction;
use crate::errors::{ChronosError, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartDataPoint {
    pub date: String,
    pub yhat: f64,
    pub yhat_lower: Option<f64>,
    pub yhat_upper: Option<f64>,
    pub trend: f64,
    pub seasonal: f64,
    pub holidays: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionExport {
    pub series: Vec<ChartDataPoint>,
}

pub struct VisualizationExporter;

impl VisualizationExporter {
    pub fn to_export_data(
        dates: &[NaiveDate],
        pred: &ProphetPrediction,
    ) -> Result<DecompositionExport> {
        let n = dates.len();
        if pred.yhat.len() != n {
            return Err(ChronosError::InvalidParameters(
                "Dates length does not match prediction length".into(),
            ));
        }

        let mut series = Vec::with_capacity(n);

        for i in 0..n {
            let yhat_lower = pred.yhat_lower.as_ref().map(|v| v[i]);
            let yhat_upper = pred.yhat_upper.as_ref().map(|v| v[i]);

            series.push(ChartDataPoint {
                date: dates[i].to_string(),
                yhat: pred.yhat[i],
                yhat_lower,
                yhat_upper,
                trend: pred.trend[i],
                seasonal: pred.seasonal[i],
                holidays: pred.holidays[i],
            });
        }

        Ok(DecompositionExport { series })
    }

    pub fn to_json(dates: &[NaiveDate], pred: &ProphetPrediction) -> Result<String> {
        let export_data = Self::to_export_data(dates, pred)?;
        Ok(serde_json::to_string_pretty(&export_data)?)
    }

    pub fn to_html_string(
        dates: &[NaiveDate],
        pred: &ProphetPrediction,
        title: &str,
    ) -> Result<String> {
        let json_data = Self::to_json(dates, pred)?;

        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #f8fafc; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .chart-card {{ background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
        #forecast-chart, #components-chart {{ width: 100%; height: 450px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="chart-card"><div id="forecast-chart"></div></div>
        <div class="chart-card"><div id="components-chart"></div></div>
    </div>
    <script>
        const payload = {json_data};
        const dates = payload.series.map(d => d.date);
        const yhat = payload.series.map(d => d.yhat);
        const trend = payload.series.map(d => d.trend);
        const seasonal = payload.series.map(d => d.seasonal);
        const holidays = payload.series.map(d => d.holidays);

        const hasUpper = payload.series[0] && payload.series[0].yhat_upper !== null;
        const forecastTraces = [];

        if (hasUpper) {{
            const yhatUpper = payload.series.map(d => d.yhat_upper);
            const yhatLower = payload.series.map(d => d.yhat_lower);
            forecastTraces.push({{
                x: dates.concat(dates.slice().reverse()),
                y: yhatUpper.concat(yhatLower.slice().reverse()),
                fill: 'tozerox',
                fillcolor: 'rgba(59, 130, 246, 0.2)',
                line: {{ color: 'transparent' }},
                name: 'Uncertainty Interval',
                type: 'scatter'
            }});
        }}

        forecastTraces.push({{ x: dates, y: yhat, mode: 'lines', name: 'Forecast (yhat)', line: {{ color: '#3b82f6', width: 2.5 }} }});
        Plotly.newPlot('forecast-chart', forecastTraces, {{ title: 'Forecast', paper_bgcolor: '#1e293b', plot_bgcolor: '#1e293b', font: {{ color: '#f8fafc' }} }});

        const componentTraces = [
            {{ x: dates, y: trend, mode: 'lines', name: 'Trend', line: {{ color: '#10b981' }} }},
            {{ x: dates, y: seasonal, mode: 'lines', name: 'Seasonality', line: {{ color: '#f59e0b' }} }},
            {{ x: dates, y: holidays, mode: 'lines', name: 'Holidays', line: {{ color: '#ec4899' }} }}
        ];
        Plotly.newPlot('components-chart', componentTraces, {{ title: 'Components', paper_bgcolor: '#1e293b', plot_bgcolor: '#1e293b', font: {{ color: '#f8fafc' }} }});
    </script>
</body>
</html>"#,
            title = title,
            json_data = json_data
        );

        Ok(html)
    }

    pub fn save_html<P: AsRef<Path>>(
        dates: &[NaiveDate],
        pred: &ProphetPrediction,
        title: &str,
        path: P,
    ) -> Result<()> {
        let html_content = Self::to_html_string(dates, pred, title)?;
        let mut file = File::create(path)?;
        file.write_all(html_content.as_bytes())?;
        Ok(())
    }
}