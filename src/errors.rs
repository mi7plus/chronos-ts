use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChronosError {
    #[error("Insufficient data: required {required}, found {found}")]
    InsufficientData { required: usize, found: usize },

    #[error("Invalid parameter: {0}")]
    InvalidParameters(String),

    #[error("Optimization failed: {0}")]
    OptimizationError(String),

    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),

    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] ndarray_linalg::error::LinalgError),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("{0}")]
    Custom(String),
}

impl From<&str> for ChronosError {
    fn from(msg: &str) -> Self {
        ChronosError::Custom(msg.to_string())
    }
}

impl From<String> for ChronosError {
    fn from(msg: String) -> Self {
        ChronosError::Custom(msg)
    }
}

pub type Result<T> = std::result::Result<T, ChronosError>;