// use localization::{
//     extended_kalman_filter::{ExtendedKalmanFilter, ExtendedKalmanFilterKnownCorrespondences},
//     particle_filter::ParticleFilterKnownCorrespondences,
//     unscented_kalman_filter::UnscentedKalmanFilter,
// };
// use nalgebra::Dyn;
use pyo3::prelude::*;

pub mod data;
pub mod localization;
pub mod mapping;
pub mod models;
pub mod utils;

// #[pyclass]
// struct EKF {
//     ekf: ExtendedKalmanFilter<f64, Dyn, Dyn, Dyn>,
// }
// #[pyclass]
// struct EKFKC {
//     ekf: ExtendedKalmanFilterKnownCorrespondences<f64, Dyn, Dyn, Dyn>,
// }

// #[pyclass]
// struct UKF {
//     ukf: UnscentedKalmanFilter<f64, Dyn, Dyn, Dyn>,
// }

// #[pyclass]
// struct PF {
//     pf: ParticleFilterKnownCorrespondences<f64, Dyn, Dyn, Dyn>,
// }

// python port // must include pyo3 to make this work
/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn robotics(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
