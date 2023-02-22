#![allow(incomplete_features)]
#![feature(generic_const_exprs)] // for testing replacing Vec<T> by [T; N] in some places

// my stuff
pub mod localization;
pub mod utils;

// // python port // must include pyo3 to make this work
// use pyo3::prelude::*;
// /// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

// /// A Python module implemented in Rust.
// #[pymodule]
// fn robotics(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
//     Ok(())
// }
