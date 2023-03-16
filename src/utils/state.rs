use nalgebra::{RealField, SMatrix, SVector};

#[derive(Debug, Clone, Copy)]
pub struct GaussianState<T: RealField, const D: usize> {
    /// State Vector
    pub x: SVector<T, D>,
    /// Covariance Matrix
    pub cov: SMatrix<T, D, D>,
}
