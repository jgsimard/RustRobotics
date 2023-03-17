use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, RealField};

#[derive(Debug, Clone)]
pub struct GaussianState<T: RealField, D: Dim>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    /// State Vector
    pub x: OVector<T, D>,
    /// Covariance Matrix
    pub cov: OMatrix<T, D, D>,
}
