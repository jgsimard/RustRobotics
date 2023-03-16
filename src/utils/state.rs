use nalgebra::allocator::Allocator;
use nalgebra::{
    DMatrix, DVector, DefaultAllocator, Dim, OMatrix, OVector, RealField, SMatrix, SVector,
};

#[derive(Debug, Clone, Copy)]
pub struct GaussianStateStatic<T: RealField, const D: usize> {
    /// State Vector
    pub x: SVector<T, D>,
    /// Covariance Matrix
    pub cov: SMatrix<T, D, D>,
}

#[derive(Debug)]
pub struct GaussianStateDynamic<T: RealField> {
    /// State Vector
    pub x: DVector<T>,
    /// Covariance Matrix
    pub cov: DMatrix<T>,
}

#[derive(Debug, Clone)]
pub struct GaussianState<T: RealField, D: Dim>
where
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, D, D>,
{
    /// State Vector
    pub x: OVector<T, D>,
    /// Covariance Matrix
    pub cov: OMatrix<T, D, D>,
}
