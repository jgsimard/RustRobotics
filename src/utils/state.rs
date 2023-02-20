#![allow(non_snake_case)]
#![allow(dead_code)]

use nalgebra::allocator::Allocator;
use nalgebra::{
    Const, DMatrix, DVector, DefaultAllocator, Dim, DimMin, OMatrix, OVector, RealField, SMatrix,
    SVector,
};

#[derive(Debug, Clone, Copy)]
pub struct GaussianStateStatic<T: RealField, const D: usize> {
    /// State Vector
    pub x: SVector<T, D>,
    /// Covariance Matrix
    pub P: SMatrix<T, D, D>,
}

fn normalD_pdf<T: RealField, const D: usize>(mean: T, variance: T, x: T) -> T
where
    Const<D>: DimMin<Const<D>, Output = Const<D>>,
{
    let neg_half = T::from_f32(-0.5).unwrap();
    let v = T::exp(neg_half * (x - mean).powi(2) / variance.clone());
    let factor = T::one() / (T::two_pi() * variance).sqrt();
    factor * v
}

fn normal_pdf<T: RealField, const D: usize>(
    mean: &SVector<T, D>,
    covariance: &SMatrix<T, D, D>,
    x: &SVector<T, D>,
) -> Option<T>
where
    Const<D>: DimMin<Const<D>, Output = Const<D>>,
{
    let neg_half = T::from_f32(-0.5)?;
    let dx = x - mean;
    // choleky works only if covariance is SDP
    let covariance_cholesky = covariance.clone().cholesky()?;
    let precision = covariance_cholesky.inverse();
    let det = covariance_cholesky.determinant();
    let interior = (dx.transpose() * precision * dx).x.clone();
    let v = T::exp(neg_half * interior);
    let factor = T::one() / (T::two_pi().powi(D as i32) * det).sqrt();
    Some(factor * v)
}

#[derive(Debug)]
pub struct GaussianStateDynamic<T: RealField> {
    /// State Vector
    pub x: DVector<T>,
    /// Covariance Matrix
    pub P: DMatrix<T>,
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
    pub P: OMatrix<T, D, D>,
}

impl<T: RealField> GaussianStateDynamic<T> {
    #[allow(dead_code)]
    fn new(x: DVector<T>, P: DMatrix<T>) -> GaussianStateDynamic<T> {
        assert!(P.is_square());
        assert_eq!(x.shape().0, P.shape().0);
        GaussianStateDynamic { x, P }
    }
}
