#![allow(dead_code)]
use nalgebra::{RealField, SMatrix, SVector};

pub enum ErrorType {
    CovarianceNotSemiDefinitePositive,
}

pub struct Error {
    error_type: ErrorType,
}

struct MultiVariateNormal<T: RealField, const D: usize> {
    mean: SVector<T, D>,
    precision: SMatrix<T, D, D>,
    factor: T,
}

impl<T: RealField, const D: usize> MultiVariateNormal<T, D> {
    fn new(mean: &SVector<T, D>, covariance: &SMatrix<T, D, D>) -> Result<Self, Error> {
        let Some(covariance_cholesky) = covariance.clone().cholesky() else {
            return Err(Error{error_type : ErrorType::CovarianceNotSemiDefinitePositive})
        };
        let det = covariance_cholesky.determinant();
        let precision = covariance_cholesky.inverse();
        let factor = T::one() / (T::two_pi().powi(D as i32) * det).sqrt();
        let mvn = MultiVariateNormal {
            mean: mean.clone(),
            precision,
            factor,
        };
        Ok(mvn)
    }

    /// Probability density function
    fn pdf(self, x: &SVector<T, D>) -> T {
        let dx = self.mean - x;
        let neg_half = T::from_f32(-0.5).unwrap();
        let interior = (dx.transpose() * self.precision * dx).x.clone();
        self.factor * T::exp(neg_half * interior)
    }
}
