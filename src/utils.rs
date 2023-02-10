#![allow(non_snake_case)]

use nalgebra::{DMatrix, DVector, RealField, SMatrix, SVector};

pub fn deg2rad(x: f32) -> f32 {
    const DEG2RAD_FACTOR: f32 = std::f32::consts::PI / 180.0;
    x * DEG2RAD_FACTOR
}

pub fn rad2deg(x: f32) -> f32 {
    const RAD2DEG_FACTOR: f32 = 180.0 / std::f32::consts::PI;
    x * RAD2DEG_FACTOR
}

#[derive(Debug)]
pub struct GaussianStateStatic<T: RealField, const D: usize> {
    /// State Vector
    pub x: SVector<T, D>,
    /// Covariance Matrix
    pub P: SMatrix<T, D, D>,
}

#[derive(Debug)]
pub struct GaussianStateDynamic<T: RealField> {
    /// State Vector
    pub x: DVector<T>,
    /// Covariance Matrix
    pub P: DMatrix<T>,
}

impl<T: RealField> GaussianStateDynamic<T> {
    #[allow(dead_code)]
    fn new(x: DVector<T>, P: DMatrix<T>) -> GaussianStateDynamic<T> {
        assert!(P.is_square());
        assert_eq!(x.shape().0, P.shape().0);
        GaussianStateDynamic { x, P }
    }
}
