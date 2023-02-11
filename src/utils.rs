#![allow(non_snake_case)]

use nalgebra::{DMatrix, DVector, Matrix2, Matrix3, RealField, SMatrix, SVector, Vector2};
use nalgebra_lapack::Eigen;

pub fn deg2rad(x: f32) -> f32 {
    const DEG2RAD_FACTOR: f32 = std::f32::consts::PI / 180.0;
    x * DEG2RAD_FACTOR
}

pub fn rad2deg(x: f32) -> f32 {
    const RAD2DEG_FACTOR: f32 = 180.0 / std::f32::consts::PI;
    x * RAD2DEG_FACTOR
}

#[derive(Debug, Clone, Copy)]
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

pub fn ellipse_series(xy: Vector2<f32>, p_xy: Matrix2<f32>) -> Option<Vec<(f64, f64)>> {
    let eigen = Eigen::new(p_xy, false, true)?;
    let eigenvectors = eigen.eigenvectors?;
    let eigenvalues = eigen.eigenvalues_re;
    let (a, b, angle) = if eigenvalues.x >= eigenvalues.y {
        (
            eigenvalues.x.sqrt(),
            eigenvalues.y.sqrt(),
            f32::atan2(eigenvectors.m21, eigenvectors.m22),
        )
    } else {
        (
            eigenvalues.y.sqrt(),
            eigenvalues.x.sqrt(),
            f32::atan2(eigenvectors.m22, eigenvectors.m21),
        )
    };

    let rot_mat = Matrix3::new_rotation(angle)
        .fixed_view::<2, 2>(0, 0)
        .clone_owned();

    let xy = (0..100)
        .map(|x| x as f32 / 100.0 * std::f32::consts::TAU)
        .map(|t| rot_mat * Vector2::new(a * t.cos(), b * t.sin()) + xy)
        .map(|xy| (xy.x as f64, xy.y as f64))
        .collect();
    Some(xy)
}
