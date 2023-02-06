use nalgebra::ArrayStorage;
use nalgebra::Const;

pub type Vector<const N: usize> =
    nalgebra::Matrix<f32, Const<N>, Const<1>, ArrayStorage<f32, N, 1>>;
pub type Matrix<const A: usize, const B: usize> =
    nalgebra::Matrix<f32, Const<A>, Const<B>, ArrayStorage<f32, A, B>>;

pub fn deg2rad(x: f32) -> f32 {
    const DEG2RAD_FACTOR: f32 = std::f32::consts::PI / 180.0;
    x * DEG2RAD_FACTOR
}

pub fn rad2deg(x: f32) -> f32 {
    const RAD2DEG_FACTOR: f32 = 180.0 / std::f32::consts::PI;
    x * RAD2DEG_FACTOR
}
