use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OVector, RealField};

use crate::utils::state::GaussianState;

pub trait BayesianFilter<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, Z>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Z, S>
        + Allocator<T, S, U>
        + Allocator<T, U, U>
        + Allocator<T, S, Z>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
{
    fn update_estimate(
        &mut self,
        // estimate: &GaussianState<T, S>,
        u: &OVector<T, U>,
        z: &OVector<T, Z>,
        dt: T,
    );
    // ) -> GaussianState<T, S>;

    fn gaussian_estimate(&self) -> GaussianState<T, S>;
}
