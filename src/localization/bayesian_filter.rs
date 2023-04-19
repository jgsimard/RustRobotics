use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, RealField};

use crate::utils::state::GaussianState;

pub trait BayesianFilter<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, U> + Allocator<T, Z> + Allocator<T, S, S>,
{
    fn update_estimate(&mut self, u: &OVector<T, U>, z: &OVector<T, Z>, dt: T);

    fn gaussian_estimate(&self) -> GaussianState<T, S>;
}

pub trait BayesianFilterKnownCorrespondences<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, U> + Allocator<T, Z> + Allocator<T, S, S>,
{
    fn update_estimate(
        &mut self,
        control: Option<OVector<T, U>>,
        measurements: Option<Vec<(u32, OVector<T, Z>)>>,
        dt: T,
    );

    fn gaussian_estimate(&self) -> GaussianState<T, S>;
}
