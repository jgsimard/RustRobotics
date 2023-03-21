use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, OVector, RealField};
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::{Standard, StandardNormal};
use rustc_hash::FxHashMap;

use crate::models::measurement::MeasurementModel;
use crate::models::motion::MotionModel;
use crate::utils::mvn::MultiVariateNormal;
use crate::utils::state::GaussianState;

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ParticleFilterKnownCorrespondences<T: RealField, S: Dim, Z: Dim, U: Dim>
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
    q: OMatrix<T, Z, Z>,
    landmarks: FxHashMap<u32, OVector<T, S>>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
    motion_model: Box<dyn MotionModel<T, S, Z, U>>,
    pub particules: Vec<OVector<T, S>>,
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> ParticleFilterKnownCorrespondences<T, S, Z, U>
where
    StandardNormal: Distribution<T>,
    Standard: Distribution<T>,
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
    pub fn new(
        initial_noise: OMatrix<T, S, S>,
        q: OMatrix<T, Z, Z>,
        landmarks: FxHashMap<u32, OVector<T, S>>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
        motion_model: Box<dyn MotionModel<T, S, Z, U>>,
        initial_state: GaussianState<T, S>,
        num_particules: usize,
    ) -> ParticleFilterKnownCorrespondences<T, S, Z, U> {
        let mvn = MultiVariateNormal::new(&initial_state.x, &initial_noise).unwrap();
        let mut particules = Vec::with_capacity(num_particules);
        for _ in 0..num_particules {
            particules.push(mvn.sample());
        }

        ParticleFilterKnownCorrespondences {
            q,
            landmarks,
            measurement_model,
            motion_model,
            particules,
        }
    }

    pub fn estimate(
        &mut self,
        control: Option<OVector<T, U>>,
        measurements: Option<Vec<(u32, OVector<T, Z>)>>,
        dt: T,
    ) {
        if let Some(u) = control {
            self.particules = self
                .particules
                .iter()
                .map(|p| self.motion_model.sample(p, &u, dt))
                .collect();
        }

        if let Some(measurements) = measurements {
            let mut weights = vec![T::one(); self.particules.len()];
            let shape = measurements[0].1.shape_generic();
            let mvn = MultiVariateNormal::new(&OMatrix::zeros_generic(shape.0, shape.1), &self.q)
                .unwrap();

            for (id, z) in measurements
                .iter()
                .filter(|(id, _)| self.landmarks.contains_key(id))
            {
                let landmark = self.landmarks.get(id);
                for (i, particule) in self.particules.iter().enumerate() {
                    let z_pred = self.measurement_model.prediction(particule, landmark);
                    let error = z - z_pred;
                    let pdf = mvn.pdf(&error);
                    weights[i] *= pdf;
                }
            }
            self.resampling(&weights);
            // self.resampling_sort(weights);
        }
    }

    fn resampling(&mut self, weights: &[T]) {
        let mut weight_tot = T::zero();
        let cum_weight: Vec<T> = (0..self.particules.len())
            .map(|i| {
                weight_tot += weights[i];
                weight_tot
            })
            .collect();

        // sampling
        let mut rng = rand::thread_rng();
        self.particules = (0..self.particules.len())
            .map(|_| {
                let rng_nb = rng.gen() * weight_tot;
                for i in 0..self.particules.len() {
                    if (&cum_weight)[i] > rng_nb {
                        return self.particules[i].clone();
                    }
                }
                unreachable!()
            })
            .collect();
    }

    // fn resampling_sort(&mut self, weights: [T; NP]) {
    //     let total_weight: T = weights.iter().fold(T::zero(), |a, b| a + *b);
    //     let mut rng = rand::thread_rng();
    //     let mut draws: Vec<T> = (0..NP).map(|_| rng.gen() * total_weight).collect();
    //     draws.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    //     let mut index = 0;
    //     let mut cum_weight = draws[0];
    //     let new_particules = core::array::from_fn(|i| {
    //         while cum_weight < draws[i] {
    //             if index == NP - 1 {
    //                 // weird precision edge case
    //                 cum_weight = total_weight;
    //                 break;
    //             } else {
    //                 cum_weight += weights[index];
    //                 index += 1;
    //             }
    //         }
    //         self.particules[index]
    //     });
    //     self.particules = new_particules;
    // }

    pub fn gaussian_estimate(&self) -> GaussianState<T, S> {
        let shape = self.particules[0].shape_generic();
        let x = self
            .particules
            .iter()
            .fold(OMatrix::zeros_generic(shape.0, shape.1), |a, b| a + b)
            / T::from_usize(self.particules.len()).unwrap();
        let cov = self
            .particules
            .iter()
            .map(|p| p - &x)
            .map(|dx| &dx * dx.transpose())
            .fold(OMatrix::zeros_generic(shape.0, shape.0), |a, b| a + b)
            / T::from_usize(self.particules.len()).unwrap();
        GaussianState { x, cov }
    }
}
