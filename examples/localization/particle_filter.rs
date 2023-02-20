// Particle Filter localization sample
// author: Atsushi Sakai (@Atsushi_twi)
//         Jean-Gabriel Simard (@jgsimard)

use nalgebra::{
    Matrix1, Matrix2, Matrix2x4, Matrix3, Matrix4, Matrix4x2, SMatrix, SVector, Vector1, Vector2,
    Vector4,
};

extern crate robotics;
use robotics::utils::{deg2rad, GaussianStateStatic};

fn gauss_likelihood(x: f32, sigma: f32) -> f32 {
    f32::exp(-x.powi(2) / (2.0 * sigma.powi(2)))
        / f32::sqrt(2.0 * std::f32::consts::PI * sigma.powi(2))
}

pub trait ParticleFilterStatic<
    const STATE_SIZE: usize,
    const OBSERVATION_SIZE: usize,
    const INPUT_SIZE: usize,
    T: nalgebra::RealField,
>
{
    fn motion_model(
        &self,
        x: &SVector<T, STATE_SIZE>,
        u: &SVector<T, INPUT_SIZE>,
        dt: T,
    ) -> SVector<T, STATE_SIZE> {
        self.f(x, dt.clone()) * x + self.b(x, dt) * u
    }

    fn observation_model(&self, x: &SVector<T, STATE_SIZE>) -> SVector<T, OBSERVATION_SIZE> {
        self.h() * x
    }

    fn f(&self, x: &SVector<T, STATE_SIZE>, dt: T) -> SMatrix<T, STATE_SIZE, STATE_SIZE>;
    fn b(&self, x: &SVector<T, STATE_SIZE>, dt: T) -> SMatrix<T, STATE_SIZE, INPUT_SIZE>;
    fn h(&self) -> SMatrix<T, OBSERVATION_SIZE, STATE_SIZE>;
    // fn r(&self) -> SMatrix<T, OBSERVATION_SIZE, OBSERVATION_SIZE>;
    // fn q(&self) -> SMatrix<T, STATE_SIZE, STATE_SIZE>;
}

/// state
/// [x, y, yaw, v]
struct SimpleProblem {
    /// Range error
    pub q: Matrix4<f32>,
    /// Input Error
    pub r: Matrix2<f32>,
    /// Number of particules
    pub np: usize,
    /// Number of particules for re-sampling
    pub nth: usize,
}

struct Simulation {
    pub q_sim: Matrix4<f32>,
    pub r_sim: Matrix2<f32>,
    pub dt: f32,
    pub sim_time: f32,
    pub max_range: f32,
}

impl ParticleFilterStatic<4, 2, 2, f32> for SimpleProblem {
    fn f(&self, _x: &Vector4<f32>, _dt: f32) -> Matrix4<f32> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4::<f32>::new(
            1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        )
    }

    fn b(&self, x: &Vector4<f32>, dt: f32) -> Matrix4x2<f32> {
        let yaw = x[2];

        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4x2::<f32>::new(
            dt * (yaw).cos(), 0.,
            dt * (yaw).sin(), 0.,
            0., dt,
            1., 0.
        )
    }

    fn h(&self) -> Matrix2x4<f32> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix2x4::<f32>::new(
            1., 0., 0., 0.,
            0., 1., 0., 0.
        )
    }
}

impl SimpleProblem {
    fn localization(particules: Vec<Vector4<f32>>, weights: Vec<f32>) {
        // for (particule, weight) in particules.iter().zip(weights.iter()) {}
    }

    fn compute_covariance(
        x_est: &Vector4<f32>,
        particules: &Vec<Vector4<f32>>,
        weights: &Vec<f32>,
    ) -> Matrix4<f32> {
        particules
            .iter()
            .zip(weights.iter())
            .map(|(p, w)| {
                let dx = p - x_est;
                *w * (dx * dx.transpose())
            })
            .sum::<Matrix4<f32>>()
    }

    // fn observation(
    //     &self,
    //     x_true: &Vector4<f32>,
    //     x_dr: &Vector4<f32>,
    //     u: &Vector2<f32>,
    //     dt: f32,
    // ) -> (Vector4<f32>, Vector2<f32>, Vector4<f32>, Vector2<f32>) {
    //     let mut rng = rand::thread_rng();
    //     let normal = Normal::new(0., 1.).unwrap();

    //     let x_true_next = self.motion_model(x_true, u, dt);

    //     // add noise to gps x-y
    //     let observation_noise =
    //         self.gps_noise * Vector2::new(normal.sample(&mut rng), normal.sample(&mut rng));
    //     let observation = self.observation_model(&x_true_next) + observation_noise;

    //     // add noise to input
    //     let u_noise =
    //         self.input_noise * Vector2::new(normal.sample(&mut rng), normal.sample(&mut rng));
    //     let ud = u + u_noise;

    //     let x_deterministic_next = self.motion_model(x_deterministic, &ud, dt);

    //     (x_true_next, observation, x_deterministic_next, ud)
    // }
}

fn run() {
    let mut q = Matrix1::<f32>::from_diagonal(&Vector1::new(0.2));
    q = q * q; // range error

    let mut r = Matrix2::<f32>::from_diagonal(&Vector2::new(2.0, deg2rad(40.0)));
    r = r * r; // input error

    // simulation parameters
    let mut q_sim = Matrix1::<f32>::from_diagonal(&Vector1::new(0.2));
    q_sim = q_sim * q_sim; // range error

    let mut r_sim = Matrix2::<f32>::from_diagonal(&Vector2::new(1.0, deg2rad(30.0)));
    r_sim = r_sim * r_sim; // input error

    let sim_time = 50.0;
    let dt = 0.1;
    let mut time = 0.;
    let max_range = 20.0; // max observation range

    let number_particules = 100;
    let number_particules_resampling = number_particules / 2;

    let u = Vector2::<f32>::new(1.0, 0.1);
}

fn main() {
    println!("uwu");
}
