// Particle Filter localization sample
// author: Atsushi Sakai (@Atsushi_twi)
//         Jean-Gabriel Simard (@jgsimard)

extern crate robotics;
use robotics::utils::{deg2rad, Matrix, Vector};
fn gauss_likelihood(x: f32, sigma: f32) -> f32 {
    1.0 / f32::sqrt(2.0 * std::f32::consts::PI * sigma.powi(2))
        * f32::exp(-x.powi(2) / (2.0 * sigma.powi(2)))
}

/// state
/// [x, y, yaw, v]
struct SimpleProblem {
    /// Range error
    pub q: Matrix<4, 4>,
    /// Input Error
    pub r: Matrix<2, 2>,
    /// Number of particules
    pub np: usize,
    /// Number of particules for re-sampling
    pub nth: usize,
}

struct Simulation {
    pub q_sim: Matrix<4, 4>,
    pub r_sim: Matrix<2, 2>,
    pub dt: f32,
    pub sim_time: f32,
    pub max_range: f32,
}

fn main() {
    println!("Twerk!")
}
