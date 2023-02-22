// // Extended kalman filter (EKF) localization sample
// // author: Atsushi Sakai (@Atsushi_twi)
// //         Jean-Gabriel Simard (@jgsimard)

use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};
use plotters::prelude::*;
use rand_distr::{Distribution, Normal};
use std::error::Error;

#[path = "../plot.rs"]
mod plot;
use plot::{chart, History};

extern crate robotics;
use robotics::localization::extended_kalman_filter::{
    ExtendedKalmanFilter, ExtendedKalmanFilterModel,
};
use robotics::utils::deg2rad;
use robotics::utils::state::GaussianStateStatic as GaussianState;

/// State
/// [x, y, yaw, v]
///
/// Observation
/// [x,y]
struct SimpleProblem {
    // motion model
    // x_{t+1} = x_t+v*dt*cos(yaw)
    // y_{t+1} = y_t+v*dt*sin(yaw)
    // yaw_{t+1} = yaw_t+omega*dt
    // v_{t+1} = v{t}
    // so
    // dx/dyaw = -v*dt*sin(yaw)
    // dx/dv = dt*cos(yaw)
    // dy/dyaw = v*dt*cos(yaw)
    // dy/dv = dt*sin(yaw)
    pub gps_noise: Matrix2<f32>,
    pub input_noise: Matrix2<f32>,
}

impl ExtendedKalmanFilterModel<f32, 4, 2, 2> for SimpleProblem {
    fn motion_model(&self, x: &Vector4<f32>, u: &Vector2<f32>, dt: f32) -> Vector4<f32> {
        let yaw = x[2];
        let v = x[3];
        Vector4::new(
            x.x + yaw.cos() * v * dt,
            x.y + yaw.sin() * v * dt,
            yaw + u.y * dt,
            u.x,
        )
    }

    fn observation_model(&self, x: &Vector4<f32>) -> Vector2<f32> {
        x.xy()
    }

    fn jacobian_motion_model(&self, x: &Vector4<f32>, u: &Vector2<f32>, dt: f32) -> Matrix4<f32> {
        let yaw = x[2];
        let v = u[0];
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4::<f32>::new(
            1., 0., -dt * v * (yaw).sin(), dt * (yaw).cos(),
            0., 1., dt * v * (yaw).cos(), dt * (yaw).sin(),
            0., 0., 1., 0.,
            0., 0., 0., 0.,
        )
    }

    fn jacobian_observation_model(&self) -> Matrix2x4<f32> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix2x4::<f32>::new(
            1., 0., 0., 0.,
            0., 1., 0., 0.
        )
    }
}

impl SimpleProblem {
    fn observation(
        &self,
        x_true: &Vector4<f32>,
        x_deterministic: &Vector4<f32>,
        u: &Vector2<f32>,
        dt: f32,
    ) -> (Vector4<f32>, Vector2<f32>, Vector4<f32>, Vector2<f32>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0., 1.).unwrap();

        let x_true_next = self.motion_model(x_true, u, dt);

        // add noise to gps x-y
        let observation_noise =
            self.gps_noise * Vector2::new(normal.sample(&mut rng), normal.sample(&mut rng));
        let observation = self.observation_model(&x_true_next) + observation_noise;

        // add noise to input
        let u_noise =
            self.input_noise * Vector2::new(normal.sample(&mut rng), normal.sample(&mut rng));
        let ud = u + u_noise;

        let x_deterministic_next = self.motion_model(x_deterministic, &ud, dt);

        (x_true_next, observation, x_deterministic_next, ud)
    }
}

fn run() -> History {
    let sim_time = 50.0;
    let dt = 0.1;
    let mut time = 0.;

    // state : [x, y, yaw, v]
    let mut q = Matrix4::<f32>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
    q = q * q; // predict state covariance

    let r = nalgebra::Matrix2::identity(); //Observation x,y position covariance
    let ekf = ExtendedKalmanFilter::new(q, r);

    let simple_problem = SimpleProblem {
        input_noise: Matrix2::new(1., 0., 0., deg2rad(30.0).powi(2)),
        gps_noise: Matrix2::new(0.25, 0., 0., 0.25),
    };

    let u = Vector2::<f32>::new(1.0, 0.1);
    let mut ud: Vector2<f32>;
    let mut x_dr = Vector4::<f32>::new(0., 0., 0., 0.);
    let mut x_true = Vector4::<f32>::new(0., 0., 0., 0.);
    let mut kalman_state = GaussianState {
        x: Vector4::<f32>::new(0., 0., 0., 0.),
        P: Matrix4::<f32>::identity(),
    };
    let mut z: Vector2<f32>;

    let mut history = History::default();

    while time < sim_time {
        time += dt;
        (x_true, z, x_dr, ud) = simple_problem.observation(&x_true, &x_dr, &u, dt);
        kalman_state = ekf.predict(&simple_problem, &kalman_state, &ud, dt);
        kalman_state = ekf.update(&simple_problem, &kalman_state, &z);

        // record step
        history.z.push((z[0] as f64, z[1] as f64));
        history.x_true.push((x_true[0] as f64, x_true[1] as f64));
        history.x_dr.push((x_dr[0] as f64, x_dr[1] as f64));
        history
            .x_est
            .push((kalman_state.x[0] as f64, kalman_state.x[1] as f64));
        history.gaussian_state.push(kalman_state.clone());
    }
    history
}

fn main() -> Result<(), Box<dyn Error>> {
    // get data
    let history = run();
    let len = history.z.len();

    // Plot image
    let root = BitMapBackend::new("./img/ekf.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    chart(&root, &history, len - 1, "Extended Kalman Filter")?;
    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect(
        "Unable to write result to file, please make sure 'img' dir exists under current dir",
    );
    println!("Result has been saved to {}", "./img/ekf.png");

    // Plot GIF
    let root = BitMapBackend::gif("./img/ekf.gif", (1024, 768), 1)?.into_drawing_area();
    for i in (0..len).step_by(5) {
        root.fill(&WHITE)?;
        chart(&root, &history, i, "Extended Kalman Filter")?;
        // To avoid the IO failure being ignored silently, we manually call the present function
        root.present().expect(
            "Unable to write result to file, please make sure 'img' dir exists under current dir",
        );
    }
    println!("Result has been saved to {}", "./img/ekf.gif");

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::run;

    #[test]
    fn it_works() {
        run();
    }
}
