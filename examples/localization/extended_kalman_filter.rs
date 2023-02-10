// // Extended kalman filter (EKF) localization sample
// // author: Atsushi Sakai (@Atsushi_twi)
// //         Jean-Gabriel Simard (@jgsimard)

use nalgebra::{Matrix2, Matrix2x4, Matrix4, Matrix4x2, Vector2, Vector4};
use plotters::prelude::*;
use rand_distr::{Distribution, Normal};
use std::error::Error;

extern crate robotics;
use robotics::localization::extended_kalman_filter::ExtendedKalmanFilterStatic;
use robotics::utils::{deg2rad, GaussianStateStatic};

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
    pub q: Matrix4<f32>,
    pub r: Matrix2<f32>,
}

impl ExtendedKalmanFilterStatic<4, 2, 2, f32> for SimpleProblem {
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

    fn jacob_f(&self, x: &Vector4<f32>, u: &Vector2<f32>, dt: f32) -> Matrix4<f32> {
        let yaw = x[2];
        let v = u[0];
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4::<f32>::new(
            1., 0., -dt * v * (yaw).sin(), dt * (yaw).cos(),
            0., 1., dt * v * (yaw).cos(), dt * (yaw).sin(),
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        )
    }

    fn jacob_h(&self) -> Matrix2x4<f32> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix2x4::<f32>::new(
            1., 0., 0., 0.,
            0., 1., 0., 0.
        )
    }

    fn q(&self) -> Matrix4<f32> {
        self.q
    }

    fn r(&self) -> Matrix2<f32> {
        self.r
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

        // let m = nalgebra::base::Matrix::from_distribution_generic(Const::<2>, Const::<1>, &normal, &mut rng);

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

#[derive(Debug, Default)]
struct History {
    pub z: Vec<(f64, f64)>,
    pub x_true: Vec<(f64, f64)>,
    pub x_dr: Vec<(f64, f64)>,
    pub x_est: Vec<(f64, f64)>,
}

fn run() -> History {
    let sim_time = 50.0;
    let dt = 0.1;
    let mut time = 0.;

    // state : [x, y, yaw, v]
    let mut q = Matrix4::<f32>::from_diagonal(&Vector4::new(0.1, deg2rad(1.0), 0.1, 1.0));
    q = q * q; // predict state covariance

    let r = nalgebra::Matrix2::identity(); //Observation x,y position covariance

    let simple_problem = SimpleProblem {
        input_noise: nalgebra::Matrix2::new(1., 0., 0., deg2rad(30.0).powi(2)),
        gps_noise: nalgebra::Matrix2::new(1.0, 0., 0., 1.0),
        q,
        r,
    };

    let u = Vector2::<f32>::new(1.0, 0.1);
    let mut ud: Vector2<f32>;
    let mut x_dr = Vector4::<f32>::new(0., 0., 0., 0.);
    let mut x_true = Vector4::<f32>::new(0., 0., 0., 0.);
    let mut kalman_state = GaussianStateStatic {
        x: Vector4::<f32>::new(0., 0., 0., 0.),
        P: Matrix4::<f32>::identity(),
    };
    let mut z: Vector2<f32>;

    let mut history = History::default();

    while time < sim_time {
        time += dt;
        (x_true, z, x_dr, ud) = simple_problem.observation(&x_true, &x_dr, &u, dt);
        kalman_state = simple_problem.predict(&kalman_state, &ud, dt);
        kalman_state = simple_problem.update(&kalman_state, &z);

        // record step
        history.z.push((z[0] as f64, z[1] as f64));
        history.x_true.push((x_true[0] as f64, x_true[1] as f64));
        history.x_dr.push((x_dr[0] as f64, x_dr[1] as f64));
        history
            .x_est
            .push((kalman_state.x[0] as f64, kalman_state.x[1] as f64));
    }
    history
}

fn main() -> Result<(), Box<dyn Error>> {
    // get data
    let history = run();

    // PLOT : this is very bad :( much better in python
    let root = BitMapBackend::new("./img/ekf.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("Extended Kalman Filter", ("sans-serif", 40))
        .build_cartesian_2d(-15.0..15.0, -5.0..25.0)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(
            history
                .z
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, RED.filled())),
        )?
        .label("observation")
        .legend(|(x, y)| Circle::new((x, y), 3, RED.filled()));
    chart
        .draw_series(
            history
                .x_true
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, BLUE.filled())),
        )?
        .label("true position")
        .legend(|(x, y)| Circle::new((x, y), 3, BLUE.filled()));
    chart
        .draw_series(
            history
                .x_dr
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, YELLOW.filled())),
        )?
        .label("no kalman")
        .legend(|(x, y)| Circle::new((x, y), 3, YELLOW.filled()));
    chart
        .draw_series(
            history
                .x_est
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, GREEN.filled())),
        )?
        .label("kalman estimate")
        .legend(|(x, y)| Circle::new((x, y), 3, GREEN.filled()));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .border_style(BLACK)
        .draw()?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect(
        "Unable to write result to file, please make sure 'img' dir exists under current dir",
    );
    println!("Result has been saved to {}", "./img/ekf.png");

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
