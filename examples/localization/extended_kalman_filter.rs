use nalgebra::{Matrix2, Matrix4, Vector2, Vector4};
use plotters::prelude::*;
use rand_distr::{Distribution, Normal};
use std::error::Error;

extern crate robotics;
use robotics::localization::extended_kalman_filter::ExtendedKalmanFilter;
use robotics::models::measurement::{MeasurementModel, SimpleProblemMeasurementModel};
use robotics::models::motion::{MotionModel, SimpleProblemMotionModel};
use robotics::utils::deg2rad;
use robotics::utils::plot::{chart, History};
use robotics::utils::state::GaussianStateStatic as GaussianState;

/// State
/// [x, y, yaw, v]
///
/// Observation
/// [x,y]
struct SimpleProblem {
    pub gps_noise: Matrix2<f64>,
    pub input_noise: Matrix2<f64>,
    pub motion_model: SimpleProblemMotionModel,
    pub observation_model: SimpleProblemMeasurementModel,
}

impl SimpleProblem {
    fn observation(
        &self,
        x_true: &Vector4<f64>,
        x_deterministic: &Vector4<f64>,
        u: &Vector2<f64>,
        dt: f64,
    ) -> (Vector4<f64>, Vector2<f64>, Vector4<f64>, Vector2<f64>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0., 1.).unwrap();

        let x_true_next = self.motion_model.prediction(x_true, u, dt);

        // add noise to gps x-y
        let observation_noise =
            self.gps_noise * Vector2::new(normal.sample(&mut rng), normal.sample(&mut rng));
        let observation = self.observation_model.prediction(&x_true_next, None) + observation_noise;

        // add noise to input
        let u_noise =
            self.input_noise * Vector2::new(normal.sample(&mut rng), normal.sample(&mut rng));
        let ud = u + u_noise;

        let x_deterministic_next = self.motion_model.prediction(x_deterministic, &ud, dt);

        (x_true_next, observation, x_deterministic_next, ud)
    }
}

fn run() -> History {
    let sim_time = 50.0;
    let dt = 0.1;
    let mut time = 0.;

    // state : [x, y, yaw, v]
    let mut q = Matrix4::<f64>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
    q = q * q; // predict state covariance

    let r = nalgebra::Matrix2::identity(); //Observation x,y position covariance
    let ekf = ExtendedKalmanFilter::new(
        q,
        r,
        Box::new(SimpleProblemMeasurementModel {}),
        Box::new(SimpleProblemMotionModel {}),
    );

    let simple_problem = SimpleProblem {
        input_noise: Matrix2::new(1., 0., 0., deg2rad(30.0).powi(2)),
        gps_noise: Matrix2::new(0.25, 0., 0., 0.25),
        observation_model: SimpleProblemMeasurementModel {},
        motion_model: SimpleProblemMotionModel {},
    };

    let u = Vector2::<f64>::new(1.0, 0.1);
    let mut ud: Vector2<f64>;
    let mut x_dr = Vector4::<f64>::new(0., 0., 0., 0.);
    let mut x_true = Vector4::<f64>::new(0., 0., 0., 0.);
    let mut kalman_state = GaussianState {
        x: Vector4::<f64>::new(0., 0., 0., 0.),
        P: Matrix4::<f64>::identity(),
    };
    let mut z: Vector2<f64>;

    let mut history = History::default();

    while time < sim_time {
        time += dt;
        (x_true, z, x_dr, ud) = simple_problem.observation(&x_true, &x_dr, &u, dt);
        kalman_state = ekf.estimate(&kalman_state, &ud, &z, dt);

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
