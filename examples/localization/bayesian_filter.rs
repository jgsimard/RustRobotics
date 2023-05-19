use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Matrix2, Matrix4, Vector2, Vector4};
use plotters::prelude::*;
use rand_distr::{Distribution, Normal};
use std::error::Error;

extern crate robotics;
use robotics::localization::{
    BayesianFilter, ExtendedKalmanFilter, ParticleFilter, ResamplingScheme, UnscentedKalmanFilter,
};
use robotics::models::measurement::{MeasurementModel, SimpleProblemMeasurementModel};
use robotics::models::motion::{MotionModel, SimpleProblemMotionModel};
use robotics::utils::deg2rad;
use robotics::utils::plot::{chart, History};
use robotics::utils::state::GaussianState;

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

fn run(algo: &str) -> History {
    let sim_time = 50.0;
    let dt = 0.1;
    let mut time = 0.;

    // state : [x, y, yaw, v]
    let mut q = Matrix4::<f64>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
    q = q * q; // predict state covariance

    let r = nalgebra::Matrix2::identity(); //Observation x,y position covariance

    let initial_state = GaussianState {
        x: Vector4::<f64>::new(0., 0., 0., 0.),
        cov: Matrix4::<f64>::identity(),
    };
    let mut bayesian_filter: Box<dyn BayesianFilter<f64, Const<4>, Const<2>, Const<2>>> = match algo
    {
        "Extended Kalman Filter (EKF)" => Box::new(ExtendedKalmanFilter::new(
            q,
            r,
            SimpleProblemMeasurementModel::new(),
            SimpleProblemMotionModel::new(),
            initial_state,
        )),
        "Unscented Kalman Filter (UKF)" => Box::new(UnscentedKalmanFilter::new(
            q,
            r,
            SimpleProblemMeasurementModel::new(),
            SimpleProblemMotionModel::new(),
            0.1,
            2.0,
            0.0,
            initial_state,
        )),
        "Particle Filter (PF)" => Box::new(ParticleFilter::new(
            q,
            r,
            SimpleProblemMeasurementModel::new(),
            SimpleProblemMotionModel::new(),
            initial_state,
            300,
            ResamplingScheme::Stratified,
        )),
        _ => unimplemented!("{}", algo),
    };

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
    let mut z: Vector2<f64>;

    let mut history = History::default();

    while time < sim_time {
        time += dt;
        (x_true, z, x_dr, ud) = simple_problem.observation(&x_true, &x_dr, &u, dt);
        bayesian_filter.update_estimate(&ud, &z, dt);
        let gaussian_state = bayesian_filter.gaussian_estimate();

        // record step
        history.z.push((z[0] as f64, z[1] as f64));
        history.x_true.push((x_true[0] as f64, x_true[1] as f64));
        history.x_dr.push((x_dr[0] as f64, x_dr[1] as f64));
        history
            .x_est
            .push((gaussian_state.x[0] as f64, gaussian_state.x[1] as f64));
        history.gaussian_state.push(gaussian_state.clone());
    }
    history
}

fn main() -> Result<(), Box<dyn Error>> {
    let algos = &[
        "Extended Kalman Filter (EKF)",
        "Unscented Kalman Filter (UKF)",
        "Particle Filter (PF)",
    ];
    let algo_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Pick localization algorithm")
        .default(0)
        .items(&algos[..])
        .interact()
        .unwrap();
    let algo = algos[algo_idx];

    // get data
    let history = run(algo);
    let len = history.z.len();

    // Create output directory if it didnt exist
    std::fs::create_dir_all("./img")?;

    let path = match algo_idx {
        0 => "ekf",
        1 => "ukf",
        2 => "pf",
        _ => unreachable!(),
    };

    // Plot image
    let path_img = format!("./img/{path}.png");
    let root = BitMapBackend::new(path_img.as_str(), (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    chart(&root, &history, len - 1, algo)?;
    root.present()?;
    println!("Result has been saved to {path_img}");

    // Plot GIF
    println!("Building a gif...");
    let path_gif = format!("./img/{path}.gif");
    let root = BitMapBackend::gif(path_gif.as_str(), (1024, 768), 1)?.into_drawing_area();
    for i in (0..len).step_by(5) {
        root.fill(&WHITE)?;
        chart(&root, &history, i, algo)?;
        root.present()?;
    }
    println!("Result has been saved to {path_gif}");

    Ok(())
}
