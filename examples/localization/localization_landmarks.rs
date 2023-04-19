use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Matrix2, Matrix3, Vector2, Vector3};
use rustc_hash::FxHashMap;
use std::error::Error;

extern crate robotics;
use robotics::data::utias::UtiasDataset;
use robotics::localization::BayesianFilterKnownCorrespondences;
use robotics::localization::{
    ExtendedKalmanFilterKnownCorrespondences, ParticleFilterKnownCorrespondences,
};
use robotics::models::measurement::RangeBearingMeasurementModel;
use robotics::models::motion::Velocity;
use robotics::utils::plot::plot_landmarks;
use robotics::utils::state::GaussianState;

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
    // let algo = algos[algo_idx];

    let dataset = UtiasDataset::new(0)?;
    let mut landmarks = FxHashMap::default();
    for (id, lm) in &dataset.landmarks {
        landmarks.insert(*id, Vector3::new(lm.x, lm.y, 0.0));
    }
    let measurement_model = RangeBearingMeasurementModel::new();
    let noise = 1.0;
    let noise_w = 30.0;
    let noise_g = 10.0;
    let motion_model = Velocity::new([noise, noise, noise_w, noise_w, noise_g, noise_g]);
    let q = Matrix2::<f64>::from_diagonal(&Vector2::new(0.1, 0.2));
    //Observation x,y position covariance
    let r = Matrix3::<f64>::from_diagonal(&Vector3::new(0.2, 0.2, 0.2));

    let skip = 0;
    let gt_state = &dataset.groundtruth[skip];
    let state = GaussianState {
        x: Vector3::new(gt_state.x, gt_state.y, gt_state.orientation),
        cov: Matrix3::<f64>::from_diagonal(&Vector3::new(1e-10, 1e-10, 1e-10)),
    };

    let mut bayes_filter: Box<
        dyn BayesianFilterKnownCorrespondences<f64, Const<3>, Const<2>, Const<2>>,
    > = match algo_idx {
        0 => Box::new(ExtendedKalmanFilterKnownCorrespondences::new(
            q,
            landmarks,
            measurement_model,
            motion_model,
            state,
        )),
        1 => todo!(),
        2 => Box::new(ParticleFilterKnownCorrespondences::new(
            r,
            q,
            landmarks,
            measurement_model,
            motion_model,
            state,
            300,
        )),
        _ => unreachable!(),
    };

    let mut time_past = gt_state.time;

    let mut states = Vec::new();
    let mut states_measurement = Vec::new();

    for (measurements, odometry) in (&dataset).into_iter().skip(skip).take(10000) {
        let (time_now, measurement_update) = if let Some(m) = &measurements {
            (m.first().unwrap().time, true)
        } else if let Some(od) = &odometry {
            (od.time, false)
        } else {
            panic!("NOOOOOOOOO")
        };
        let dt = time_now - time_past;
        time_past = time_now;

        let measurements = measurements.map(|ms| {
            ms.iter()
                .map(|m| (m.subject_nb, Vector2::new(m.range, m.bearing)))
                .collect::<Vec<(u32, Vector2<f64>)>>()
        });

        let odometry = odometry.map(|od| Vector2::new(od.forward_velocity, od.angular_velocity));

        bayes_filter.update_estimate(odometry, measurements, dt);

        states.push(bayes_filter.gaussian_estimate());
        if measurement_update {
            states_measurement.push(states.last().unwrap().clone())
        }
    }
    println!("measurement updates = {}", states_measurement.len());

    let (path, name) = match algo_idx {
        0 => ("ekf_landmarks", "EKF landmarks"),
        1 => todo!(),
        2 => ("pf_landmarks", "Particle Filter (Monte Carlo) landmarks"),
        _ => unreachable!(),
    };
    plot_landmarks(
        &dataset,
        &states,
        &states_measurement,
        time_past,
        format!("./img/{path}.png").as_str(),
        name,
    )?;

    Ok(())
}
