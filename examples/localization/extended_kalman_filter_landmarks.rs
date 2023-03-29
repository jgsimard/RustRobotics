use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use robotics::utils::state::GaussianState;
use rustc_hash::FxHashMap;
use std::error::Error;

extern crate robotics;
use robotics::data::utias::UtiasDataset;
use robotics::localization::extended_kalman_filter::ExtendedKalmanFilterKnownCorrespondences;
use robotics::models::measurement::RangeBearingMeasurementModel;
use robotics::models::motion::Velocity;
use robotics::utils::plot::plot_landmarks;

fn main() -> Result<(), Box<dyn Error>> {
    let dataset = UtiasDataset::new(0)?;

    let mut landmarks = FxHashMap::default();
    for (id, lm) in &dataset.landmarks {
        landmarks.insert(*id, Vector3::new(lm.x, lm.y, 0.0));
    }

    let measurement_model = RangeBearingMeasurementModel::new();
    let noise = 1000000.0;
    let motion_model = Velocity::new(noise, noise, noise, noise, noise, noise);
    let mut q = Matrix2::<f64>::from_diagonal(&Vector2::new(350.0, 350.0));
    q = q * q;
    let r = Matrix3::identity(); //Observation x,y position covariance

    let gt_state = &dataset.groundtruth[0];
    let mut state = GaussianState {
        x: Vector3::new(gt_state.x, gt_state.y, gt_state.orientation),
        cov: Matrix3::<f64>::from_diagonal(&Vector3::new(1e-10, 1e-10, 1e-10)),
    };
    let mut time_past = gt_state.time;

    let mut states = Vec::new();
    let mut states_measurement = Vec::new();

    let ekf = ExtendedKalmanFilterKnownCorrespondences::new(
        r,
        q,
        landmarks,
        measurement_model,
        motion_model,
        false,
    );
    for (measurements, odometry) in (&dataset).into_iter().take(10000) {
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

        state = ekf.estimate(&state, odometry, measurements, dt);

        states.push(state.clone());
        if measurement_update {
            states_measurement.push(state.clone())
        }
    }
    println!("measurement updates = {}", states_measurement.len());

    plot_landmarks(
        &dataset,
        &states,
        &states_measurement,
        time_past,
        "./img/ekf_landmarks.png",
        "EKF landmarks",
    )?;

    Ok(())
}
