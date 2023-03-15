use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use plotters::prelude::*;

use rustc_hash::FxHashMap;
use std::error::Error;

extern crate robotics;
use robotics::data::UtiasDataset;
use robotics::localization::particle_filter::ParticleFilterKnownCorrespondences;
use robotics::models::measurement::RangeBearingMeasurementModel;
use robotics::models::motion::Velocity;
use robotics::utils::state::GaussianStateStatic;

fn plot(
    dataset: &UtiasDataset,
    states: &Vec<GaussianStateStatic<f64, 3>>,
    states_measurement: &Vec<GaussianStateStatic<f64, 3>>,
    max_time: f64,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("./img/pf_landmarks.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let name = "Particle Filter (Monte Carlo) landmarks";
    let min_x = 0.0;
    let max_x = 5.0;
    let min_y = -6.0;
    let max_y = 5.0;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(name, ("sans-serif", 40))
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().draw()?;

    // Ground Truth
    chart
        .draw_series(
            dataset
                .groundtruth
                .iter()
                .filter(|p| p.time <= max_time)
                .map(|p| Circle::new((p.x, p.y), 1, BLUE.filled())),
        )?
        .label("Ground truth")
        .legend(|(x, y)| Circle::new((x, y), 3, BLUE.filled()));

    // Landmarks
    chart
        .draw_series(
            dataset
                .landmarks
                .values()
                .map(|lm| Circle::new((lm.x, lm.y), 5, RED.filled())),
        )?
        .label("Landmarks")
        .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));

    chart.draw_series(dataset.landmarks.values().map(|lm| {
        Text::new(
            format!("{:?}", lm.subject_nb),
            (lm.x + 0.05, lm.y),
            ("sans-serif", 15),
        )
    }))?;

    // States
    chart
        .draw_series(
            states
                .iter()
                .map(|gs| Circle::new((gs.x[0], gs.x[1]), 1, GREEN.filled())),
        )?
        .label("Estimates")
        .legend(|(x, y)| Circle::new((x, y), 3, GREEN.filled()));

    // States Measurements
    chart
        .draw_series(
            states_measurement
                .iter()
                .map(|gs| Circle::new((gs.x[0], gs.x[1]), 1, RED.filled())),
        )?
        .label("Estimates Measurements")
        .legend(|(x, y)| Cross::new((x, y), 3, RED.filled()));

    // Legend
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .border_style(BLACK)
        .draw()?;

    root.present().expect(
        "Unable to write result to file, please make sure 'img' dir exists under current dir",
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let dataset = UtiasDataset::new(0)?;
    let mut landmarks = FxHashMap::default();
    for (id, lm) in &dataset.landmarks {
        landmarks.insert(*id, Vector3::new(lm.x, lm.y, 0.0));
    }
    let measurement_model = Box::new(RangeBearingMeasurementModel {});
    let noise = 1.0;
    let noise_w = 30.0;
    let noise_g = 10.0;
    // let motion_model = Box::new(Velocity4::new(noise, noise, noise_w, noise_w));
    let motion_model = Box::new(Velocity::new(
        noise, noise, noise_w, noise_w, noise_g, noise_g,
    ));
    let q = Matrix2::<f64>::from_diagonal(&Vector2::new(0.1, 0.2));
    //Observation x,y position covariance
    let r = Matrix3::<f64>::from_diagonal(&Vector3::new(0.2, 0.2, 0.2));

    let skip = 0;
    let gt_state = &dataset.groundtruth[skip];
    let state = GaussianStateStatic {
        x: Vector3::new(gt_state.x, gt_state.y, gt_state.orientation),
        P: Matrix3::<f64>::from_diagonal(&Vector3::new(1e-10, 1e-10, 1e-10)),
    };

    let mut particle_filter = ParticleFilterKnownCorrespondences::<f64, 3, 2, 2, 300>::new(
        r,
        q,
        landmarks,
        measurement_model,
        motion_model,
        state,
    );
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

        particle_filter.estimate(odometry, measurements, dt);

        states.push(particle_filter.gaussian_estimate());
        if measurement_update {
            states_measurement.push(states.last().unwrap().clone())
        }
    }
    println!("measurement updates = {}", states_measurement.len());

    plot(&dataset, &states, &states_measurement, time_past)?;

    Ok(())
}