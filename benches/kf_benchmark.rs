use criterion::{criterion_group, criterion_main, Criterion};

use nalgebra::{Matrix4, Vector2, Vector4};
extern crate robotics;
use robotics::localization::extended_kalman_filter::ExtendedKalmanFilter;
use robotics::localization::unscented_kalman_filter::UnscentedKalmanFilter;
use robotics::models::measurement::SimpleProblemMeasurementModel;
use robotics::models::motion::SimpleProblemMotionModel;
use robotics::utils::deg2rad;
use robotics::utils::state::GaussianStateStatic as GaussianState;

fn ekf(b: &mut Criterion) {
    // setup ekf
    let q = Matrix4::<f64>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
    let r = nalgebra::Matrix2::identity();
    let motion_model = Box::new(SimpleProblemMotionModel {});
    let measurement_model = Box::new(SimpleProblemMeasurementModel {});
    let ekf = ExtendedKalmanFilter::<f64, 4, 2, 2>::new(q, r, measurement_model, motion_model);

    let dt = 0.1;
    let u: Vector2<f64> = Default::default();
    let kalman_state = GaussianState {
        x: Vector4::<f64>::new(0., 0., 0., 0.),
        P: Matrix4::<f64>::identity(),
    };
    let z: Vector2<f64> = Default::default();

    b.bench_function("ekf", |b| {
        b.iter(|| ekf.estimate(&kalman_state, &u, &z, dt))
    });
}

fn ukf(b: &mut Criterion) {
    // setup ukf
    let dt = 0.1;
    let q = Matrix4::<f64>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
    let r = nalgebra::Matrix2::identity(); //Observation x,y position covariance
    let ukf = UnscentedKalmanFilter::<f64, 4, 2, 2>::new(
        q,
        r,
        Box::new(SimpleProblemMeasurementModel {}),
        Box::new(SimpleProblemMotionModel {}),
        0.1,
        2.0,
        0.0,
    );

    let u: Vector2<f64> = Default::default();
    let kalman_state = GaussianState {
        x: Vector4::<f64>::new(0., 0., 0., 0.),
        P: Matrix4::<f64>::identity(),
    };
    let z: Vector2<f64> = Default::default();

    b.bench_function("ukf", |b| {
        b.iter(|| ukf.estimate(&kalman_state, &u, &z, dt))
    });
}

criterion_group!(benches, ekf, ukf);
criterion_main!(benches);
