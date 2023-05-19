use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Matrix1, Matrix4, Matrix4x1, Vector4};
use plotpy::{Curve, Plot};
use std::error::Error;
use std::time::Instant;

extern crate robotics;
use robotics::control::lqr::{lqr, LinearModel};

/// [x, x_dot, theta, thetha_dot]
struct InvertedPendulumModel {
    da: Matrix4<f64>,
    db: Matrix4x1<f64>,
    r: Matrix1<f64>,
    q: Matrix4<f64>,
}

impl InvertedPendulumModel {
    fn new(l_bar: f64, mass_cart: f64, mass_ball: f64, g: f64) -> InvertedPendulumModel {
        let q = Matrix4::from_diagonal(&Vector4::new(10.00, 1.0, 10.0, 1.0));
        let r = Matrix1::new(0.01);

        #[rustfmt::skip]
        let da = Matrix4::new(
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, mass_ball * g / mass_cart, 0.0,
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, g * (mass_cart + mass_ball) / (l_bar * mass_cart), 0.0
        );

        let db = Matrix4x1::new(0.0, 1.0 / mass_cart, 0.0, 1.0 / (l_bar * mass_cart));

        InvertedPendulumModel { da, db, r, q }
    }
}

impl<'a> LinearModel<'a, f64, Const<4>, Const<1>> for InvertedPendulumModel {
    fn a(&self, dt: f64) -> Matrix4<f64> {
        Matrix4::identity() + dt * self.da
    }
    fn b(&self, dt: f64) -> Matrix4x1<f64> {
        dt * self.db
    }
    fn q(&'a self) -> &'a Matrix4<f64> {
        &self.q
    }
    fn r(&'a self) -> &'a Matrix1<f64> {
        &self.r
    }
}

fn run() -> Option<(Vec<Vector4<f64>>, Vec<f64>)> {
    let sim_time = 5.0;
    let dt = 0.01;
    let mut time = 0.;
    let max_iter = 500;
    let epsilon = 0.01;

    let l_bar = 2.0; // length of bar
    let mass_cart = 1.0; // [kg]
    let mass_ball = 0.3; // [kg]
    let g = 9.8; // [m/s^2]

    let linear_model = InvertedPendulumModel::new(l_bar, mass_cart, mass_ball, g);

    let mut x = Vector4::new(0.0, 0.0, -0.2, 0.0);
    let x_goal = Vector4::new(-0.0, 0.0, 0.0, 0.0);

    let mut states = vec![x];
    let mut commands = vec![0.0];

    while time < sim_time {
        time += dt;

        let start = Instant::now();
        let u = lqr(&(&x - &x_goal), dt, &linear_model, max_iter, epsilon)?;
        let duration = start.elapsed();
        println!("Time elapsed in lqr() is: {:?}", duration);

        x = linear_model.step(&x, &u, dt);

        states.push(x.clone());
        commands.push(u.x);
    }
    Some((states, commands))
}

fn main() -> Result<(), Box<dyn Error>> {
    let algos = &[
        "Linear Quadratic Regulator (LQR)",
        "Model Predictive Control (MPC)",
    ];
    let algo_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Pick control algorithm")
        .default(0)
        .items(&algos[..])
        .interact()
        .unwrap();
    // let algo = algos[algo_idx];

    // get data
    let (states, commands) = run().expect("unable to run");

    // Create output directory if it didnt exist
    std::fs::create_dir_all("./img")?;

    let path = match algo_idx {
        0 => "lqr",
        1 => "mpc",
        _ => unreachable!(),
    };

    let time = (0..states.len())
        .map(|i| (i as f64) / (states.len() as f64) * 5.0)
        .collect::<Vec<f64>>();

    let mut curve_x = Curve::new();
    curve_x
        .set_label("x")
        .draw(&time, &states.iter().map(|s| s.x).collect());

    let mut curve_theta = Curve::new();
    curve_theta
        .set_label("theta")
        .draw(&time, &states.iter().map(|s| s.z).collect());

    let mut curve_x_dot = Curve::new();
    curve_x_dot
        .set_label("x dot")
        .draw(&time, &states.iter().map(|s| s.y).collect());

    let mut curve_theta_dot = Curve::new();
    curve_theta_dot
        .set_label("theta dot")
        .draw(&time, &states.iter().map(|s| s.w).collect());

    let mut curve_u = Curve::new();
    curve_u.set_label("u").draw(&time, &commands);

    // add features to plot
    let mut plot = Plot::new();

    plot.add(&curve_x)
        .add(&curve_theta)
        .add(&curve_x_dot)
        .add(&curve_theta_dot)
        // .add(&curve_u)
        .legend()
        .set_equal_axes(true) // save figure
        .set_figure_size_points(1000.0, 1000.0)
        .save(format!("img/{}-{}.svg", path, "done").as_str())?;
    Ok(())
}
