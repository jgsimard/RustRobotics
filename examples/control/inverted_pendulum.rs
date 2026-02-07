use dialoguer::{theme::ColorfulTheme, Select};
use nalgebra::{Const, Matrix1, Matrix4, Matrix4x1, Vector4};
use plotters::prelude::*;
use std::error::Error;

extern crate robotics;
use robotics::control::lqr::{lqr, LinearTimeInvariantModel};

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

    fn linearize(&self, dt: f64) -> LinearTimeInvariantModel<f64, Const<4>, Const<1>> {
        LinearTimeInvariantModel {
            A: Matrix4::identity() + dt * self.da,
            B: dt * self.db,
            Q: self.q.clone(),
            R: self.r.clone(),
        }
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

    let model = InvertedPendulumModel::new(l_bar, mass_cart, mass_ball, g);
    let linear_model = model.linearize(dt);
    let k_lqr = lqr(&linear_model, max_iter, epsilon)?;

    let mut x = Vector4::new(0.0, 0.0, -0.2, 0.0);
    let x_goal = Vector4::new(0.0, 0.0, 0.0, 0.0);

    let mut states = vec![x];
    let mut commands = vec![0.0];

    while time < sim_time {
        time += dt;
        let u = -k_lqr * (&x - &x_goal);
        x = &linear_model.A * x + &linear_model.B * u;

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
    let (states, _commands) = run().expect("unable to run");

    // Create output directory if it didnt exist
    std::fs::create_dir_all("./img")?;

    let path_prefix = match algo_idx {
        0 => "lqr",
        1 => "mpc",
        _ => unreachable!(),
    };

    let full_path = format!("img/{}-{}.svg", path_prefix, "done");

    let root = SVGBackend::new(&full_path, (1000, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let time: Vec<f64> = (0..states.len())
        .map(|i| (i as f64) * 0.01) // dt = 0.01
        .collect();

    let mut min_y = states.iter().map(|s| s.min()).fold(f64::INFINITY, f64::min);
    let mut max_y = states
        .iter()
        .map(|s| s.max())
        .fold(f64::NEG_INFINITY, f64::max);

    min_y -= 0.1;
    max_y += 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Inverted Pendulum - {}", algos[algo_idx]),
            ("sans-serif", 40),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..5f64, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc("Time [s]")
        .y_desc("State Value")
        .draw()?;

    let state_configs = vec![
        (states.iter().map(|s| s.x).collect::<Vec<_>>(), &BLUE, "x"),
        (
            states.iter().map(|s| s.y).collect::<Vec<_>>(),
            &CYAN,
            "x dot",
        ),
        (
            states.iter().map(|s| s.z).collect::<Vec<_>>(),
            &RED,
            "theta",
        ),
        (
            states.iter().map(|s| s.w).collect::<Vec<_>>(),
            &GREEN,
            "theta dot",
        ),
    ];

    for (data, color, label) in state_configs {
        chart
            .draw_series(LineSeries::new(
                time.iter().zip(data.iter()).map(|(&t, &v)| (t, v)),
                color.stroke_width(2),
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
