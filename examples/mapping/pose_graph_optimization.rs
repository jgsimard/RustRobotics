use dialoguer::{theme::ColorfulTheme, Select};
use std::error::Error;

use robotics::mapping::{PoseGraph, PoseGraphSolver};

fn main() -> Result<(), Box<dyn Error>> {
    // Create output directory if it didnt exist
    std::fs::create_dir_all("./img")?;

    let filenames = &[
        "simulation-pose-pose.g2o",
        "simulation-pose-landmark.g2o",
        "dlr.g2o",
        "intel.g2o",
        "input_M3500_g2o.g2o",
        "sphere2500.g2o",
        "torus3D.g2o",
        "parking-garage.g2o",
    ];
    let filename_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Pick g2o file")
        .default(0)
        .items(&filenames[..])
        .interact()
        .unwrap();
    let filename = format!("dataset/g2o/{}", filenames[filename_idx]);

    let solvers = &["GaussNewton", "LevenbergMarquardt"];
    let solver_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Pick g2o file")
        .default(0)
        .items(&solvers[..])
        .interact()
        .unwrap();
    let solver = match solver_idx {
        0 => PoseGraphSolver::GaussNewton,
        1 => PoseGraphSolver::LevenbergMarquardt,
        _ => unreachable!(),
    };

    let plot = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Plot the resut?")
        .default(0)
        .items(&[true, false])
        .interact()
        .unwrap();
    let plot = plot == 0;

    let mut graph = PoseGraph::new(filename.as_str(), solver)?;
    graph.optimize(50, true, plot)?;
    Ok(())
}
