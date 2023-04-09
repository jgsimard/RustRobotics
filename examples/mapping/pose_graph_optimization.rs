use dialoguer::{theme::ColorfulTheme, Select};
use std::error::Error;

use robotics::mapping::pose_graph_optimization::PoseGraph;

fn main() -> Result<(), Box<dyn Error>> {
    // Create output directory if it didnt exist
    std::fs::create_dir_all("./img")?;

    let filenames = &[
        "dataset/g2o/simulation-pose-pose.g2o",
        "dataset/g2o/simulation-pose-landmark.g2o",
        "dataset/g2o/dlr.g2o",
        "dataset/g2o/intel.g2o",
        "dataset/g2o/input_M3500_g2o.g2o",
        "dataset/g2o/sphere2500.g2o",
    ];
    let filename_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Pick g2o file")
        .default(0)
        .items(&filenames[..])
        .interact()
        .unwrap();
    let filename = filenames[filename_idx];

    let plot = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Plot the resut?")
        .default(0)
        .items(&[true, false])
        .interact()
        .unwrap();
    let plot = plot == 0;

    let mut graph = PoseGraph::from_g2o(filename)?;
    graph.optimize(50, true, plot)?;
    Ok(())
}
