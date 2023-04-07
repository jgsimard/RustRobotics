use std::error::Error;

use robotics::mapping::pose_graph_optimization::PoseGraph;

fn main() -> Result<(), Box<dyn Error>> {
    // Create output directory if it didnt exist
    std::fs::create_dir_all("./img")?;

    let filename = "dataset/g2o/simulation-pose-pose.g2o";
    // let filename = "dataset/g2o/simulation-pose-landmark.g2o";
    // let filename = "dataset/g2o/dlr.g2o";
    // let filename = "dataset/g2o/intel.g2o";

    // let filename = "dataset/g2o/input_M3500_g2o.g2o";
    // let filename = "dataset/g2o/sphere2500.g2o";
    let mut graph = PoseGraph::from_g2o(filename)?;
    graph.optimize(50, false)?;
    Ok(())
}
