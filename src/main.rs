use std::error::Error;

use robotics::mapping::pose_graph_slam::PoseGraph;

fn main() -> Result<(), Box<dyn Error>> {
    // let filename = "dataset/g2o/simulation-pose-pose.g2o";
    // let filename = "dataset/g2o/simulation-pose-landmark.g2o";
    // let filename = "dataset/g2o/dlr.g2o";
    let filename = "dataset/g2o/intel.g2o";
    let mut graph = PoseGraph::from_g2o_file(filename)?;
    graph.optimize(50, false)?;
    Ok(())
}
