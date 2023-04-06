use std::error::Error;

use robotics::mapping::graph_slam::PoseGraph;

fn main() -> Result<(), Box<dyn Error>> {
    // let filename = "dataset/new_slam_course/simulation-pose-pose.g2o";
    let filename = "dataset/new_slam_course/simulation-pose-landmark.g2o";
    // let filename = "dataset/new_slam_course/dlr.g2o";
    // let filename = "dataset/new_slam_course/intel.g2o";
    let mut graph = PoseGraph::from_g2o_file(filename)?;
    graph.optimize(5)?;
    graph.plot();
    Ok(())
}
