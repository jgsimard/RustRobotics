use std::error::Error;

use robotics::mapping::graph_slam::{read_g2o_file, Graph};

fn main() -> Result<(), Box<dyn Error>> {
    // let filename = "dataset/new_slam_course/simulation-pose-pose.g2o";
    // let filename = "dataset/new_slam_course/simulation-pose-landmark.g2o";
    // let filename = "dataset/new_slam_course/dlr.g2o";
    let filename = "dataset/new_slam_course/intel.g2o";
    let mut graph = read_g2o_file(filename)?;
    graph.run(5);
    Ok(())
}
