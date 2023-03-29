#![allow(dead_code)]
use plotters::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::read_to_string;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Odometry {
    pub rotation1: f64,
    pub translation: f64,
    pub rotation2: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RangeBearing {
    pub id: u32,
    pub range: f64,
    pub bearing: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Landmark {
    pub id: u32,
    pub x: f64,
    pub y: f64,
}
pub struct SlamCourseDataset {
    odometry: Vec<Odometry>,
    sensors: Vec<Vec<RangeBearing>>,
    landmarks: FxHashMap<u32, Landmark>,
}

// TODO: use nom library instead
impl SlamCourseDataset {
    fn new() -> Result<SlamCourseDataset, Box<dyn Error>> {
        let base = "dataset/slam_course";

        let mut odometry = Vec::new();
        let mut sensors = Vec::new();
        let mut sensors_same_timestep = Vec::new();
        for (i, line) in read_to_string(format!("{base}/sensor_data.dat"))?
            .lines()
            .enumerate()
        {
            let line: Vec<&str> = line.split(' ').collect();
            match line[0] {
                "ODOMETRY" => {
                    if i != 0 {
                        sensors.push(sensors_same_timestep.clone());
                    }
                    sensors_same_timestep.clear();
                    let odom = Odometry {
                        rotation1: line[1].parse::<f64>()?,
                        translation: line[2].parse::<f64>()?,
                        rotation2: line[3].parse::<f64>()?,
                    };
                    odometry.push(odom);
                }
                "SENSOR" => {
                    let sensor = RangeBearing {
                        id: line[1].parse::<u32>()?,
                        range: line[2].parse::<f64>()?,
                        bearing: line[3].parse::<f64>()?,
                    };
                    sensors_same_timestep.push(sensor)
                }
                _ => panic!("no"),
            }
        }
        sensors.push(sensors_same_timestep.clone());

        println!("{}, {}", odometry.len(), sensors.len());

        let mut landmarks = FxHashMap::default();
        for line in std::fs::read_to_string(format!("{base}/world.dat"))?.lines() {
            let line: Vec<&str> = line.split(' ').collect();
            let id = line[0].parse::<u32>()?;
            landmarks.insert(
                id,
                Landmark {
                    id,
                    x: line[1].parse::<f64>()?,
                    y: line[2].parse::<f64>()?,
                },
            );
        }

        println!("{:?}", landmarks);

        Ok(SlamCourseDataset {
            odometry,
            sensors,
            landmarks,
        })
    }
}

// Non-Consuming iterator
pub struct SlamCourseDatasetIteratorRef<'a> {
    dataset: &'a SlamCourseDataset,
    index: usize,
}

impl<'a> IntoIterator for &'a SlamCourseDataset {
    type IntoIter = SlamCourseDatasetIteratorRef<'a>;
    type Item = (Option<&'a Vec<RangeBearing>>, Option<&'a Odometry>);

    fn into_iter(self) -> Self::IntoIter {
        SlamCourseDatasetIteratorRef {
            dataset: self,
            index: 0,
        }
    }
}

impl<'a> Iterator for SlamCourseDatasetIteratorRef<'a> {
    type Item = (Option<&'a Vec<RangeBearing>>, Option<&'a Odometry>);
    fn next(&mut self) -> Option<Self::Item> {
        let odometry = &self.dataset.odometry[self.index];
        let measurements = &self.dataset.sensors[self.index];
        self.index += 1;
        Some((Some(measurements), Some(odometry)))
    }
}

/// Visualizes the robot in the map.
///
/// The resulting plot displays the following information:
/// - the landmarks in the map (black +'s)
/// - current robot pose (red)
/// - observations made at this time step (line between robot and landmark)
fn plot(dataset: &SlamCourseDataset) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all("./img")?;
    let filename = "./img/slam_course.png";
    let name = "SLAM course";
    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(name, ("sans-serif", 40))
        .build_cartesian_2d(-2.0..12.0, -2.0..12.0)?;

    chart.configure_mesh().draw()?;

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

    root.present().expect(
        "Unable to write result to file, please make sure 'img' dir exists under current dir",
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_slam_course_dataset() -> Result<(), Box<dyn Error>> {
        let dataset = SlamCourseDataset::new()?;
        plot(&dataset)?;
        Ok(())
    }
}
