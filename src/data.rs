use csv;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct RangeBearing {
    time: f64,
    subject_nb: u32,
    range: f64,
    bearing: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Position {
    time: f64,
    x: f64,
    y: f64,
    orientation: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Odometry {
    time: f64,
    forward_velocity: f64,
    angular_velocity: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Landmark {
    subject_nb: u32,
    x: f64,
    y: f64,
    x_std_dev: f64,
    y_std_dev: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Barcode {
    subject_nb: u32,
    barcode_nb: u32,
}

// #[allow(dead_code)]
struct UtiasDataset {
    #[allow(dead_code)]
    pub barcodes: Vec<Barcode>,
    pub groundtruth: Vec<Position>,
    pub landmarks: Vec<Landmark>,
    pub measurements: Vec<RangeBearing>,
    pub odometry: Vec<Odometry>,
    // index_measurements: usize,
    // index_odometry: usize,
}

// iterator
// Consuming iterator
// TODO : remove the clones
struct UtiasDatasetIterator {
    dataset: UtiasDataset,
    index_measurements: usize,
    index_odometry: usize,
}

impl IntoIterator for UtiasDataset {
    type IntoIter = UtiasDatasetIterator;
    type Item = (Option<Vec<RangeBearing>>, Option<Odometry>);

    fn into_iter(self) -> Self::IntoIter {
        UtiasDatasetIterator {
            dataset: self,
            index_measurements: 0,
            index_odometry: 0,
        }
    }
}

impl Iterator for UtiasDatasetIterator {
    type Item = (Option<Vec<RangeBearing>>, Option<Odometry>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut me = self
            .dataset
            .measurements
            .iter()
            .skip(self.index_measurements);
        let mut od = self.dataset.odometry.iter().skip(self.index_odometry);

        let me_next = me.next().unwrap();
        let od_next = od.next().unwrap();

        if od_next.time < me_next.time {
            self.index_odometry += 1;
            return Some((None, Some(od_next.clone())));
        }
        let mut measurements = vec![me_next.clone()];
        self.index_measurements += 1;
        loop {
            let me_next = me.next().unwrap();
            if me_next.time == measurements.last().unwrap().time {
                measurements.push(me_next.clone());
                self.index_measurements += 1;
            } else {
                break;
            }
        }
        if od_next.time == me_next.time {
            return Some((Some(measurements), Some(od_next.clone())));
        }
        Some((Some(measurements), None))
    }
}

// Non-Consuming iterator
struct UtiasDatasetIteratorRef<'a> {
    dataset: &'a UtiasDataset,
    index_measurements: usize,
    index_odometry: usize,
}

impl<'a> IntoIterator for &'a UtiasDataset {
    type IntoIter = UtiasDatasetIteratorRef<'a>;
    type Item = (Option<Vec<&'a RangeBearing>>, Option<&'a Odometry>);

    fn into_iter(self) -> Self::IntoIter {
        UtiasDatasetIteratorRef {
            dataset: self,
            index_measurements: 0,
            index_odometry: 0,
        }
    }
}

impl<'a> Iterator for UtiasDatasetIteratorRef<'a> {
    type Item = (Option<Vec<&'a RangeBearing>>, Option<&'a Odometry>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut me = self
            .dataset
            .measurements
            .iter()
            .skip(self.index_measurements);
        let mut od = self.dataset.odometry.iter().skip(self.index_odometry);

        let me_next = me.next().unwrap();
        let od_next = od.next().unwrap();

        if od_next.time < me_next.time {
            self.index_odometry += 1;
            return Some((None, Some(od_next)));
        }
        let mut measurements = vec![me_next];
        self.index_measurements += 1;
        loop {
            let me_next = me.next().unwrap();
            if me_next.time == measurements.last().unwrap().time {
                measurements.push(me_next);
                self.index_measurements += 1;
            } else {
                break;
            }
        }
        if od_next.time == me_next.time {
            return Some((Some(measurements), Some(od_next)));
        }
        Some((Some(measurements), None))
    }
}

impl UtiasDataset {
    #[allow(dead_code)]
    fn new(dataset: i32) -> Result<UtiasDataset, Box<dyn Error>> {
        let base = match dataset {
            0 => "dataset/dataset0",
            1 => "dataset/dataset1",
            _ => panic!("dataset {dataset} not supported"),
        };

        let barcodes: Vec<Barcode> = csv::Reader::from_path(format!("{base}/Barcodes.csv"))?
            .deserialize()
            .map(|x| x.unwrap())
            .collect();

        let landmarks: Vec<Landmark> =
            csv::Reader::from_path(format!("{base}/Landmark_Groundtruth.csv"))?
                .deserialize()
                .map(|x| x.unwrap())
                .collect();

        let groundtruth: Vec<Position> = csv::Reader::from_path(format!("{base}/Groundtruth.csv"))?
            .deserialize()
            .map(|x| x.unwrap())
            .collect();

        let min_time = groundtruth.iter().map(|p| p.time).reduce(f64::min).unwrap();

        let mut measurements: Vec<RangeBearing> =
            csv::Reader::from_path(format!("{base}/Measurement.csv"))?
                .deserialize()
                .map(|x| x.unwrap())
                .filter(|rb: &RangeBearing| (rb.range != 0.0) | (rb.bearing != 0.0))
                .filter(|rb: &RangeBearing| rb.time >= min_time)
                .collect();

        measurements.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

        let mut odometry: Vec<Odometry> = csv::Reader::from_path(format!("{base}/Odometry.csv"))?
            .deserialize()
            .map(|x| x.unwrap())
            .filter(|od: &Odometry| od.time >= min_time)
            .collect();
        odometry.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

        Ok(UtiasDataset {
            barcodes,
            groundtruth,
            landmarks,
            measurements,
            odometry,
        })
    }
}

#[allow(dead_code)]
fn plot(dataset: &UtiasDataset) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("./img/ekf_landmarksXXX.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let name = "EKF landmarks";
    let min_x = 0.0;
    let max_x = 5.0;
    let min_y = -6.0;
    let max_y = 5.0;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(name, ("sans-serif", 40))
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(
            dataset
                .groundtruth
                .iter()
                .take(5000)
                .map(|p| Circle::new((p.x, p.y), 1, BLUE.filled())),
        )?
        .label("Ground truth")
        .legend(|(x, y)| Circle::new((x, y), 3, BLUE.filled()));

    chart
        .draw_series(
            dataset
                .landmarks
                .iter()
                .map(|lm| Circle::new((lm.x, lm.y), 5, RED.filled())),
        )?
        .label("Landmarks")
        .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));

    chart.draw_series(dataset.landmarks.iter().map(|lm| {
        Text::new(
            format!("{:?}", lm.subject_nb),
            (lm.x + 0.05, lm.y),
            ("sans-serif", 15),
        )
    }))?;

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .border_style(BLACK)
        .draw()?;

    root.present().expect(
        "Unable to write result to file, please make sure 'img' dir exists under current dir",
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_dataset0() -> Result<(), Box<dyn Error>> {
        let dataset = UtiasDataset::new(0)?;
        println!("{:?}", dataset.barcodes);
        plot(&dataset);
        for (i, data) in (&dataset).into_iter().take(5).enumerate() {
            println!("{i} => {:?}", data);
        }
        plot(&dataset);
        Ok(())
    }

    // #[test]
    // fn read_dataset1() -> Result<(), Box<dyn Error>> {
    //     UtiasDataset::new(1)?;
    //     Ok(())
    // }

    // #[test]
    // #[should_panic]
    // fn read_dataset_bad() {
    //     UtiasDataset::new(-1).unwrap();
    // }
}
