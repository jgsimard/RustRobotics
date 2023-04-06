use csv;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RangeBearing {
    pub time: f64,
    pub subject_nb: u32,
    pub range: f64,
    pub bearing: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Position {
    pub time: f64,
    pub x: f64,
    pub y: f64,
    pub orientation: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Odometry {
    pub time: f64,
    pub forward_velocity: f64,
    pub angular_velocity: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Landmark {
    pub subject_nb: u32,
    pub x: f64,
    pub y: f64,
    pub x_std_dev: f64,
    pub y_std_dev: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Barcode {
    subject_nb: u32,
    barcode_nb: u32,
}

pub struct UtiasDataset {
    pub groundtruth: Vec<Position>,
    pub landmarks: FxHashMap<u32, Landmark>,
    pub measurements: Vec<RangeBearing>,
    pub odometry: Vec<Odometry>,
}

// iterator
// Consuming iterator
// TODO : remove the clonessubject_nb
pub struct UtiasDatasetIterator {
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

        let me_next = me.next()?;
        let od_next = od.next()?;

        if od_next.time < me_next.time {
            self.index_odometry += 1;
            return Some((None, Some(od_next.clone())));
        }
        let mut measurements = vec![me_next.clone()];
        self.index_measurements += 1;
        loop {
            let me_next = me.next().unwrap();
            if me_next.time == measurements.last()?.time {
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
pub struct UtiasDatasetIteratorRef<'a> {
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

        let me_next = me.next()?;
        let od_next = od.next()?;

        if od_next.time < me_next.time {
            self.index_odometry += 1;
            return Some((None, Some(od_next)));
        }
        let mut measurements = vec![me_next];
        self.index_measurements += 1;
        loop {
            let me_next = me.next()?;
            if me_next.time == measurements.last()?.time {
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
    pub fn new(dataset: i32) -> Result<UtiasDataset, Box<dyn Error>> {
        let base = match dataset {
            0 => "dataset/utias0",
            1 => "dataset/utias1",
            _ => panic!("dataset {dataset} not supported"),
        };

        let barcodes: Vec<Barcode> = csv::Reader::from_path(format!("{base}/Barcodes.csv"))?
            .deserialize()
            .map(|x| x.unwrap())
            .collect();

        let landmarks_vec: Vec<Landmark> =
            csv::Reader::from_path(format!("{base}/Landmark_Groundtruth.csv"))?
                .deserialize()
                .map(|x| x.unwrap())
                .collect();

        let mut landmarks = FxHashMap::default();
        for lm in landmarks_vec {
            let k = barcodes
                .iter()
                .find(|bc| bc.subject_nb == lm.subject_nb)
                .unwrap()
                .barcode_nb;
            landmarks.insert(k, lm);
        }

        let mut groundtruth: Vec<Position> =
            csv::Reader::from_path(format!("{base}/Groundtruth.csv"))?
                .deserialize()
                .map(|x| x.unwrap())
                .collect();
        groundtruth.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        let min_time = groundtruth.first().unwrap().time;

        let mut measurements: Vec<RangeBearing> =
            csv::Reader::from_path(format!("{base}/Measurement.csv"))?
                .deserialize()
                .map(|x| x.unwrap())
                // .filter(|rb: &RangeBearing| (rb.range != 0.0) | (rb.bearing != 0.0))
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
            groundtruth,
            landmarks,
            measurements,
            odometry,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_dataset0() -> Result<(), Box<dyn Error>> {
        let dataset = UtiasDataset::new(0)?;
        for (i, data) in (&dataset).into_iter().take(5).enumerate() {
            println!("{i} => {:?}", data);
        }
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
