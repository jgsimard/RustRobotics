use plotters::prelude::*;
use polars::prelude::*;
use std::error::Error;

#[allow(dead_code)]
struct UtiasDataset {
    pub barecodes: DataFrame,
    pub groundtruth: DataFrame,
    pub landmark_groundtruth: DataFrame,
    pub data: DataFrame,
}

impl UtiasDataset {
    #[allow(dead_code)]
    fn new(dataset: i32) -> Result<UtiasDataset, Box<dyn Error>> {
        let base = match dataset {
            0 => "dataset/dataset0".to_owned(),
            1 => "dataset/dataset1".to_owned(),
            _ => panic!("dataset {dataset} not supported"),
        };

        let barecodes = CsvReader::from_path(base.clone() + "/Barcodes.csv")?.finish()?;
        let groundtruth = CsvReader::from_path(base.clone() + "/Groundtruth.csv")?.finish()?;
        let landmark_groundtruth =
            CsvReader::from_path(base.clone() + "/Landmark_Groundtruth.csv")?.finish()?;
        let measurement = CsvReader::from_path(base.clone() + "/Measurement.csv")?.finish()?;
        let odometry = CsvReader::from_path(base + "/Odometry.csv")?.finish()?;
        let odometry = odometry
            .lazy()
            .filter(
                col("forward_velocity")
                    .neq(lit(0.0))
                    .or(col("angular_velocity").neq(lit(0.0))),
            )
            .collect()?;

        // Remove all data before the fisrt timestamp of groundtruth
        // Use first groundtruth data as the initial location of the robot
        let min_time: f64 = groundtruth["time"].min().unwrap();

        let data = measurement.join(&odometry, ["time"], ["time"], JoinType::Outer, None)?;
        let data = data
            .lazy()
            .filter(col("time").gt(min_time))
            .sort("time", Default::default())
            .collect()?;

        for i in 0..5 {
            let row = data.get_row(i)?;
            println!("row {i} = {:?}", row);
        }

        Ok(UtiasDataset {
            barecodes,
            groundtruth,
            landmark_groundtruth,
            data,
        })
    }
}

#[allow(dead_code)]
fn plot(dataset: &UtiasDataset) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("./img/ekf_landmarks.png", (1024, 768)).into_drawing_area();
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

    println!("groundtruth \n{}", dataset.groundtruth);

    println!("PATATE!!!!");
    let x: Vec<f64> = dataset
        .groundtruth
        .column("x")?
        .f64()?
        .into_no_null_iter()
        .collect();
    let y: Vec<f64> = dataset
        .groundtruth
        .column("y")?
        .f64()?
        .into_no_null_iter()
        .collect();

    let nb_lm: Vec<i64> = dataset
        .landmark_groundtruth
        .column("subject_nb")?
        .i64()?
        .into_no_null_iter()
        .collect();

    let x_lm: Vec<f64> = dataset
        .landmark_groundtruth
        .column("x")?
        .f64()?
        .into_no_null_iter()
        .collect();

    let y_lm: Vec<f64> = dataset
        .landmark_groundtruth
        .column("y")?
        .f64()?
        .into_no_null_iter()
        .collect();

    chart
        .draw_series(
            x.iter()
                .zip(y.iter())
                .take(5000)
                .map(|(x, y)| Circle::new((*x, *y), 1, BLUE.filled())),
        )?
        .label("Ground truth")
        .legend(|(x, y)| Circle::new((x, y), 3, BLUE.filled()));

    chart
        .draw_series(
            x_lm.iter()
                .zip(y_lm.iter())
                .take(5000)
                .map(|(x, y)| Circle::new((*x, *y), 5, RED.filled())),
        )?
        .label("Landmarks")
        .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));

    chart.draw_series(
        nb_lm
            .iter()
            .zip(x_lm.iter().zip(y_lm.iter()))
            .take(5000)
            .map(|(nb, (x, y))| {
                Text::new(format!("{:?}", nb), (*x + 0.05, *y), ("sans-serif", 15))
            }),
    )?;

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
        UtiasDataset::new(0)?;
        Ok(())
    }

    // #[test]
    // fn read_dataset1() -> Result<(), Box<dyn Error>> {
    //     UtiasDataset::new(1)?;
    //     Ok(())
    // }

    #[test]
    #[should_panic]
    fn read_dataset_bad() {
        UtiasDataset::new(-1).unwrap();
    }
}
