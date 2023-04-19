#![allow(non_camel_case_types)]

use nalgebra::{
    Isometry2, Isometry3, Matrix2, Matrix3, Matrix6, Quaternion, Translation2, Translation3,
    UnitComplex, UnitQuaternion, Vector2,
};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::error::Error;

use crate::mapping::pose_graph_optimization::{Edge, EdgeSE2, EdgeSE2_XY, EdgeSE3, Node};

fn iso2(x: f64, y: f64, angle: f64) -> Isometry2<f64> {
    Isometry2::from_parts(Translation2::new(x, y), UnitComplex::from_angle(angle))
}

fn iso3(x: f64, y: f64, z: f64, qx: f64, qy: f64, qz: f64, qw: f64) -> Isometry3<f64> {
    let translation = Translation3::new(x, y, z);
    let rotation = UnitQuaternion::from_quaternion(Quaternion::new(qx, qy, qz, qw));
    Isometry3::from_parts(translation, rotation)
}

#[derive(Serialize, Deserialize, Debug)]
struct VERTEX_SE2 {
    id: u32,
    x: f64,
    y: f64,
    z: f64,
}

#[allow(clippy::type_complexity)]
pub fn parse_g2o(
    filename: &str,
) -> Result<
    (
        usize,
        Vec<Edge<f64>>,
        FxHashMap<u32, usize>,
        FxHashMap<u32, Node>,
    ),
    Box<dyn Error>,
> {
    let mut edges = Vec::new();
    let mut lut = FxHashMap::default();
    let mut nodes = FxHashMap::default();
    let mut offset = 0;

    for line in std::fs::read_to_string(filename)?.lines() {
        let line: Vec<&str> = line.split(' ').filter(|x| !x.is_empty()).collect();
        match line[0] {
            "VERTEX_SE2" => {
                let id = line[1].parse::<u32>()?;
                let x = line[2].parse::<f64>()?;
                let y = line[3].parse::<f64>()?;
                let angle = line[4].parse::<f64>()?;
                nodes.insert(id, Node::SE2(iso2(x, y, angle)));
                lut.insert(id, offset);
                offset += 3;
            }
            "VERTEX_XY" => {
                let id = line[1].parse::<u32>()?;
                let x = line[2].parse::<f64>()?;
                let y = line[3].parse::<f64>()?;
                nodes.insert(id, Node::XY(Vector2::new(x, y)));
                lut.insert(id, offset);
                offset += 2;
            }
            "VERTEX_SE3:QUAT" => {
                let id = line[1].parse::<u32>()?;
                let x = line[2].parse::<f64>()?;
                let y = line[3].parse::<f64>()?;
                let z = line[4].parse::<f64>()?;
                let qx = line[5].parse::<f64>()?;
                let qy = line[6].parse::<f64>()?;
                let qz = line[7].parse::<f64>()?;
                let qw = line[8].parse::<f64>()?;

                nodes.insert(id, Node::SE3(iso3(x, y, z, qx, qy, qz, qw)));
                lut.insert(id, offset);
                offset += 6;
            }
            "EDGE_SE2" => {
                let from = line[1].parse::<u32>()?;
                let to = line[2].parse::<u32>()?;
                let x = line[3].parse::<f64>()?;
                let y = line[4].parse::<f64>()?;
                let angle = line[5].parse::<f64>()?;
                let tri_11 = line[6].parse::<f64>()?;
                let tri_12 = line[7].parse::<f64>()?;
                let tri_13 = line[8].parse::<f64>()?;
                let tri_22 = line[9].parse::<f64>()?;
                let tri_23 = line[10].parse::<f64>()?;
                let tri_33 = line[11].parse::<f64>()?;

                let measurement = iso2(x, y, angle);

                #[rustfmt::skip]
                let information = Matrix3::new(
                    tri_11, tri_12, tri_13,
                    tri_12, tri_22, tri_23,
                    tri_13, tri_23, tri_33
                );
                let edge = Edge::SE2_SE2(EdgeSE2::new(from, to, measurement, information));
                edges.push(edge);
            }
            "EDGE_SE2_XY" => {
                let from = line[1].parse::<u32>()?;
                let to = line[2].parse::<u32>()?;
                let x = line[3].parse::<f64>()?;
                let y = line[4].parse::<f64>()?;
                let tri_11 = line[5].parse::<f64>()?;
                let tri_12 = line[6].parse::<f64>()?;
                let tri_22 = line[7].parse::<f64>()?;

                let measurement = Vector2::new(x, y);

                #[rustfmt::skip]
                let information = Matrix2::new(
                    tri_11, tri_12,
                    tri_12, tri_22
                );
                let edge = Edge::SE2_XY(EdgeSE2_XY::new(from, to, measurement, information));
                edges.push(edge);
            }
            "EDGE_SE3:QUAT" => {
                let from = line[1].parse::<u32>()?;
                let to = line[2].parse::<u32>()?;
                let line: Vec<f64> = line
                    .iter()
                    .skip(3)
                    .map(|x| x.parse::<f64>().unwrap())
                    .collect();
                let x = line[0];
                let y = line[1];
                let z = line[2];
                let qx = line[3];
                let qy = line[4];
                let qz = line[5];
                let qw = line[6];
                let tri_11 = line[7];
                let tri_12 = line[8];
                let tri_13 = line[9];
                let tri_14 = line[10];
                let tri_15 = line[11];
                let tri_16 = line[12];
                let tri_22 = line[13];
                let tri_23 = line[14];
                let tri_24 = line[15];
                let tri_25 = line[16];
                let tri_26 = line[17];
                let tri_33 = line[18];
                let tri_34 = line[19];
                let tri_35 = line[20];
                let tri_36 = line[21];
                let tri_44 = line[22];
                let tri_45 = line[23];
                let tri_46 = line[24];
                let tri_55 = line[25];
                let tri_56 = line[26];
                let tri_66 = line[27];

                let measurement = iso3(x, y, z, qx, qy, qz, qw);

                #[rustfmt::skip]
                let information = Matrix6::new(
                    tri_11, tri_12, tri_13, tri_14, tri_15, tri_16,
                    tri_12, tri_22, tri_23, tri_24, tri_25, tri_26,
                    tri_13, tri_23, tri_33, tri_34, tri_35, tri_36,
                    tri_14, tri_24, tri_34, tri_44, tri_45, tri_46,
                    tri_15, tri_25, tri_35, tri_45, tri_55, tri_56,
                    tri_16, tri_26, tri_36, tri_46, tri_56, tri_66
                );

                let edge = Edge::SE3_SE3(EdgeSE3::new(from, to, measurement, information));
                edges.push(edge);
            }
            _ => unimplemented!("{}", line[0]),
        }
    }

    Ok((offset, edges, lut, nodes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_g2o() -> Result<(), Box<dyn Error>> {
        let filename = "dataset/g2o/simulation-pose-pose.g2o";
        let (len, edges, _lut, nodes) = parse_g2o(filename)?;
        assert_eq!(400, nodes.len());
        assert_eq!(1773, edges.len());
        assert_eq!(1200, len);

        let filename = "dataset/g2o/simulation-pose-landmark.g2o";
        let (len, edges, _lut, nodes) = parse_g2o(filename)?;
        assert_eq!(77, nodes.len());
        assert_eq!(297, edges.len());
        assert_eq!(195, len);

        let filename = "dataset/g2o/intel.g2o";
        let (len, edges, _lut, nodes) = parse_g2o(filename)?;
        assert_eq!(1728, nodes.len());
        assert_eq!(4830, edges.len());
        assert_eq!(5184, len);

        let filename = "dataset/g2o/dlr.g2o";
        let (len, edges, _lut, nodes) = parse_g2o(filename)?;
        assert_eq!(3873, nodes.len());
        assert_eq!(17605, edges.len());
        assert_eq!(11043, len);
        Ok(())
    }
}
