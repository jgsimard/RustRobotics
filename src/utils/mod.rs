pub mod mvn;
// pub mod plot;
pub mod state;

pub fn deg2rad(x: f64) -> f64 {
    const DEG2RAD_FACTOR: f64 = std::f64::consts::PI / 180.0;
    x * DEG2RAD_FACTOR
}

pub fn rad2deg(x: f64) -> f64 {
    const RAD2DEG_FACTOR: f64 = 180.0 / std::f64::consts::PI;
    x * RAD2DEG_FACTOR
}
