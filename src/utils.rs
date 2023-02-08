pub fn deg2rad(x: f32) -> f32 {
    const DEG2RAD_FACTOR: f32 = std::f32::consts::PI / 180.0;
    x * DEG2RAD_FACTOR
}

pub fn rad2deg(x: f32) -> f32 {
    const RAD2DEG_FACTOR: f32 = 180.0 / std::f32::consts::PI;
    x * RAD2DEG_FACTOR
}
