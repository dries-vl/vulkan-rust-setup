const PHI: f32 = (1.0 + 2.23606) / 2.0; // 2.236 is sqrt(5)

// len 48
pub fn get_indices() -> Vec<u16> {
    vec![
        11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 0, 5, 9, 1, 11, 4, 5, 10, 2, 11, 7, 6, 10, 1,
        8, 7, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 3, 9, 5, 4, 4, 11, 2, 2, 10, 6, 6, 7, 8, 1,
        9, 8,
    ]
}

fn calculate_uv(x: f32, y: f32, z: f32) -> crate::helper::Vertex {
    let length = (x * x + y * y + z * z).sqrt();
    let u = 0.5 + (z.atan2(x) / (2.0 * std::f32::consts::PI));
    let v = 0.5 - (y / length).asin() / std::f32::consts::PI;

    crate::helper::Vertex {
        pos: [x, y, z, 2.0],
        color: [u, v, 0.0, 1.0]
    }
}

// len 12
pub fn get_vertices() -> Vec<crate::helper::Vertex> {
    vec![
        calculate_uv(-1.0, PHI, 0.0),
        calculate_uv(1.0, PHI, 0.0),
        calculate_uv(-1.0, -PHI, 0.0),
        calculate_uv(1.0, -PHI, 0.0),
        calculate_uv(0.0, -1.0, PHI),
        calculate_uv(0.0, 1.0, PHI),
        calculate_uv(0.0, -1.0, -PHI),
        calculate_uv(0.0, 1.0, -PHI),
        calculate_uv(PHI, 0.0, -1.0),
        calculate_uv(PHI, 0.0, 1.0),
        calculate_uv(-PHI, 0.0, -1.0),
        calculate_uv(-PHI, 0.0, 1.0),
    ]
}