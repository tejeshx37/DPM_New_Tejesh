//! CPU-side wireframe tessellation for the preview viewport.
//!
//! Strictly render-only — these are NOT the simulation meshes. The
//! simulation tetrahedral mesh is generated later in Step 3 from the same
//! `Shape3D` parameters by a separate code path.

use nalgebra::Vector3;

use super::super::shape::Shape3D;

/// A 3D line segment in world space.
pub struct Edge {
    pub a: Vector3<f64>,
    pub b: Vector3<f64>,
}

pub fn edges_for(shape: &Shape3D) -> Vec<Edge> {
    match shape {
        Shape3D::Cube { center, size } => {
            cuboid_edges(*center, Vector3::new(*size, *size, *size))
        }
        Shape3D::Cuboid { center, extents } => cuboid_edges(*center, *extents),
        Shape3D::Sphere { center, radius } => sphere_edges(*center, *radius),
        Shape3D::Cylinder {
            base_center,
            axis,
            radius,
            height,
        } => cylinder_edges(*base_center, *axis, *radius, *height),
    }
}

fn cuboid_edges(center: Vector3<f64>, extents: Vector3<f64>) -> Vec<Edge> {
    let h = extents * 0.5;
    let signs = [-1.0, 1.0];
    let mut corners = Vec::with_capacity(8);
    for &sx in &signs {
        for &sy in &signs {
            for &sz in &signs {
                corners.push(center + Vector3::new(sx * h.x, sy * h.y, sz * h.z));
            }
        }
    }
    // Index in corners: bit0=x sign, bit1=y sign, bit2=z sign in the loop above.
    // Loop order: sx outer, sy middle, sz inner → idx = 4*(sx==1) + 2*(sy==1) + (sz==1).
    let e = |i: usize, j: usize| Edge {
        a: corners[i],
        b: corners[j],
    };
    vec![
        // bottom (sy=-1): idx 0,1,4,5
        e(0, 1), e(1, 5), e(5, 4), e(4, 0),
        // top (sy=1): idx 2,3,6,7
        e(2, 3), e(3, 7), e(7, 6), e(6, 2),
        // verticals
        e(0, 2), e(1, 3), e(4, 6), e(5, 7),
    ]
}

fn sphere_edges(center: Vector3<f64>, radius: f64) -> Vec<Edge> {
    // Three orthogonal great circles, 32 segments each.
    const SEG: usize = 32;
    let mut edges = Vec::with_capacity(SEG * 3);
    for plane in 0..3 {
        let mut prev = circle_point(plane, 0.0, radius);
        for i in 1..=SEG {
            let t = (i as f64) / (SEG as f64) * std::f64::consts::TAU;
            let next = circle_point(plane, t, radius);
            edges.push(Edge {
                a: center + prev,
                b: center + next,
            });
            prev = next;
        }
    }
    edges
}

fn circle_point(plane: usize, t: f64, r: f64) -> Vector3<f64> {
    let (c, s) = (t.cos() * r, t.sin() * r);
    match plane {
        0 => Vector3::new(c, s, 0.0), // XY
        1 => Vector3::new(c, 0.0, s), // XZ
        _ => Vector3::new(0.0, c, s), // YZ
    }
}

fn cylinder_edges(
    base_center: Vector3<f64>,
    axis: Vector3<f64>,
    radius: f64,
    height: f64,
) -> Vec<Edge> {
    const SEG: usize = 32;
    let axis_n = axis.normalize();
    let top_center = base_center + axis_n * height;
    // Build an orthonormal basis around the axis.
    let helper = if axis_n.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = axis_n.cross(&helper).normalize();
    let v = axis_n.cross(&u).normalize();

    let ring = |center: Vector3<f64>| -> Vec<Vector3<f64>> {
        (0..SEG)
            .map(|i| {
                let t = (i as f64) / (SEG as f64) * std::f64::consts::TAU;
                center + (u * t.cos() + v * t.sin()) * radius
            })
            .collect()
    };
    let bottom = ring(base_center);
    let top = ring(top_center);

    let mut edges = Vec::with_capacity(SEG * 3);
    for i in 0..SEG {
        let j = (i + 1) % SEG;
        edges.push(Edge { a: bottom[i], b: bottom[j] });
        edges.push(Edge { a: top[i], b: top[j] });
        edges.push(Edge { a: bottom[i], b: top[i] });
    }
    edges
}
