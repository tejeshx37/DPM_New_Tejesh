//! Triangle tessellation for the wgpu viewport.
//!
//! Produces flat `Vec<Vertex>` lists (no index buffer) so the renderer
//! can do back-to-front sorting cheaply. Each triangle's three vertices
//! share the same face normal — flat shading — which matches the look
//! we want for parametric primitives and a deformed-mesh surface.

use nalgebra::Vector3;

use super::super::shape::Shape3D;
use super::wgpu_scene::Vertex;

const SPHERE_LAT_SEG: usize = 12;
const SPHERE_LON_SEG: usize = 24;
const CYL_SEG: usize = 32;

pub fn triangles_for(shape: &Shape3D, color: [f32; 4]) -> Vec<Vertex> {
    match shape {
        Shape3D::Cube { center, size } => {
            cuboid_triangles(*center, Vector3::new(*size, *size, *size), color)
        }
        Shape3D::Cuboid { center, extents } => cuboid_triangles(*center, *extents, color),
        Shape3D::Sphere { center, radius } => sphere_triangles(*center, *radius, color),
        Shape3D::Cylinder {
            base_center,
            axis,
            radius,
            height,
        } => cylinder_triangles(*base_center, *axis, *radius, *height, color),
    }
}

fn cuboid_triangles(center: Vector3<f64>, extents: Vector3<f64>, color: [f32; 4]) -> Vec<Vertex> {
    let h = extents * 0.5;
    let p = |sx: f64, sy: f64, sz: f64| center + Vector3::new(sx * h.x, sy * h.y, sz * h.z);
    let corners = [
        p(-1.0, -1.0, -1.0), // 0
        p( 1.0, -1.0, -1.0), // 1
        p( 1.0,  1.0, -1.0), // 2
        p(-1.0,  1.0, -1.0), // 3
        p(-1.0, -1.0,  1.0), // 4
        p( 1.0, -1.0,  1.0), // 5
        p( 1.0,  1.0,  1.0), // 6
        p(-1.0,  1.0,  1.0), // 7
    ];

    let mut tris = Vec::with_capacity(36);
    let mut face = |normal: Vector3<f64>, q: [usize; 4]| {
        // Quad q[0..3] CCW when looking along -normal.
        push_tri(&mut tris, corners[q[0]], corners[q[1]], corners[q[2]], normal, color);
        push_tri(&mut tris, corners[q[0]], corners[q[2]], corners[q[3]], normal, color);
    };

    face(Vector3::new(0.0, 0.0, -1.0), [0, 3, 2, 1]); // -Z (back)
    face(Vector3::new(0.0, 0.0,  1.0), [4, 5, 6, 7]); // +Z (front)
    face(Vector3::new(0.0, -1.0, 0.0), [0, 1, 5, 4]); // -Y (bottom)
    face(Vector3::new(0.0,  1.0, 0.0), [3, 7, 6, 2]); // +Y (top)
    face(Vector3::new(-1.0, 0.0, 0.0), [0, 4, 7, 3]); // -X (left)
    face(Vector3::new( 1.0, 0.0, 0.0), [1, 2, 6, 5]); // +X (right)

    tris
}

fn sphere_triangles(center: Vector3<f64>, radius: f64, color: [f32; 4]) -> Vec<Vertex> {
    // Latitude/longitude grid. Caps are degenerate at the poles which we
    // handle as triangle fans.
    let mut grid: Vec<Vec<Vector3<f64>>> = Vec::with_capacity(SPHERE_LAT_SEG + 1);
    for i in 0..=SPHERE_LAT_SEG {
        let phi = std::f64::consts::PI * (i as f64) / (SPHERE_LAT_SEG as f64); // 0..PI
        let (sp, cp) = phi.sin_cos();
        let mut row = Vec::with_capacity(SPHERE_LON_SEG + 1);
        for j in 0..=SPHERE_LON_SEG {
            let theta =
                std::f64::consts::TAU * (j as f64) / (SPHERE_LON_SEG as f64); // 0..2PI
            let (st, ct) = theta.sin_cos();
            row.push(Vector3::new(sp * ct, cp, sp * st) * radius + center);
        }
        grid.push(row);
    }

    let mut tris = Vec::with_capacity(SPHERE_LAT_SEG * SPHERE_LON_SEG * 6);
    for i in 0..SPHERE_LAT_SEG {
        for j in 0..SPHERE_LON_SEG {
            let a = grid[i][j];
            let b = grid[i + 1][j];
            let c = grid[i + 1][j + 1];
            let d = grid[i][j + 1];
            // Outward normals are simply the radial direction.
            let n_abc = ((a - center) + (b - center) + (c - center)).normalize();
            let n_acd = ((a - center) + (c - center) + (d - center)).normalize();
            push_tri(&mut tris, a, b, c, n_abc, color);
            push_tri(&mut tris, a, c, d, n_acd, color);
        }
    }

    tris
}

fn cylinder_triangles(
    base_center: Vector3<f64>,
    axis: Vector3<f64>,
    radius: f64,
    height: f64,
    color: [f32; 4],
) -> Vec<Vertex> {
    let axis_n = if axis.norm() < 1e-9 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        axis.normalize()
    };
    let top_center = base_center + axis_n * height;
    let helper = if axis_n.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = axis_n.cross(&helper).normalize();
    let v = axis_n.cross(&u).normalize();

    let ring = |c: Vector3<f64>| -> Vec<Vector3<f64>> {
        (0..CYL_SEG)
            .map(|i| {
                let t = (i as f64) / (CYL_SEG as f64) * std::f64::consts::TAU;
                c + (u * t.cos() + v * t.sin()) * radius
            })
            .collect()
    };
    let bottom = ring(base_center);
    let top = ring(top_center);
    let radial = |i: usize| {
        let t = (i as f64) / (CYL_SEG as f64) * std::f64::consts::TAU;
        u * t.cos() + v * t.sin()
    };

    let mut tris = Vec::with_capacity(CYL_SEG * 12);
    // Side quads.
    for i in 0..CYL_SEG {
        let j = (i + 1) % CYL_SEG;
        let n_i = radial(i);
        let n_j = radial(j);
        let normal_quad = (n_i + n_j).normalize();
        push_tri(&mut tris, bottom[i], bottom[j], top[j], normal_quad, color);
        push_tri(&mut tris, bottom[i], top[j], top[i], normal_quad, color);
    }
    // Bottom cap (fan, normal = -axis_n).
    let cap_bottom_n = -axis_n;
    for i in 0..CYL_SEG {
        let j = (i + 1) % CYL_SEG;
        push_tri(
            &mut tris,
            base_center,
            bottom[j],
            bottom[i],
            cap_bottom_n,
            color,
        );
    }
    // Top cap (fan, normal = +axis_n).
    let cap_top_n = axis_n;
    for i in 0..CYL_SEG {
        let j = (i + 1) % CYL_SEG;
        push_tri(&mut tris, top_center, top[i], top[j], cap_top_n, color);
    }

    tris
}

/// Surface triangulation of a `Mesh3D` using its `boundary_faces` regions.
/// `vertex_color` supplies the RGBA per source-mesh vertex.
pub fn triangles_for_mesh<F>(mesh: &mesh::d3::Mesh3D, vertex_color: F) -> Vec<Vertex>
where
    F: Fn(usize) -> [f32; 4],
{
    let mut tris =
        Vec::with_capacity(mesh.boundary_faces.regions.iter().map(|r| r.faces.len()).sum::<usize>() * 3);
    for region in &mesh.boundary_faces.regions {
        for f in &region.faces {
            let a = mesh.vertices[f[0]];
            let b = mesh.vertices[f[1]];
            let c = mesh.vertices[f[2]];
            let normal = (b - a).cross(&(c - a)).normalize();
            tris.push(Vertex {
                position: vec3_f32(&a),
                normal: vec3_f32_v(&normal),
                color: vertex_color(f[0]),
            });
            tris.push(Vertex {
                position: vec3_f32(&b),
                normal: vec3_f32_v(&normal),
                color: vertex_color(f[1]),
            });
            tris.push(Vertex {
                position: vec3_f32(&c),
                normal: vec3_f32_v(&normal),
                color: vertex_color(f[2]),
            });
        }
    }
    tris
}

/// Surface triangulation of a deformed mesh whose vertex positions come
/// from the solver (so reference indices stay valid, but worldspace
/// positions are looked up from a separate buffer).
pub fn triangles_for_deformed<F>(
    mesh: &mesh::d3::Mesh3D,
    positions_f32: &[nalgebra::Vector3<f32>],
    vertex_color: F,
) -> Vec<Vertex>
where
    F: Fn(usize) -> [f32; 4],
{
    let mut tris =
        Vec::with_capacity(mesh.boundary_faces.regions.iter().map(|r| r.faces.len()).sum::<usize>() * 3);
    for region in &mesh.boundary_faces.regions {
        for f in &region.faces {
            let a = positions_f32[f[0]];
            let b = positions_f32[f[1]];
            let c = positions_f32[f[2]];
            let normal = (b - a).cross(&(c - a)).normalize();
            tris.push(Vertex {
                position: [a.x, a.y, a.z],
                normal: [normal.x, normal.y, normal.z],
                color: vertex_color(f[0]),
            });
            tris.push(Vertex {
                position: [b.x, b.y, b.z],
                normal: [normal.x, normal.y, normal.z],
                color: vertex_color(f[1]),
            });
            tris.push(Vertex {
                position: [c.x, c.y, c.z],
                normal: [normal.x, normal.y, normal.z],
                color: vertex_color(f[2]),
            });
        }
    }
    tris
}

fn push_tri(
    tris: &mut Vec<Vertex>,
    a: Vector3<f64>,
    b: Vector3<f64>,
    c: Vector3<f64>,
    normal: Vector3<f64>,
    color: [f32; 4],
) {
    let n = [normal.x as f32, normal.y as f32, normal.z as f32];
    tris.push(Vertex {
        position: [a.x as f32, a.y as f32, a.z as f32],
        normal: n,
        color,
    });
    tris.push(Vertex {
        position: [b.x as f32, b.y as f32, b.z as f32],
        normal: n,
        color,
    });
    tris.push(Vertex {
        position: [c.x as f32, c.y as f32, c.z as f32],
        normal: n,
        color,
    });
}

fn vec3_f32(v: &Vector3<f64>) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

fn vec3_f32_v(v: &Vector3<f64>) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_yields_36_vertices() {
        let tris = triangles_for(
            &Shape3D::Cube {
                center: Vector3::zeros(),
                size: 1.0,
            },
            [1.0, 1.0, 1.0, 1.0],
        );
        assert_eq!(tris.len(), 36);
    }

    #[test]
    fn cuboid_yields_36_vertices() {
        let tris = triangles_for(
            &Shape3D::Cuboid {
                center: Vector3::new(1.0, 2.0, 3.0),
                extents: Vector3::new(2.0, 4.0, 6.0),
            },
            [0.5, 0.5, 0.5, 1.0],
        );
        assert_eq!(tris.len(), 36);
    }

    #[test]
    fn sphere_vertex_count_is_multiple_of_three() {
        let tris = triangles_for(
            &Shape3D::Sphere {
                center: Vector3::zeros(),
                radius: 1.0,
            },
            [0.0, 0.0, 1.0, 1.0],
        );
        assert!(!tris.is_empty());
        assert_eq!(tris.len() % 3, 0);
    }

    #[test]
    fn cylinder_vertex_count_is_multiple_of_three() {
        let tris = triangles_for(
            &Shape3D::Cylinder {
                base_center: Vector3::zeros(),
                axis: Vector3::new(0.0, 1.0, 0.0),
                radius: 0.5,
                height: 2.0,
            },
            [1.0, 0.5, 0.0, 1.0],
        );
        assert!(!tris.is_empty());
        assert_eq!(tris.len() % 3, 0);
    }
}
