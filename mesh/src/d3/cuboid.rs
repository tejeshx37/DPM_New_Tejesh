//! Structured tetrahedral mesh for an axis-aligned cuboid.
//!
//! Algorithm: subdivide the cuboid into Nx*Ny*Nz hexahedral voxels and split
//! each voxel into six tetrahedra using the standard Kuhn decomposition.
//! Vertices live on a regular lattice; tet indices are computed in O(1) per
//! voxel without any search. Six tets per voxel is the lowest count that
//! produces a conforming mesh (adjacent voxels share triangular faces).
//!
//! Boundary regions: each of the six cuboid faces is exposed as a named
//! region (`x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`) with the
//! triangulation of the surface vertices for that face. Downstream code
//! uses these for boundary-condition assignment.

use nalgebra::Vector3;

use super::{grading, BoundaryFaces, BoundaryRegion, DensityHint, Mesh3D};

/// Build a tetrahedral mesh for an axis-aligned cuboid centered at `center`
/// with given full `extents`, subdivided into `nx * ny * nz` voxels. Each
/// subdivision count is clamped to at least 1.
///
/// `density_hints` optionally biases lattice spacing near specific
/// world-space locations (see [`grading`]); pass an empty slice for the
/// original uniform behavior.
pub fn generate(
    center: Vector3<f64>,
    extents: Vector3<f64>,
    nx: u32,
    ny: u32,
    nz: u32,
    density_hints: &[DensityHint],
) -> Mesh3D {
    let nx = nx.max(1) as usize;
    let ny = ny.max(1) as usize;
    let nz = nz.max(1) as usize;

    let min = center - extents * 0.5;
    let max = center + extents * 0.5;

    // Per-axis bumps: project each world-space hint onto this axis using
    // its coordinate and radius directly. This is an axis-aligned
    // approximation ("graded slab", not an isolated 3D hotspot) — see
    // `grading` module docs.
    let bumps_for = |axis: usize, lo: f64, hi: f64| -> Vec<(f64, f64, f32)> {
        density_hints
            .iter()
            .map(|h| {
                let center = h.center_world[axis].clamp(lo, hi);
                let sigma = h.radius_world.max(1e-6) * (0.3 + h.falloff as f64);
                (center, sigma, h.multiplier)
            })
            .collect()
    };
    let xs = grading::graded_axis_positions(min.x, max.x, nx, &bumps_for(0, min.x, max.x));
    let ys = grading::graded_axis_positions(min.y, max.y, ny, &bumps_for(1, min.y, max.y));
    let zs = grading::graded_axis_positions(min.z, max.z, nz, &bumps_for(2, min.z, max.z));

    // Vertex grid: (nx+1) * (ny+1) * (nz+1) lattice points.
    let stride_y = nx + 1;
    let stride_z = (nx + 1) * (ny + 1);
    let vertex_index = |i: usize, j: usize, k: usize| -> usize {
        i + j * stride_y + k * stride_z
    };

    let mut vertices = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                vertices.push(Vector3::new(xs[i], ys[j], zs[k]));
            }
        }
    }

    let mut tetrahedra = Vec::with_capacity(nx * ny * nz * 6);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                // Voxel corners in standard order:
                //   0: (i,   j,   k)     1: (i+1, j,   k)
                //   2: (i,   j+1, k)     3: (i+1, j+1, k)
                //   4: (i,   j,   k+1)   5: (i+1, j,   k+1)
                //   6: (i,   j+1, k+1)   7: (i+1, j+1, k+1)
                let c = [
                    vertex_index(i, j, k),
                    vertex_index(i + 1, j, k),
                    vertex_index(i, j + 1, k),
                    vertex_index(i + 1, j + 1, k),
                    vertex_index(i, j, k + 1),
                    vertex_index(i + 1, j, k + 1),
                    vertex_index(i, j + 1, k + 1),
                    vertex_index(i + 1, j + 1, k + 1),
                ];
                // Kuhn decomposition: 6 tetrahedra sharing the 0-7 diagonal.
                tetrahedra.extend_from_slice(&[
                    [c[0], c[1], c[3], c[7]],
                    [c[0], c[3], c[2], c[7]],
                    [c[0], c[2], c[6], c[7]],
                    [c[0], c[6], c[4], c[7]],
                    [c[0], c[4], c[5], c[7]],
                    [c[0], c[5], c[1], c[7]],
                ]);
            }
        }
    }

    let boundary_faces = build_boundary_faces(nx, ny, nz, vertex_index);

    Mesh3D {
        vertices,
        tetrahedra,
        boundary_faces,
    }
}

fn build_boundary_faces(
    nx: usize,
    ny: usize,
    nz: usize,
    vertex_index: impl Fn(usize, usize, usize) -> usize,
) -> BoundaryFaces {
    let mut regions = Vec::with_capacity(6);

    // Each cuboid face: emit two triangles per cell on that face. Vertex
    // orientation isn't critical for the DPM solver (it doesn't use face
    // normals), but we keep it outward-consistent for future surface
    // rendering.
    let mut add_face = |name: &str, faces: Vec<[usize; 3]>| {
        let mut verts: Vec<usize> = faces.iter().flat_map(|t| t.iter().copied()).collect();
        verts.sort_unstable();
        verts.dedup();
        regions.push(BoundaryRegion {
            name: name.to_string(),
            faces,
            vertices: verts,
        });
    };

    // x_min (i = 0)
    let mut faces = Vec::with_capacity(ny * nz * 2);
    for k in 0..nz {
        for j in 0..ny {
            let a = vertex_index(0, j, k);
            let b = vertex_index(0, j + 1, k);
            let c = vertex_index(0, j, k + 1);
            let d = vertex_index(0, j + 1, k + 1);
            faces.push([a, c, d]);
            faces.push([a, d, b]);
        }
    }
    add_face("x_min", faces);

    // x_max (i = nx)
    let mut faces = Vec::with_capacity(ny * nz * 2);
    for k in 0..nz {
        for j in 0..ny {
            let a = vertex_index(nx, j, k);
            let b = vertex_index(nx, j + 1, k);
            let c = vertex_index(nx, j, k + 1);
            let d = vertex_index(nx, j + 1, k + 1);
            faces.push([a, b, d]);
            faces.push([a, d, c]);
        }
    }
    add_face("x_max", faces);

    // y_min (j = 0)
    let mut faces = Vec::with_capacity(nx * nz * 2);
    for k in 0..nz {
        for i in 0..nx {
            let a = vertex_index(i, 0, k);
            let b = vertex_index(i + 1, 0, k);
            let c = vertex_index(i, 0, k + 1);
            let d = vertex_index(i + 1, 0, k + 1);
            faces.push([a, b, d]);
            faces.push([a, d, c]);
        }
    }
    add_face("y_min", faces);

    // y_max (j = ny)
    let mut faces = Vec::with_capacity(nx * nz * 2);
    for k in 0..nz {
        for i in 0..nx {
            let a = vertex_index(i, ny, k);
            let b = vertex_index(i + 1, ny, k);
            let c = vertex_index(i, ny, k + 1);
            let d = vertex_index(i + 1, ny, k + 1);
            faces.push([a, d, b]);
            faces.push([a, c, d]);
        }
    }
    add_face("y_max", faces);

    // z_min (k = 0)
    let mut faces = Vec::with_capacity(nx * ny * 2);
    for j in 0..ny {
        for i in 0..nx {
            let a = vertex_index(i, j, 0);
            let b = vertex_index(i + 1, j, 0);
            let c = vertex_index(i, j + 1, 0);
            let d = vertex_index(i + 1, j + 1, 0);
            faces.push([a, d, b]);
            faces.push([a, c, d]);
        }
    }
    add_face("z_min", faces);

    // z_max (k = nz)
    let mut faces = Vec::with_capacity(nx * ny * 2);
    for j in 0..ny {
        for i in 0..nx {
            let a = vertex_index(i, j, nz);
            let b = vertex_index(i + 1, j, nz);
            let c = vertex_index(i, j + 1, nz);
            let d = vertex_index(i + 1, j + 1, nz);
            faces.push([a, b, d]);
            faces.push([a, d, c]);
        }
    }
    add_face("z_max", faces);

    BoundaryFaces { regions }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_and_tet_counts_are_correct() {
        let m = generate(Vector3::zeros(), Vector3::new(1.0, 1.0, 1.0), 2, 3, 4, &[]);
        assert_eq!(m.vertices.len(), 3 * 4 * 5);
        assert_eq!(m.tetrahedra.len(), 2 * 3 * 4 * 6);
        // Six boundary regions.
        assert_eq!(m.boundary_faces.regions.len(), 6);
    }

    #[test]
    fn all_tet_indices_are_in_range() {
        let m = generate(Vector3::zeros(), Vector3::new(2.0, 2.0, 2.0), 3, 3, 3, &[]);
        let n = m.vertices.len();
        for tet in &m.tetrahedra {
            for &v in tet {
                assert!(v < n, "tet index {v} out of range {n}");
            }
        }
    }

    #[test]
    fn boundary_vertex_counts_match_face_grids() {
        let m = generate(Vector3::zeros(), Vector3::new(1.0, 1.0, 1.0), 2, 3, 4, &[]);
        let by_name = |n: &str| {
            m.boundary_faces
                .regions
                .iter()
                .find(|r| r.name == n)
                .unwrap()
        };
        // Faces normal to x have (ny+1)*(nz+1) = 4*5 = 20 unique vertices.
        assert_eq!(by_name("x_min").vertices.len(), 4 * 5);
        assert_eq!(by_name("x_max").vertices.len(), 4 * 5);
        assert_eq!(by_name("y_min").vertices.len(), 3 * 5);
        assert_eq!(by_name("y_max").vertices.len(), 3 * 5);
        assert_eq!(by_name("z_min").vertices.len(), 3 * 4);
        assert_eq!(by_name("z_max").vertices.len(), 3 * 4);
    }
}
