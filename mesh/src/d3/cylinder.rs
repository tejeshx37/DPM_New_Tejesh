//! Structured tetrahedral mesh for a finite cylinder.
//!
//! Algorithm: structured `(r, θ, z)` grid. The axial slabs are stacked
//! along the cylinder's axis; each θ-wedge of a slab is split into prism
//! cells (radial × axial), and each prism becomes 3 tetrahedra. The
//! central axis is handled with a degenerate "fan" of tetrahedra meeting
//! at a single axial vertex per z-level, avoiding a sliver collapse.
//!
//! Boundary regions: `side` (lateral surface), `bottom` (base disk), and
//! `top` (top disk).

use nalgebra::Vector3;

use super::{grading, BoundaryFaces, BoundaryRegion, DensityHint, Mesh3D};

/// Generate a tet mesh for a cylinder.
///
/// - `base_center`: world-space center of the base disk.
/// - `axis`: axis direction (need not be unit length; zero vector falls back
///   to +Z to avoid panics).
/// - `radius`, `height`: positive scalars.
/// - `radial`: number of radial cells from axis to lateral surface (≥ 1).
/// - `circumferential`: number of θ wedges (≥ 3).
/// - `axial`: number of axial slabs (≥ 1).
/// - `density_hints`: optional world-space local density bias. Only the
///   radial and axial spacing are graded — circumferential (θ) spacing
///   stays uniform, a v1 simplification (see `grading` module docs).
pub fn generate(
    base_center: Vector3<f64>,
    axis: Vector3<f64>,
    radius: f64,
    height: f64,
    radial: u32,
    circumferential: u32,
    axial: u32,
    density_hints: &[DensityHint],
) -> Mesh3D {
    let nr = radial.max(1) as usize;
    let nt = circumferential.max(3) as usize;
    let nz = axial.max(1) as usize;

    let axis_n = if axis.norm() < 1e-9 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        axis.normalize()
    };
    let (u, v) = orthonormal_basis(axis_n);

    // Project each world-space hint onto the axial coordinate (distance
    // along `axis_n` from `base_center`) and the radial coordinate
    // (distance from the axis line), to grade the z and r spacing
    // respectively.
    let axial_bumps: Vec<(f64, f64, f32)> = density_hints
        .iter()
        .map(|h| {
            let rel = h.center_world - base_center;
            let z = rel.dot(&axis_n).clamp(0.0, height);
            let sigma = h.radius_world.max(1e-6) * (0.3 + h.falloff as f64);
            (z, sigma, h.multiplier)
        })
        .collect();
    let radial_bumps: Vec<(f64, f64, f32)> = density_hints
        .iter()
        .map(|h| {
            let rel = h.center_world - base_center;
            let z = rel.dot(&axis_n);
            let radial_component = rel - axis_n * z;
            let r = radial_component.norm().clamp(0.0, radius);
            let sigma = h.radius_world.max(1e-6) * (0.3 + h.falloff as f64);
            (r, sigma, h.multiplier)
        })
        .collect();
    let z_levels = grading::graded_axis_positions(0.0, height, nz, &axial_bumps);
    let r_levels = grading::graded_axis_positions(0.0, radius, nr, &radial_bumps);

    // Vertex layout: per axial level k = 0..=nz, one axis vertex followed by
    // (nr * nt) ring vertices. Axis vertex is at radial=0 (shared by all θ).
    let per_level = 1 + nr * nt;
    let mut vertices = Vec::with_capacity(per_level * (nz + 1));
    for k in 0..=nz {
        let z = z_levels[k];
        let center = base_center + axis_n * z;
        vertices.push(center);
        for ir in 1..=nr {
            let r = r_levels[ir];
            for it in 0..nt {
                let theta = (it as f64) / (nt as f64) * std::f64::consts::TAU;
                vertices.push(center + (u * theta.cos() + v * theta.sin()) * r);
            }
        }
    }

    let axis_idx = |k: usize| k * per_level;
    let ring_idx = |k: usize, ir: usize, it: usize| -> usize {
        // ir is 1-based (1..=nr); it is 0..nt.
        k * per_level + 1 + (ir - 1) * nt + (it % nt)
    };

    let mut tetrahedra = Vec::with_capacity(nz * nt * (3 * nr - 2));
    for k in 0..nz {
        for it in 0..nt {
            let it1 = (it + 1) % nt;
            // Innermost wedge meets the axis: 1 tet of corner indices
            //   bottom axis, bottom ring(it), bottom ring(it+1), top axis
            // plus 2 tets connecting top ring.
            // We use the standard prism→3-tet split with the axis vertex
            // acting as both interior corners at radial=0.
            let b_axis = axis_idx(k);
            let t_axis = axis_idx(k + 1);

            // Inner shell (ir=1) is a triangular prism with one edge
            // collapsed onto the axis. Decompose as 3 tets:
            let b_a = ring_idx(k, 1, it);
            let b_b = ring_idx(k, 1, it1);
            let t_a = ring_idx(k + 1, 1, it);
            let t_b = ring_idx(k + 1, 1, it1);
            tetrahedra.push([b_axis, b_a, b_b, t_axis]);
            tetrahedra.push([b_a, b_b, t_b, t_axis]);
            tetrahedra.push([b_a, t_b, t_a, t_axis]);

            // Outer shells (ir = 2..=nr): each is a hexahedral prism cell
            // between two rings, split into 6 tetrahedra via Kuhn.
            for ir in 2..=nr {
                let c = [
                    ring_idx(k, ir - 1, it),
                    ring_idx(k, ir - 1, it1),
                    ring_idx(k, ir, it),
                    ring_idx(k, ir, it1),
                    ring_idx(k + 1, ir - 1, it),
                    ring_idx(k + 1, ir - 1, it1),
                    ring_idx(k + 1, ir, it),
                    ring_idx(k + 1, ir, it1),
                ];
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

    let boundary_faces = build_boundary_faces(nr, nt, nz, axis_idx, ring_idx);

    Mesh3D {
        vertices,
        tetrahedra,
        boundary_faces,
    }
}

fn orthonormal_basis(axis: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let helper = if axis.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = axis.cross(&helper).normalize();
    let v = axis.cross(&u).normalize();
    (u, v)
}

fn build_boundary_faces(
    nr: usize,
    nt: usize,
    nz: usize,
    axis_idx: impl Fn(usize) -> usize,
    ring_idx: impl Fn(usize, usize, usize) -> usize,
) -> BoundaryFaces {
    let mut regions = Vec::with_capacity(3);

    // Lateral surface: outermost ring at every (k, k+1) slab pair.
    let mut side_faces = Vec::with_capacity(nt * nz * 2);
    for k in 0..nz {
        for it in 0..nt {
            let it1 = (it + 1) % nt;
            let bl = ring_idx(k, nr, it);
            let br = ring_idx(k, nr, it1);
            let tl = ring_idx(k + 1, nr, it);
            let tr = ring_idx(k + 1, nr, it1);
            side_faces.push([bl, br, tr]);
            side_faces.push([bl, tr, tl]);
        }
    }
    regions.push(make_region("side", side_faces));

    // Bottom disk (k = 0): fans of triangles emanating from axis vertex.
    let mut bottom_faces = Vec::with_capacity(nt * (nr * 2 - 1));
    for it in 0..nt {
        let it1 = (it + 1) % nt;
        // Innermost triangle to axis.
        bottom_faces.push([axis_idx(0), ring_idx(0, 1, it1), ring_idx(0, 1, it)]);
        // Outer annulus quads → 2 triangles each.
        for ir in 2..=nr {
            let a = ring_idx(0, ir - 1, it);
            let b = ring_idx(0, ir - 1, it1);
            let c = ring_idx(0, ir, it);
            let d = ring_idx(0, ir, it1);
            bottom_faces.push([a, d, b]);
            bottom_faces.push([a, c, d]);
        }
    }
    regions.push(make_region("bottom", bottom_faces));

    // Top disk (k = nz): same as bottom but reversed orientation.
    let mut top_faces = Vec::with_capacity(nt * (nr * 2 - 1));
    for it in 0..nt {
        let it1 = (it + 1) % nt;
        top_faces.push([axis_idx(nz), ring_idx(nz, 1, it), ring_idx(nz, 1, it1)]);
        for ir in 2..=nr {
            let a = ring_idx(nz, ir - 1, it);
            let b = ring_idx(nz, ir - 1, it1);
            let c = ring_idx(nz, ir, it);
            let d = ring_idx(nz, ir, it1);
            top_faces.push([a, b, d]);
            top_faces.push([a, d, c]);
        }
    }
    regions.push(make_region("top", top_faces));

    BoundaryFaces { regions }
}

fn make_region(name: &str, faces: Vec<[usize; 3]>) -> BoundaryRegion {
    let mut verts: Vec<usize> = faces.iter().flat_map(|t| t.iter().copied()).collect();
    verts.sort_unstable();
    verts.dedup();
    BoundaryRegion {
        name: name.to_string(),
        faces,
        vertices: verts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_match_grid() {
        let m = generate(
            Vector3::zeros(),
            Vector3::new(0.0, 0.0, 1.0),
            1.0,
            2.0,
            2, // nr
            8, // nt
            3, // nz
            &[],
        );
        // per-level vertex count = 1 + nr*nt = 17, total = 17 * (nz+1) = 68.
        assert_eq!(m.vertices.len(), 17 * 4);
        // per slab per wedge = 3 (inner) + 6 * (nr - 1) outer = 9 tets.
        // total = nz * nt * 9 = 3 * 8 * 9 = 216.
        assert_eq!(m.tetrahedra.len(), 3 * 8 * 9);
        assert_eq!(m.boundary_faces.regions.len(), 3);
    }

    #[test]
    fn all_indices_in_range() {
        let m = generate(Vector3::zeros(), Vector3::new(0.0, 0.0, 1.0), 1.0, 1.0, 3, 12, 4, &[]);
        let n = m.vertices.len();
        for tet in &m.tetrahedra {
            for &v in tet {
                assert!(v < n);
            }
        }
        for region in &m.boundary_faces.regions {
            for f in &region.faces {
                for &v in f {
                    assert!(v < n);
                }
            }
        }
    }

    #[test]
    fn side_region_excludes_axis_and_caps() {
        let m = generate(Vector3::zeros(), Vector3::new(0.0, 0.0, 1.0), 1.0, 1.0, 2, 8, 2, &[]);
        let side = m
            .boundary_faces
            .regions
            .iter()
            .find(|r| r.name == "side")
            .unwrap();
        // Side has only outermost ring × (nz+1) levels = nt * (nz+1) = 24.
        assert_eq!(side.vertices.len(), 8 * 3);
    }
}
