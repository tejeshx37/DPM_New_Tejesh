//! Tetrahedral mesh for a sphere via the spherified-cube map.
//!
//! Algorithm: generate a tet mesh of the unit cube `[-1, 1]^3` with the
//! cuboid mesher, then radially warp each vertex so that the cube's outer
//! surface lands on the sphere of given radius (and interior vertices scale
//! proportionally on inner spherical shells). The warp is monotonic, so no
//! tetrahedra invert; tet quality degrades slightly toward the corners but
//! remains acceptable for DPM particle stencils.
//!
//! Boundary: the six cube faces all become parts of the sphere surface, so
//! the per-face regions from the cuboid mesher are merged into a single
//! `surface` region.

use nalgebra::Vector3;

use super::{cuboid, BoundaryFaces, BoundaryRegion, DensityHint, Mesh3D};

/// Generate a tet mesh for a sphere centered at `center` with given
/// `radius`, derived from an `n*n*n` cube grid (`n >= 1`).
///
/// `density_hints` are given in world space; they're projected into the
/// unit-cube grading space (before the spherify warp below) by re-centering
/// on `center` and scaling by `radius`, since grading has to happen on the
/// undistorted lattice for the equal-integral technique to stay monotonic.
pub fn generate(center: Vector3<f64>, radius: f64, n: u32, density_hints: &[DensityHint]) -> Mesh3D {
    let n = n.max(1);

    let cube_space_hints: Vec<DensityHint> = density_hints
        .iter()
        .map(|h| DensityHint {
            center_world: (h.center_world - center) / radius.max(1e-9),
            radius_world: (h.radius_world / radius.max(1e-9)).max(1e-6),
            multiplier: h.multiplier,
            falloff: h.falloff,
        })
        .collect();

    // Start with the unit cube [-1, 1]^3.
    let mut cube = cuboid::generate(
        Vector3::zeros(),
        Vector3::new(2.0, 2.0, 2.0),
        n,
        n,
        n,
        &cube_space_hints,
    );

    // Spherify each vertex and translate to center.
    for v in &mut cube.vertices {
        let max_comp = v.x.abs().max(v.y.abs()).max(v.z.abs());
        if max_comp < 1e-12 {
            *v = center;
            continue;
        }
        let dir = v.normalize();
        *v = center + dir * (max_comp * radius);
    }

    // Merge the six per-face boundary regions into one `surface` region.
    let mut faces = Vec::new();
    for region in cube.boundary_faces.regions.drain(..) {
        faces.extend(region.faces);
    }
    let mut verts: Vec<usize> = faces.iter().flat_map(|t| t.iter().copied()).collect();
    verts.sort_unstable();
    verts.dedup();

    Mesh3D {
        vertices: cube.vertices,
        tetrahedra: cube.tetrahedra,
        boundary_faces: BoundaryFaces {
            regions: vec![BoundaryRegion {
                name: "surface".to_string(),
                faces,
                vertices: verts,
            }],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_counts_match_cube_grid() {
        let m = generate(Vector3::zeros(), 1.5, 4, &[]);
        // Same vertex/tet count as a 4x4x4 cuboid.
        assert_eq!(m.vertices.len(), 5 * 5 * 5);
        assert_eq!(m.tetrahedra.len(), 4 * 4 * 4 * 6);
        assert_eq!(m.boundary_faces.regions.len(), 1);
    }

    #[test]
    fn surface_vertices_lie_on_the_sphere() {
        let center = Vector3::new(0.5, -1.0, 2.0);
        let radius = 3.0;
        let m = generate(center, radius, 5, &[]);
        let surface = &m.boundary_faces.regions[0];
        for &idx in &surface.vertices {
            let d = (m.vertices[idx] - center).norm();
            assert!(
                (d - radius).abs() < 1e-9,
                "boundary vertex {idx} is at distance {d}, expected {radius}"
            );
        }
    }

    #[test]
    fn interior_vertices_lie_inside_the_sphere() {
        let center = Vector3::zeros();
        let radius = 1.0;
        let m = generate(center, radius, 3, &[]);
        let surface_set: std::collections::HashSet<usize> =
            m.boundary_faces.regions[0].vertices.iter().copied().collect();
        for (idx, v) in m.vertices.iter().enumerate() {
            if surface_set.contains(&idx) {
                continue;
            }
            assert!(
                v.norm() < radius - 1e-12,
                "interior vertex {idx} is at distance {} >= {radius}",
                v.norm()
            );
        }
    }

    #[test]
    fn density_hint_still_lands_surface_on_sphere() {
        let center = Vector3::new(1.0, 2.0, -1.0);
        let radius = 2.0;
        let hint = DensityHint {
            center_world: center + Vector3::new(radius, 0.0, 0.0),
            radius_world: 0.5,
            multiplier: 3.0,
            falloff: 0.5,
        };
        let m = generate(center, radius, 6, &[hint]);
        let surface = &m.boundary_faces.regions[0];
        for &idx in &surface.vertices {
            let d = (m.vertices[idx] - center).norm();
            assert!((d - radius).abs() < 1e-9);
        }
    }
}
