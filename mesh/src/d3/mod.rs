//! 3D tetrahedral meshing for the four primitive shapes. The output type
//! `Mesh3D` is dimension-symmetric with the 2D `Mesh` (vertices + element
//! indices) so the DPM solver can consume both with parallel code paths.

pub mod cuboid;
pub mod cylinder;
pub mod grading;
pub mod sphere;

pub use grading::DensityHint;

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Tetrahedral mesh: a flat vertex buffer plus 4-index tetrahedra. Boundary
/// faces are stored separately so 3D boundary-condition assignment can map
/// shape faces to vertex sets (same idea as `mesh::Mesh::point_id_map`).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Mesh3D {
    pub vertices: Vec<Vector3<f64>>,
    pub tetrahedra: Vec<[usize; 4]>,
    /// Boundary faces grouped by named region (e.g. "x_min", "x_max"). Each
    /// face is a triangle of vertex indices; multiple faces per region are
    /// expected. Empty for stubbed meshers.
    #[serde(default)]
    pub boundary_faces: BoundaryFaces,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BoundaryFaces {
    pub regions: Vec<BoundaryRegion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryRegion {
    pub name: String,
    pub faces: Vec<[usize; 3]>,
    /// Unique vertices touching this boundary region, for fast BC lookup.
    pub vertices: Vec<usize>,
}

impl Mesh3D {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn tet_count(&self) -> usize {
        self.tetrahedra.len()
    }

    /// Stitch multiple meshes into one. Vertex and tet indices are
    /// offset so the merged mesh references its own flat arrays;
    /// boundary regions are renamed `"body{i+1}/{region}"` where `i` is
    /// the **slice position**, not the populated-mesh index.
    ///
    /// This matters when some shapes in the scene aren't meshed yet: a
    /// `None` entry consumes a body number without contributing
    /// regions, so the names stay stable. The Boundary Conditions phase
    /// relies on this — it pre-populates BC HashMap keys as
    /// `body{shape_index+1}/{region}` before any mesh exists, and they
    /// must still match once meshes are generated.
    pub fn combine(meshes: &[Option<&Mesh3D>]) -> Mesh3D {
        let mut out = Mesh3D::default();
        for (i, m_opt) in meshes.iter().enumerate() {
            let Some(m) = m_opt else { continue };
            let v_offset = out.vertices.len();
            out.vertices.extend(m.vertices.iter().copied());
            for tet in &m.tetrahedra {
                out.tetrahedra
                    .push([tet[0] + v_offset, tet[1] + v_offset, tet[2] + v_offset, tet[3] + v_offset]);
            }
            for region in &m.boundary_faces.regions {
                let faces: Vec<[usize; 3]> = region
                    .faces
                    .iter()
                    .map(|f| [f[0] + v_offset, f[1] + v_offset, f[2] + v_offset])
                    .collect();
                let vertices: Vec<usize> =
                    region.vertices.iter().map(|v| v + v_offset).collect();
                out.boundary_faces.regions.push(BoundaryRegion {
                    name: format!("body{}/{}", i + 1, region.name),
                    faces,
                    vertices,
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_with_one_tet() -> Mesh3D {
        Mesh3D {
            vertices: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            tetrahedra: vec![[0, 1, 2, 3]],
            boundary_faces: BoundaryFaces {
                regions: vec![BoundaryRegion {
                    name: "all".to_string(),
                    faces: vec![[0, 1, 2], [0, 2, 3]],
                    vertices: vec![0, 1, 2, 3],
                }],
            },
        }
    }

    #[test]
    fn combine_offsets_indices_and_renames_regions() {
        let a = unit_cube_with_one_tet();
        let b = unit_cube_with_one_tet();
        let m = Mesh3D::combine(&[Some(&a), Some(&b)]);

        assert_eq!(m.vertices.len(), 8);
        assert_eq!(m.tetrahedra.len(), 2);
        // First tet keeps its 0..4 indices; second is offset to 4..8.
        assert_eq!(m.tetrahedra[0], [0, 1, 2, 3]);
        assert_eq!(m.tetrahedra[1], [4, 5, 6, 7]);

        let names: Vec<&str> = m
            .boundary_faces
            .regions
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(names, vec!["body1/all", "body2/all"]);
        assert_eq!(m.boundary_faces.regions[1].vertices, vec![4, 5, 6, 7]);
    }

    /// `None` slots consume a body number without contributing regions.
    /// This guarantee is what lets the BC phase use shape index for
    /// stable region keys (`body{shape_index+1}/...`) even when some
    /// shapes haven't been meshed yet.
    #[test]
    fn combine_skips_none_but_preserves_body_index() {
        let a = unit_cube_with_one_tet();
        let m = Mesh3D::combine(&[None, Some(&a), None]);
        assert_eq!(m.vertices.len(), 4);
        assert_eq!(m.tetrahedra.len(), 1);
        // The only populated mesh was at slot 1 -> body2.
        let names: Vec<&str> = m
            .boundary_faces
            .regions
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(names, vec!["body2/all"]);
    }
}
