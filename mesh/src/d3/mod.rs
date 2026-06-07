//! 3D tetrahedral meshing for the four primitive shapes. The output type
//! `Mesh3D` is dimension-symmetric with the 2D `Mesh` (vertices + element
//! indices) so the DPM solver can consume both with parallel code paths.

pub mod cuboid;

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
}
