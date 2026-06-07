//! 3D tetrahedral particle stencil.
//!
//! Each `Element3D` references four node indices and caches the inverse
//! of its reference edge matrix `R = [r_ba | r_ca | r_da]`. With `R^{-1}`
//! precomputed:
//!   - deformation gradient F = D · R^{-1} where D = [d_ba | d_ca | d_da]
//!     uses current node positions
//!   - small-strain symmetric tensor ε = (F + F^T)/2 − I
//!   - shape-function gradients are the columns of (R^{-1})^T (for nodes
//!     1, 2, 3) with node-0 gradient = −Σ of the others. Internal force
//!     at node i is f_i = −V · σ · ∇N_i.

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element3D {
    pub indices: [usize; 4],
    pub volume: f32,
    /// Inverse of the reference edge matrix `R = [r_ba | r_ca | r_da]`.
    /// Constant for the lifetime of the simulation (reference config does
    /// not change).
    pub ref_inv: Matrix3<f32>,
    /// Latest computed stress tensor (cached for visualization / reuse).
    pub stress: Matrix3<f32>,
    /// Latest small-strain tensor.
    pub strain: Matrix3<f32>,
}

impl Element3D {
    /// Build an element from its reference vertex positions. Returns
    /// `None` if the tetrahedron is degenerate (collinear / coplanar
    /// vertices) — the reference edge matrix is singular in that case.
    pub fn from_reference(indices: [usize; 4], reference: [Vector3<f32>; 4]) -> Option<Self> {
        let r_ba = reference[1] - reference[0];
        let r_ca = reference[2] - reference[0];
        let r_da = reference[3] - reference[0];
        let r = Matrix3::from_columns(&[r_ba, r_ca, r_da]);
        let det = r.determinant();
        if det.abs() < 1e-20 {
            return None;
        }
        let ref_inv = r.try_inverse()?;
        let volume = det.abs() / 6.0;
        Some(Self {
            indices,
            volume,
            ref_inv,
            stress: Matrix3::zeros(),
            strain: Matrix3::zeros(),
        })
    }

    /// Update strain and stress from the current node positions, using
    /// the supplied material to evaluate the constitutive law. Uses the
    /// Green-Lagrange strain tensor E = (F^T F − I) / 2 for parity with
    /// the 2D solver (`green_lagrange_strain_tensor` in
    /// `cpd/src/computer.rs`). Coincides with small-strain to leading
    /// order; captures geometric nonlinearity for larger deformations.
    pub fn update_strain_stress(
        &mut self,
        positions: [Vector3<f32>; 4],
        material: &super::IsotropicProps3D,
    ) {
        let d_ba = positions[1] - positions[0];
        let d_ca = positions[2] - positions[0];
        let d_da = positions[3] - positions[0];
        let d = Matrix3::from_columns(&[d_ba, d_ca, d_da]);
        let f = d * self.ref_inv;
        let identity = Matrix3::identity();
        self.strain = (f.transpose() * f - identity) * 0.5;
        self.stress = material.eval_stress(&self.strain);
    }

    /// Per-node internal forces produced by this element's current stress
    /// state. Returns `[f0, f1, f2, f3]` in the same order as `indices`.
    pub fn nodal_forces(&self) -> [Vector3<f32>; 4] {
        // Shape function gradients for nodes 1..=3 are the columns of
        // (R^{-1})^T; node 0 is the negative sum.
        let bt = self.ref_inv.transpose();
        let g1 = bt.column(0).into_owned();
        let g2 = bt.column(1).into_owned();
        let g3 = bt.column(2).into_owned();
        let g0 = -(g1 + g2 + g3);
        let factor = -self.volume;
        [
            self.stress * g0 * factor,
            self.stress * g1 * factor,
            self.stress * g2 * factor,
            self.stress * g3 * factor,
        ]
    }
}
