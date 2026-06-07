//! 3D orthotropic linear-elastic material.
//!
//! Nine independent elastic constants — three Young's moduli, three
//! Poisson's ratios, three shear moduli — define the orthotropic
//! compliance matrix. The constitutive law in Voigt notation is
//! σ_v = C · ε_v, where C = S⁻¹ is the 6×6 stiffness matrix:
//!
//! ```text
//!   ε_v = [ε11, ε22, ε33, 2ε23, 2ε13, 2ε12]ᵀ   (engineering shear)
//!   σ_v = [σ11, σ22, σ33,  σ23,  σ13,  σ12]ᵀ
//! ```
//!
//! Symmetry: ν_ji / E_j = ν_ij / E_i. We trust the user's three input
//! Poisson's ratios and derive the reciprocals analytically.

use nalgebra::{Matrix3, SMatrix, Vector6};
use serde::{Deserialize, Serialize};

use super::BulkProps3D;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrthotropicProps3D {
    pub elasticity_modulus_x: f32,
    pub elasticity_modulus_y: f32,
    pub elasticity_modulus_z: f32,
    pub poissons_ratio_xy: f32,
    pub poissons_ratio_xz: f32,
    pub poissons_ratio_yz: f32,
    pub shear_modulus_xy: f32,
    pub shear_modulus_xz: f32,
    pub shear_modulus_yz: f32,
    #[serde(default)]
    pub bulk: BulkProps3D,
}

impl Default for OrthotropicProps3D {
    fn default() -> Self {
        // Isotropic-equivalent defaults so a fresh Orthotropic material
        // behaves like the existing isotropic preset until the user
        // tunes the directional parameters.
        let e = 1.0e6;
        let v = 0.3;
        let g = e / (2.0 * (1.0 + v));
        Self {
            elasticity_modulus_x: e,
            elasticity_modulus_y: e,
            elasticity_modulus_z: e,
            poissons_ratio_xy: v,
            poissons_ratio_xz: v,
            poissons_ratio_yz: v,
            shear_modulus_xy: g,
            shear_modulus_xz: g,
            shear_modulus_yz: g,
            bulk: BulkProps3D::default(),
        }
    }
}

impl OrthotropicProps3D {
    pub fn stiffness(&self) -> SMatrix<f32, 6, 6> {
        let e1 = self.elasticity_modulus_x;
        let e2 = self.elasticity_modulus_y;
        let e3 = self.elasticity_modulus_z;
        let v12 = self.poissons_ratio_xy;
        let v13 = self.poissons_ratio_xz;
        let v23 = self.poissons_ratio_yz;
        // Symmetric reciprocals.
        let v21 = v12 * e2 / e1;
        let v31 = v13 * e3 / e1;
        let v32 = v23 * e3 / e2;

        // Compliance S (6x6) in Voigt order [11, 22, 33, 23, 13, 12].
        let mut s = SMatrix::<f32, 6, 6>::zeros();
        s[(0, 0)] = 1.0 / e1;
        s[(1, 1)] = 1.0 / e2;
        s[(2, 2)] = 1.0 / e3;
        s[(0, 1)] = -v21 / e2;
        s[(1, 0)] = -v12 / e1;
        s[(0, 2)] = -v31 / e3;
        s[(2, 0)] = -v13 / e1;
        s[(1, 2)] = -v32 / e3;
        s[(2, 1)] = -v23 / e2;
        s[(3, 3)] = 1.0 / self.shear_modulus_yz.max(1e-12);
        s[(4, 4)] = 1.0 / self.shear_modulus_xz.max(1e-12);
        s[(5, 5)] = 1.0 / self.shear_modulus_xy.max(1e-12);

        s.try_inverse().unwrap_or_else(SMatrix::<f32, 6, 6>::zeros)
    }

    pub fn eval_stress(&self, strain: &Matrix3<f32>) -> Matrix3<f32> {
        let c = self.stiffness();
        // Voigt strain with engineering shear (factor of 2 on off-diagonals).
        let strain_voigt = Vector6::new(
            strain.m11,
            strain.m22,
            strain.m33,
            2.0 * strain.m23,
            2.0 * strain.m13,
            2.0 * strain.m12,
        );
        let stress_voigt = c * strain_voigt;
        // Voigt back to a symmetric 3×3 tensor.
        let mut s = Matrix3::<f32>::zeros();
        s.m11 = stress_voigt[0];
        s.m22 = stress_voigt[1];
        s.m33 = stress_voigt[2];
        s.m23 = stress_voigt[3];
        s.m32 = stress_voigt[3];
        s.m13 = stress_voigt[4];
        s.m31 = stress_voigt[4];
        s.m12 = stress_voigt[5];
        s.m21 = stress_voigt[5];
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An orthotropic material whose nine constants match an isotropic
    /// material should produce the same stress as the isotropic law.
    #[test]
    fn isotropic_equivalent_matches_isotropic() {
        let e = 2.0e6;
        let v = 0.25;
        let g = e / (2.0 * (1.0 + v));
        let ortho = OrthotropicProps3D {
            elasticity_modulus_x: e,
            elasticity_modulus_y: e,
            elasticity_modulus_z: e,
            poissons_ratio_xy: v,
            poissons_ratio_xz: v,
            poissons_ratio_yz: v,
            shear_modulus_xy: g,
            shear_modulus_xz: g,
            shear_modulus_yz: g,
            bulk: BulkProps3D::default(),
        };
        let iso = super::super::isotropic::IsotropicProps3D {
            elasticity_modulus: e,
            poissons_ratio: v,
            bulk: BulkProps3D::default(),
        };
        let strain = nalgebra::matrix![
            0.01, 0.002, 0.001;
            0.002, -0.003, 0.0005;
            0.001, 0.0005, 0.004;
        ];
        let s_ortho = ortho.eval_stress(&strain);
        let s_iso = iso.eval_stress(&strain);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (s_ortho[(i, j)] - s_iso[(i, j)]).abs() < 1.0,
                    "stress mismatch at ({i},{j}): ortho={} iso={}",
                    s_ortho[(i, j)],
                    s_iso[(i, j)],
                );
            }
        }
    }
}
