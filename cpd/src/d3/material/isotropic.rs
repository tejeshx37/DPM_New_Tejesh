//! 3D isotropic linear-elastic material model.
//!
//! Constitutive law: σ = λ tr(ε) I + 2μ ε with Lamé parameters
//! λ = E ν / ((1+ν)(1-2ν)) and μ = E / (2(1+ν)).

use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

use super::BulkProps3D;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IsotropicProps3D {
    pub elasticity_modulus: f32,
    pub poissons_ratio: f32,
    /// Density / damping / failure criteria live here so they're
    /// uniform across the 3D material kinds.
    #[serde(default)]
    pub bulk: BulkProps3D,
}

impl Default for IsotropicProps3D {
    fn default() -> Self {
        Self {
            elasticity_modulus: 1.0e6,
            poissons_ratio: 0.3,
            bulk: BulkProps3D::default(),
        }
    }
}

impl IsotropicProps3D {
    pub fn lame_lambda(&self) -> f32 {
        let v = self.poissons_ratio;
        self.elasticity_modulus * v / ((1.0 + v) * (1.0 - 2.0 * v))
    }

    pub fn lame_mu(&self) -> f32 {
        self.elasticity_modulus / (2.0 * (1.0 + self.poissons_ratio))
    }

    pub fn eval_stress(&self, strain: &Matrix3<f32>) -> Matrix3<f32> {
        let lambda = self.lame_lambda();
        let mu = self.lame_mu();
        let tr = strain.m11 + strain.m22 + strain.m33;
        Matrix3::identity() * (lambda * tr) + strain * (2.0 * mu)
    }
}
