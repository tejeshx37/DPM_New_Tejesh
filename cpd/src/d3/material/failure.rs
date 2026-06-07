//! 3D failure criteria — mirror of `cpd::material::FailureCriteria`.
//!
//! Three optional thresholds:
//! - strain energy (scalar)
//! - tensile principal stress (positive eigenvalue)
//! - compressive principal stress (negative eigenvalue, compared as |σ|)
//!
//! An element is marked broken as soon as any active threshold is hit.
//! Once broken, the solver zeroes its stress contribution.

use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FailureCriteria3D {
    pub strain_energy: Option<f32>,
    pub tensional_stress: Option<f32>,
    pub compressional_stress: Option<f32>,
}

impl FailureCriteria3D {
    /// True if any active threshold is exceeded by the supplied stress
    /// state (eigenvalues of the symmetric stress tensor are the
    /// principal stresses).
    pub fn satisfies(&self, strain_energy: f32, stress: &Matrix3<f32>) -> bool {
        if self.strain_energy.is_some_and(|v| strain_energy >= v) {
            return true;
        }
        if self.tensional_stress.is_none() && self.compressional_stress.is_none() {
            return false;
        }

        let principals = stress.symmetric_eigenvalues();

        let tensile_exceeded = principals.iter().filter(|p| p.is_sign_positive()).any(|p| {
            self.tensional_stress.is_some_and(|t| *p >= t)
        });
        if tensile_exceeded {
            return true;
        }

        principals
            .iter()
            .filter(|p| p.is_sign_negative())
            .map(|p| p.abs())
            .any(|p| {
                self.compressional_stress.is_some_and(|t| p >= t)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_thresholds_never_breaks() {
        let crit = FailureCriteria3D::default();
        let s = Matrix3::from_element(1e9);
        assert!(!crit.satisfies(1e9, &s));
    }

    #[test]
    fn strain_energy_threshold_works() {
        let crit = FailureCriteria3D {
            strain_energy: Some(100.0),
            ..Default::default()
        };
        let s = Matrix3::zeros();
        assert!(!crit.satisfies(99.0, &s));
        assert!(crit.satisfies(100.0, &s));
        assert!(crit.satisfies(200.0, &s));
    }

    #[test]
    fn tensile_principal_stress_threshold_works() {
        let crit = FailureCriteria3D {
            tensional_stress: Some(1.0e5),
            ..Default::default()
        };
        let mut s = Matrix3::<f32>::zeros();
        s.m11 = 2.0e5; // principal stress = 2e5
        assert!(crit.satisfies(0.0, &s));
        s.m11 = 0.5e5;
        assert!(!crit.satisfies(0.0, &s));
    }

    #[test]
    fn compressive_principal_stress_threshold_works() {
        let crit = FailureCriteria3D {
            compressional_stress: Some(1.0e5),
            ..Default::default()
        };
        let mut s = Matrix3::<f32>::zeros();
        s.m22 = -2.0e5;
        assert!(crit.satisfies(0.0, &s));
        s.m22 = -0.5e5;
        assert!(!crit.satisfies(0.0, &s));
    }
}
