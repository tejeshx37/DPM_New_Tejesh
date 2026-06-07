//! 3D material models.
//!
//! Mirrors the 2D material tree in `cpd/src/material/`:
//! - `isotropic` — linear elastic isotropic (E, ν) constitutive law
//! - `orthotropic` — 9-parameter orthotropic linear elasticity via the
//!   standard 6×6 stiffness matrix in Voigt notation
//! - `failure` — energy / principal-stress thresholds for element
//!   breakage, shared across material kinds via `BulkProps3D`
//!
//! `MaterialProps3D` is the enum that wraps both material kinds; it
//! exposes a unified `eval_stress`, `density`, `damping`, and
//! `failure_criteria` so the solver doesn't need to match on the kind.

pub mod failure;
pub mod isotropic;
pub mod orthotropic;

use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

pub use failure::FailureCriteria3D;
pub use isotropic::IsotropicProps3D;
pub use orthotropic::OrthotropicProps3D;

/// Density, damping, and failure criteria are shared by all material
/// kinds. Mirrors `cpd::material::BulkProps`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BulkProps3D {
    pub density: f32,
    pub damping: f32,
    pub failure_criteria: FailureCriteria3D,
}

impl Default for BulkProps3D {
    fn default() -> Self {
        Self {
            density: 1000.0,
            damping: 1.0,
            failure_criteria: FailureCriteria3D::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MaterialProps3D {
    Isotropic(IsotropicProps3D),
    Orthotropic(OrthotropicProps3D),
}

impl Default for MaterialProps3D {
    fn default() -> Self {
        Self::Isotropic(IsotropicProps3D::default())
    }
}

impl MaterialProps3D {
    pub fn bulk(&self) -> &BulkProps3D {
        match self {
            Self::Isotropic(p) => &p.bulk,
            Self::Orthotropic(p) => &p.bulk,
        }
    }

    pub fn density(&self) -> f32 {
        self.bulk().density
    }

    pub fn damping(&self) -> f32 {
        self.bulk().damping
    }

    pub fn failure_criteria(&self) -> FailureCriteria3D {
        self.bulk().failure_criteria
    }

    pub fn eval_stress(&self, strain: &Matrix3<f32>) -> Matrix3<f32> {
        match self {
            Self::Isotropic(p) => p.eval_stress(strain),
            Self::Orthotropic(p) => p.eval_stress(strain),
        }
    }
}
