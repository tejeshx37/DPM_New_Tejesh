pub mod isotropic;
pub mod orthotropic;

use derive_getters::Getters;
use nalgebra::{Matrix2, Vector2, Vector3};
use typed_builder::TypedBuilder;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Default, Clone, Copy, PartialEq, TypedBuilder, Getters)]
pub struct FailureCriteria {
    strain_energy: Option<f32>,
    tensional_stress: Option<f32>,
    compressional_stress: Option<f32>,
}

impl FailureCriteria {
    pub(crate) fn satisfies(&self, strain_energy: f32, stress: &Matrix2<f32>) -> bool {
        if self
            .strain_energy()
            .is_some_and(|value| strain_energy >= value)
        {
            return true;
        }

        if self.tensional_stress().is_none() && self.compressional_stress().is_none() {
            return false;
        }

        let principal_stress: Vector2<f32> = stress
            .eigenvalues()
            .expect("Stress matrix has real principal stress components");

        let compression_exceeded = principal_stress
            .iter()
            .filter(|value| value.is_sign_negative())
            .map(|value| value.abs())
            .any(|value| {
                self.compressional_stress()
                    .is_some_and(|stress| value >= stress)
            });
        if compression_exceeded {
            return true;
        }

        principal_stress
            .iter()
            .filter(|value| value.is_sign_positive())
            .any(|value| {
                self.tensional_stress()
                    .is_some_and(|stress| *value >= stress)
            })
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum ElasticityCondition {
    #[default]
    PlaneStress,
    PlaneStrain,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Props {
    Isotropic(isotropic::Props),
    Orthotropic(orthotropic::Props),
}

impl Props {
    pub fn bulk_props(&self) -> &BulkProps {
        match self {
            Props::Isotropic(p) => p.bulk_props(),
            Props::Orthotropic(p) => p.bulk_props(),
        }
    }

    pub(crate) fn eval_stress(&self, strain: &Matrix2<f32>) -> Matrix2<f32> {
        match self {
            Props::Isotropic(p) => p.eval_stress(strain),
            Props::Orthotropic(p) => p.eval_stress(strain),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, TypedBuilder, Getters)]
pub struct BulkProps {
    density: f32,
    damping: f32,
    failure_criteria: FailureCriteria,
}

pub(crate) fn strain_matrix_to_vector(strain: &Matrix2<f32>) -> Vector3<f32> {
    Vector3::new(strain.m11, strain.m22, strain.m12)
}

pub(crate) fn stress_vector_to_matrix(stress: Vector3<f32>) -> Matrix2<f32> {
    Matrix2::new(stress.x, stress.z, stress.z, stress.y)
}
