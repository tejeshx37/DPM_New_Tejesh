use super::{BulkProps, ElasticityCondition};
use derive_getters::Getters;
use nalgebra::{Matrix2, Matrix3};
use typed_builder::TypedBuilder;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, TypedBuilder, Getters)]
pub struct Props {
    bulk_props: BulkProps,
    elasticity_modulus: f32,
    elasticity_condition: ElasticityCondition,
    poissons_ratio: f32,
}

impl Props {
    pub(crate) fn eval_stress(&self, strain: &Matrix2<f32>) -> Matrix2<f32> {
        match self.elasticity_condition {
            ElasticityCondition::PlaneStress => {
                plane_stress(self.elasticity_modulus, self.poissons_ratio, strain)
            }
            ElasticityCondition::PlaneStrain => {
                plane_strain(self.elasticity_modulus, self.poissons_ratio, strain)
            }
        }
    }
}

fn plane_stress_matrix(v: f32) -> Matrix3<f32> {
    nalgebra::matrix![
        1.0, v, 0.0;
        v, 1.0, 0.0;
        0.0, 0.0, (1.0 - v) / 2.0;
    ]
}

fn plane_stress(e: f32, v: f32, strain: &Matrix2<f32>) -> Matrix2<f32> {
    super::stress_vector_to_matrix(
        (e / (1.0 - v.powi(2))) * plane_stress_matrix(v) * super::strain_matrix_to_vector(strain),
    )
}

fn plane_strain_matrix(v: f32) -> Matrix3<f32> {
    nalgebra::matrix![
        1.0 - v, v, 0.0;
        v, 1.0 - v, 0.0;
        0.0, 0.0, (1.0 - v * 2.0) / 2.0;
    ]
}

fn plane_strain(e: f32, v: f32, strain: &Matrix2<f32>) -> Matrix2<f32> {
    super::stress_vector_to_matrix(
        (e / (1.0 + v) / (1.0 - v * 2.0))
            * plane_strain_matrix(v)
            * super::strain_matrix_to_vector(strain),
    )
}
