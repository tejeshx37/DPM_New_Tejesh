use super::BulkProps;
use derive_getters::Getters;
use nalgebra::{Matrix2, Matrix3};
use typed_builder::TypedBuilder;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, TypedBuilder, Getters)]
pub struct Props {
    bulk_props: BulkProps,
    elasticity_modulus_x: f32,
    elasticity_modulus_y: f32,
    poissons_ratio_xy: f32,
    poissons_ratio_yx: f32,
    shear_modulus_xy: f32,
}

impl Props {
    pub(crate) fn eval_stress(&self, strain: &Matrix2<f32>) -> Matrix2<f32> {
        plane_stress(
            self.elasticity_modulus_x,
            self.elasticity_modulus_y,
            self.poissons_ratio_xy,
            self.poissons_ratio_yx,
            self.shear_modulus_xy,
            strain,
        )
    }
}

fn plane_stress(
    e_x: f32,
    e_y: f32,
    v_xy: f32,
    v_yx: f32,
    g_xy: f32,
    strain: &Matrix2<f32>,
) -> Matrix2<f32> {
    let stress_matrix: Matrix3<f32> = nalgebra::matrix![
        e_x, v_yx * e_x, 0.0;
        v_xy * e_y, e_y, 0.0;
        0.0, 0.0, g_xy * (1.0 - v_xy * v_yx);
    ];
    super::stress_vector_to_matrix(
        (1.0 / (1.0 - v_xy * v_yx)) * stress_matrix * super::strain_matrix_to_vector(strain),
    )
}
