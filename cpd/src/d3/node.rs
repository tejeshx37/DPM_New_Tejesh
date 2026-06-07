//! 3D particle node.
//!
//! Each node stores its current kinematic state plus its initial position
//! (needed both for displacement BCs and for computing element strain).
//! The `bc` field is `BoundaryCondition3D::Free` for interior particles.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use super::BoundaryCondition3D;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node3D {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub force: Vector3<f32>,
    pub initial_position: Vector3<f32>,
    pub mass: f32,
    pub bc: BoundaryCondition3D,
}

impl Node3D {
    pub fn new(position: Vector3<f32>, mass: f32) -> Self {
        Self {
            position,
            velocity: Vector3::zeros(),
            force: Vector3::zeros(),
            initial_position: position,
            mass,
            bc: BoundaryCondition3D::Free,
        }
    }

    pub fn displacement(&self) -> Vector3<f32> {
        self.position - self.initial_position
    }
}
