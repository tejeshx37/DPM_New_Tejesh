//! Configuration knobs for a 3D simulation run.

use serde::{Deserialize, Serialize};

use super::MaterialProps3D;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Config3D {
    pub material: MaterialProps3D,
    pub time_delta_seconds: f32,
    pub duration_seconds: f32,
    /// Constant body acceleration applied to every node (gravity / pseudo-
    /// gravity). Force added per node per step is `mass * body_force`.
    #[serde(default)]
    pub body_force: [f32; 3],
}

impl Default for Config3D {
    fn default() -> Self {
        Self {
            material: MaterialProps3D::default(),
            // Conservative explicit step; users can tune for stability.
            time_delta_seconds: 1.0e-5,
            duration_seconds: 1.0e-2,
            body_force: [0.0; 3],
        }
    }
}

impl Config3D {
    pub fn total_steps(&self) -> u64 {
        (self.duration_seconds / self.time_delta_seconds).max(1.0).round() as u64
    }
}
