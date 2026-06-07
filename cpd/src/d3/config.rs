//! Configuration knobs for a 3D simulation run.

use serde::{Deserialize, Serialize};

use super::IsotropicProps3D;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Config3D {
    pub material: IsotropicProps3D,
    pub time_delta_seconds: f32,
    pub duration_seconds: f32,
}

impl Default for Config3D {
    fn default() -> Self {
        Self {
            material: IsotropicProps3D::default(),
            // Conservative explicit step; users can tune for stability.
            time_delta_seconds: 1.0e-5,
            duration_seconds: 1.0e-2,
        }
    }
}

impl Config3D {
    pub fn total_steps(&self) -> u64 {
        (self.duration_seconds / self.time_delta_seconds).max(1.0).round() as u64
    }
}
