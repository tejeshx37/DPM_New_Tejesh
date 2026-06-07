//! 3D boundary conditions. Variants follow the 2D model:
//! - `Free`: no constraint.
//! - `Pinned`: constrain selected axes to their initial position.
//! - `ConstantForce`: add a time-constant force vector to the node every
//!   step.
//! - `ConstantDisplacement`: ramp specified axes to a target displacement
//!   over `ramp_seconds`, then hold.
//!
//! Per-axis selection is encoded via the `Axis` bitset so a single node
//! can be pinned in (e.g.) x and z while free in y. Time-varying
//! `Function` BCs from the 2D solver are deliberately omitted for now —
//! constant + linear-ramp covers the common load cases for validation.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Bitmask of cartesian axes for partial constraints.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Axis {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl Axis {
    pub const NONE: Self = Self {
        x: false,
        y: false,
        z: false,
    };
    pub const ALL: Self = Self {
        x: true,
        y: true,
        z: true,
    };
    pub const fn x_only() -> Self {
        Self {
            x: true,
            y: false,
            z: false,
        }
    }
    pub const fn y_only() -> Self {
        Self {
            x: false,
            y: true,
            z: false,
        }
    }
    pub const fn z_only() -> Self {
        Self {
            x: false,
            y: false,
            z: true,
        }
    }

    pub fn any(self) -> bool {
        self.x || self.y || self.z
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryCondition3D {
    Free,
    Pinned {
        axes: Axis,
    },
    ConstantForce {
        force: [f32; 3],
    },
    ConstantDisplacement {
        axes: Axis,
        displacement: [f32; 3],
        ramp_seconds: f32,
    },
}

impl Default for BoundaryCondition3D {
    fn default() -> Self {
        Self::Free
    }
}

impl BoundaryCondition3D {
    /// Evaluate a ramped displacement at a given simulation time. Returns
    /// the constrained target offset from the initial position; callers
    /// apply only the components selected by `axes`.
    pub(crate) fn ramped_displacement(
        target: Vector3<f32>,
        ramp_seconds: f32,
        t: f32,
    ) -> Vector3<f32> {
        if ramp_seconds <= 0.0 {
            return target;
        }
        let frac = (t / ramp_seconds).clamp(0.0, 1.0);
        target * frac
    }
}
