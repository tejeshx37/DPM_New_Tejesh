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

/// Piecewise-linear time series used by time-varying BCs. Keyframes are
/// `(time_seconds, value)` and interpreted with linear interpolation
/// between them. Outside the first/last keyframe the value is clamped to
/// the nearest endpoint (so a constant tail past the last keyframe
/// behaves predictably during long runs). Empty = always 0.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSeries {
    pub points: Vec<(f32, f32)>,
}

impl TimeSeries {
    pub fn evaluate(&self, t: f32) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        if t <= self.points[0].0 {
            return self.points[0].1;
        }
        let last = self.points.last().unwrap();
        if t >= last.0 {
            return last.1;
        }
        for w in self.points.windows(2) {
            let (t0, v0) = w[0];
            let (t1, v1) = w[1];
            if t1 < t0 {
                continue;
            }
            if t >= t0 && t <= t1 {
                let dt = (t1 - t0).max(1e-12);
                let frac = (t - t0) / dt;
                return v0 + (v1 - v0) * frac;
            }
        }
        last.1
    }

    /// Push a keyframe and keep points sorted by time.
    pub fn push_keyframe(&mut self, t: f32, v: f32) {
        self.points.push((t, v));
        self.points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Three time series, one per axis. Used inside [`BoundaryCondition3D`]
/// variants that need per-axis time-varying values.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisTimeSeries {
    pub x: TimeSeries,
    pub y: TimeSeries,
    pub z: TimeSeries,
}

impl AxisTimeSeries {
    pub fn evaluate(&self, t: f32) -> Vector3<f32> {
        Vector3::new(self.x.evaluate(t), self.y.evaluate(t), self.z.evaluate(t))
    }
}

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
    /// Force whose three components vary with time independently via
    /// piecewise-linear keyframes. Each axis can be left empty (= 0).
    TimeForce {
        profile: AxisTimeSeries,
    },
    /// Displacement target whose three components vary with time;
    /// applied only on axes selected by `axes` (others integrate freely).
    TimeDisplacement {
        axes: Axis,
        profile: AxisTimeSeries,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_series_empty_is_zero() {
        let s = TimeSeries::default();
        assert_eq!(s.evaluate(0.0), 0.0);
        assert_eq!(s.evaluate(10.0), 0.0);
    }

    #[test]
    fn time_series_linearly_interpolates() {
        let s = TimeSeries {
            points: vec![(0.0, 0.0), (1.0, 10.0), (3.0, 30.0)],
        };
        assert!((s.evaluate(0.0) - 0.0).abs() < 1e-6);
        assert!((s.evaluate(0.5) - 5.0).abs() < 1e-6);
        assert!((s.evaluate(1.0) - 10.0).abs() < 1e-6);
        assert!((s.evaluate(2.0) - 20.0).abs() < 1e-6);
        assert!((s.evaluate(3.0) - 30.0).abs() < 1e-6);
    }

    #[test]
    fn time_series_clamps_outside_range() {
        let s = TimeSeries {
            points: vec![(1.0, 5.0), (2.0, 8.0)],
        };
        // Before the first keyframe -> first value.
        assert!((s.evaluate(-1.0) - 5.0).abs() < 1e-6);
        assert!((s.evaluate(0.5) - 5.0).abs() < 1e-6);
        // After the last keyframe -> last value.
        assert!((s.evaluate(5.0) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn push_keyframe_keeps_sorted() {
        let mut s = TimeSeries::default();
        s.push_keyframe(2.0, 20.0);
        s.push_keyframe(0.0, 0.0);
        s.push_keyframe(1.0, 10.0);
        assert_eq!(
            s.points,
            vec![(0.0, 0.0), (1.0, 10.0), (2.0, 20.0)]
        );
    }
}
