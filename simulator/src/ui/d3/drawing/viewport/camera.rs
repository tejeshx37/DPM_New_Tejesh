//! Orbit camera for the 3D preview viewport.
//!
//! Spherical-coordinate orbit around a focus point: yaw, pitch, and
//! distance. Builds a right-handed look-at view matrix combined with a
//! standard perspective projection. All math is in `f64` for parity with
//! shape parameters; conversion to `f32` happens at the painter boundary.

use nalgebra::{Matrix4, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitCamera {
    pub focus: [f64; 3],
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub fov_y_deg: f64,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: [0.0, 0.0, 0.0],
            yaw: std::f64::consts::FRAC_PI_4,
            pitch: -std::f64::consts::FRAC_PI_6,
            distance: 4.0,
            fov_y_deg: 45.0,
        }
    }
}

impl OrbitCamera {
    pub fn distance(&self) -> f64 {
        self.distance
    }

    /// Rotate by mouse-drag deltas (screen pixels). Pitch is clamped to
    /// avoid gimbal flip at the poles.
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        let sens = 0.005;
        self.yaw += dx as f64 * sens;
        self.pitch += dy as f64 * sens;
        let limit = std::f64::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    /// Pan the focus point in the camera's local XY plane.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let (right, up, _fwd) = self.basis();
        let scale = self.distance * 0.0015;
        let delta = right * (-dx as f64 * scale) + up * (dy as f64 * scale);
        self.focus[0] += delta.x;
        self.focus[1] += delta.y;
        self.focus[2] += delta.z;
    }

    pub fn zoom(&mut self, scroll: f32) {
        let factor = (1.0 - scroll as f64 * 0.002).clamp(0.5, 1.5);
        self.distance = (self.distance * factor).clamp(1e-3, 1e6);
    }

    /// Multiplicative zoom for trackpad pinch / Ctrl+scroll. `factor > 1.0`
    /// = zoom in (camera distance decreases). Clamped like the scroll
    /// path so a single gesture can't blow past the distance limits.
    pub fn zoom_by(&mut self, factor: f32) {
        let inv = (1.0 / factor as f64).clamp(0.1, 10.0);
        self.distance = (self.distance * inv).clamp(1e-3, 1e6);
    }

    /// Auto-frame the camera so the AABB fits in view.
    pub fn frame_aabb(&mut self, lo: Vector3<f64>, hi: Vector3<f64>) {
        let center = (lo + hi) * 0.5;
        self.focus = [center.x, center.y, center.z];
        let extent = (hi - lo).norm().max(1e-6);
        let fov = self.fov_y_deg.to_radians();
        self.distance = (extent * 0.5) / (fov * 0.5).tan() + extent * 0.5;
    }

    /// World-space camera basis: (right, up, forward). Public so the
    /// Meshing page's "Show particles" toggle can orient billboard
    /// quads to face the camera.
    pub fn basis_world(&self) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        self.basis()
    }

    /// Advance yaw by an angle in radians. Used by the auto-rotate
    /// toggle on the Meshing page.
    pub fn rotate_yaw(&mut self, delta: f64) {
        self.yaw += delta;
    }

    /// World-space camera basis: (right, up, forward).
    fn basis(&self) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        // Forward from camera to focus, derived from spherical coords.
        let forward = Vector3::new(-cp * sy, -sp, -cp * cy).normalize();
        let world_up = Vector3::new(0.0, 1.0, 0.0);
        let right = forward.cross(&world_up).normalize();
        let up = right.cross(&forward).normalize();
        (right, up, forward)
    }

    /// World-space camera eye (viewpoint) position. Public so screen-to-world
    /// raycasting (area-selection tool on the Meshing page) can construct a
    /// ray origin without duplicating the orbit math here.
    pub fn eye(&self) -> Vector3<f64> {
        let (_r, _u, fwd) = self.basis();
        Vector3::from(self.focus) - fwd * self.distance
    }

    pub fn view_projection(&self, aspect: f32) -> Matrix4<f64> {
        let eye = self.eye();
        let target = Vector3::from(self.focus);
        let up = Vector3::new(0.0, 1.0, 0.0);
        let view = look_at_rh(eye, target, up);
        let proj = perspective_rh(self.fov_y_deg.to_radians(), aspect as f64, 0.01, 1e4);
        proj * view
    }
}

fn look_at_rh(eye: Vector3<f64>, target: Vector3<f64>, up: Vector3<f64>) -> Matrix4<f64> {
    let f = (target - eye).normalize();
    let s = f.cross(&up).normalize();
    let u = s.cross(&f);
    let mut m = Matrix4::<f64>::identity();
    m[(0, 0)] = s.x;
    m[(0, 1)] = s.y;
    m[(0, 2)] = s.z;
    m[(1, 0)] = u.x;
    m[(1, 1)] = u.y;
    m[(1, 2)] = u.z;
    m[(2, 0)] = -f.x;
    m[(2, 1)] = -f.y;
    m[(2, 2)] = -f.z;
    m[(0, 3)] = -s.dot(&eye);
    m[(1, 3)] = -u.dot(&eye);
    m[(2, 3)] = f.dot(&eye);
    m
}

fn perspective_rh(fov_y: f64, aspect: f64, near: f64, far: f64) -> Matrix4<f64> {
    let f = 1.0 / (fov_y * 0.5).tan();
    let mut m = Matrix4::<f64>::zeros();
    m[(0, 0)] = f / aspect;
    m[(1, 1)] = f;
    m[(2, 2)] = (far + near) / (near - far);
    m[(2, 3)] = (2.0 * far * near) / (near - far);
    m[(3, 2)] = -1.0;
    m
}
