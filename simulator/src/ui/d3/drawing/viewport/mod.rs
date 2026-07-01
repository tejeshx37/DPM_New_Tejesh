//! Preview viewport for the 3D drawing page.
//!
//! Architecture: this module owns the orbit camera, manual perspective
//! projection, and per-shape wireframe tessellation. Rendering currently
//! goes through egui's 2D `Painter` (line segments). The renderer is
//! deliberately isolated behind `paint_scene` so a future `egui-wgpu`
//! backend can replace this without touching the page or dialog code.

pub mod camera;
#[allow(dead_code)]
pub mod mesh_builder;
pub mod scene_mesh;
pub mod wgpu_scene;

use camera::OrbitCamera;
use egui::{Color32, Pos2, Sense, Stroke, Ui, Vec2};
use nalgebra::{Matrix4, Point3, Vector3};
use serde::{Deserialize, Serialize};

use super::shape::{CsgOp3D, Geometry3D};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportState {
    pub camera: OrbitCamera,
    /// When true, auto-frame the next time we paint (set on first paint and
    /// whenever the user adds a shape outside the current view).
    #[serde(default = "default_true")]
    pub auto_frame: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            camera: OrbitCamera::default(),
            auto_frame: true,
        }
    }
}

pub fn show(state: &mut ViewportState, geometry: &Geometry3D, ui: &mut Ui) {
    let available = ui.available_size();
    let size = Vec2::new(available.x.max(100.0), available.y.max(100.0));
    let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
    let rect = response.rect;

    if state.auto_frame && !geometry.shapes.is_empty() {
        let (lo, hi) = geometry.aabb();
        state.camera.frame_aabb(lo, hi);
        state.auto_frame = false;
    }

    // Mouse interaction.
    if response.dragged_by(egui::PointerButton::Primary) {
        let delta = response.drag_delta();
        state.camera.rotate(delta.x, delta.y);
    }
    if response.dragged_by(egui::PointerButton::Secondary) {
        let delta = response.drag_delta();
        state.camera.pan(delta.x, delta.y);
    }
    if response.hovered() {
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        if scroll.abs() > 0.0 {
            state.camera.zoom(scroll);
        }
        if (pinch - 1.0).abs() > 1e-4 {
            state.camera.zoom_by(pinch);
        }
    }

    // Background.
    painter.rect_filled(rect, 0.0, Color32::from_gray(20));

    // World axes (subtle).
    paint_world_axes(&state.camera, rect, &painter);

    // Shapes.
    paint_scene(&state.camera, rect, &painter, geometry);

    // HUD.
    let label = format!(
        "{} shapes  |  LMB rotate  RMB pan  scroll zoom",
        geometry.shapes.len()
    );
    painter.text(
        rect.left_top() + Vec2::new(8.0, 8.0),
        egui::Align2::LEFT_TOP,
        label,
        egui::FontId::monospace(11.0),
        Color32::from_gray(180),
    );
}

fn paint_scene(camera: &OrbitCamera, rect: egui::Rect, painter: &egui::Painter, geometry: &Geometry3D) {
    let view_proj = camera.view_projection(rect.width() / rect.height().max(1.0));
    let mut tris = Vec::new();
    for (shape, op) in &geometry.shapes {
        let color = match op {
            CsgOp3D::Union => [0.47, 0.78, 1.0, 0.95],
            CsgOp3D::Difference => [1.0, 0.55, 0.55, 0.35],
        };
        tris.extend(scene_mesh::triangles_for(shape, color));
    }
    if tris.is_empty() {
        return;
    }
    wgpu_scene::sort_back_to_front(&mut tris, &view_proj);
    let callback = wgpu_scene::SceneCallback::from_world_mvp(tris, &view_proj);
    painter.add(eframe::egui_wgpu::Callback::new_paint_callback(
        rect, callback,
    ));
}

fn paint_world_axes(camera: &OrbitCamera, rect: egui::Rect, painter: &egui::Painter) {
    let view_proj = camera.view_projection(rect.width() / rect.height().max(1.0));
    let origin = Vector3::new(0.0, 0.0, 0.0);
    let len = camera.distance() * 0.15;
    let axes = [
        (Vector3::new(len, 0.0, 0.0), Color32::from_rgb(220, 80, 80)),
        (Vector3::new(0.0, len, 0.0), Color32::from_rgb(80, 220, 80)),
        (Vector3::new(0.0, 0.0, len), Color32::from_rgb(80, 140, 240)),
    ];
    for (dir, color) in axes {
        if let (Some(o), Some(p)) = (project(&view_proj, rect, origin), project(&view_proj, rect, dir))
        {
            painter.line_segment([o, p], Stroke::new(1.0, color));
        }
    }
}

/// Project a world-space point to screen coordinates. Returns `None` if the
/// point is behind the camera or outside reasonable clip space.
pub fn project(view_proj: &Matrix4<f64>, rect: egui::Rect, p: Vector3<f64>) -> Option<Pos2> {
    let p4 = view_proj * nalgebra::Vector4::new(p.x, p.y, p.z, 1.0);
    if p4.w <= 1e-6 {
        return None;
    }
    let ndc_x = p4.x / p4.w;
    let ndc_y = p4.y / p4.w;
    let _ = Point3::new(ndc_x, ndc_y, p4.z / p4.w);
    let x = rect.left() + ((ndc_x + 1.0) * 0.5) as f32 * rect.width();
    let y = rect.top() + ((1.0 - (ndc_y + 1.0) * 0.5) as f32) * rect.height();
    Some(Pos2::new(x, y))
}

/// Build a world-space ray (origin, normalized direction) from a screen
/// point, the exact inverse of [`project`]. Used by the Meshing page's
/// area-selection tool to figure out which shape/surface point the user
/// clicked on.
///
/// Unprojects two points along the same screen-space ray (near and far NDC
/// depth) through the inverse view-projection matrix and derives the
/// direction from their difference — this naturally recovers the camera eye
/// as the ray origin without needing it as a separate input.
pub fn ray_from_screen(
    view_proj: &Matrix4<f64>,
    rect: egui::Rect,
    screen_pos: Pos2,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let inv = view_proj.try_inverse()?;

    let ndc_x = ((screen_pos.x - rect.left()) / rect.width().max(1.0)) as f64 * 2.0 - 1.0;
    let ndc_y = 1.0 - ((screen_pos.y - rect.top()) / rect.height().max(1.0)) as f64 * 2.0;

    let unproject = |ndc_z: f64| -> Option<Vector3<f64>> {
        let clip = nalgebra::Vector4::new(ndc_x, ndc_y, ndc_z, 1.0);
        let world = inv * clip;
        if world.w.abs() < 1e-12 {
            return None;
        }
        Some(Vector3::new(world.x, world.y, world.z) / world.w)
    };

    let near = unproject(-1.0)?;
    let far = unproject(1.0)?;
    let dir = (far - near).normalize();
    if !dir.iter().all(|c| c.is_finite()) {
        return None;
    }
    Some((near, dir))
}

#[cfg(test)]
mod ray_tests {
    use super::*;
    use camera::OrbitCamera;

    #[test]
    fn project_and_unproject_round_trip() {
        let camera = OrbitCamera::default();
        let rect = egui::Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(800.0, 600.0));
        let view_proj = camera.view_projection(rect.width() / rect.height());

        let world_point = Vector3::new(0.3, -0.2, 0.1);
        let screen = project(&view_proj, rect, world_point).expect("point should project");

        let (origin, dir) = ray_from_screen(&view_proj, rect, screen).expect("ray should unproject");
        // The world point should lie on the ray (some positive t along dir).
        let to_point = world_point - origin;
        let t = to_point.dot(&dir);
        let closest = origin + dir * t;
        assert!((closest - world_point).norm() < 1e-6);
    }
}
