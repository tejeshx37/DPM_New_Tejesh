//! Preview viewport for the 3D drawing page.
//!
//! Architecture: this module owns the orbit camera, manual perspective
//! projection, and per-shape wireframe tessellation. Rendering currently
//! goes through egui's 2D `Painter` (line segments). The renderer is
//! deliberately isolated behind `paint_scene` so a future `egui-wgpu`
//! backend can replace this without touching the page or dialog code.

pub mod camera;
pub mod mesh_builder;

use camera::OrbitCamera;
use egui::{Color32, Pos2, Sense, Stroke, Ui, Vec2};
use mesh_builder::Edge;
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
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.0 {
            state.camera.zoom(scroll);
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
    for (shape, op) in &geometry.shapes {
        let edges = mesh_builder::edges_for(shape);
        let color = match op {
            CsgOp3D::Union => Color32::from_rgb(120, 200, 255),
            CsgOp3D::Difference => Color32::from_rgb(255, 140, 140),
        };
        let stroke = Stroke::new(1.2, color);
        for Edge { a, b } in edges {
            if let (Some(pa), Some(pb)) = (project(&view_proj, rect, a), project(&view_proj, rect, b))
            {
                painter.line_segment([pa, pb], stroke);
            }
        }
    }
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
fn project(view_proj: &Matrix4<f64>, rect: egui::Rect, p: Vector3<f64>) -> Option<Pos2> {
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
