//! 3D meshing page. Reads the `Geometry3D` from the drawing state, generates
//! a tetrahedral mesh for each supported primitive, and renders the result
//! in the same orbit-camera viewport used by the drawing page.

use egui::{Color32, ScrollArea, Sense, SidePanel, Slider, Stroke, Ui, Vec2};
use mesh::d3::{cuboid, Mesh3D};
use serde::{Deserialize, Serialize};

use super::drawing::{
    shape::{Geometry3D, Shape3D},
    viewport::{camera::OrbitCamera, project, ViewportState},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// User-controlled subdivision count (applied to all axes for cubes;
    /// scaled by extents ratio for cuboids).
    #[serde(default = "default_subdivisions")]
    pub subdivisions: u32,
    /// Generated meshes, one per source shape. Cleared when subdivisions or
    /// geometry change. Indices match `geometry.shapes`.
    #[serde(default)]
    pub meshes: Vec<Option<Mesh3D>>,
    #[serde(default)]
    pub viewport: ViewportState,
    /// Last error message from mesh generation, if any.
    #[serde(default)]
    pub error: Option<String>,
}

fn default_subdivisions() -> u32 {
    4
}

impl Default for State {
    fn default() -> Self {
        Self {
            subdivisions: 4,
            meshes: Vec::new(),
            viewport: ViewportState::default(),
            error: None,
        }
    }
}

pub fn show(state: &mut State, geometry: &Geometry3D, ui: &mut Ui) {
    SidePanel::right("d3_meshing_side_panel")
        .resizable(true)
        .default_width(240.0)
        .show_inside(ui, |ui| {
            ui.heading("3D Meshing");
            ui.label("Subdivisions per axis:");
            ui.add(Slider::new(&mut state.subdivisions, 1..=30));

            ui.add_space(4.0);
            if ui.button("Generate Mesh").clicked() {
                regenerate(state, geometry);
            }
            if ui.button("Clear").clicked() {
                state.meshes.clear();
                state.error = None;
            }

            if let Some(err) = &state.error {
                ui.add_space(4.0);
                ui.colored_label(Color32::from_rgb(240, 120, 120), err);
            }

            ui.separator();
            add_stats(state, geometry, ui);
        });

    let available = ui.available_size();
    let size = Vec2::new(available.x.max(100.0), available.y.max(100.0));
    let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
    let rect = response.rect;

    if state.viewport.auto_frame && !geometry.shapes.is_empty() {
        let (lo, hi) = geometry.aabb();
        state.viewport.camera.frame_aabb(lo, hi);
        state.viewport.auto_frame = false;
    }

    handle_camera_input(&mut state.viewport.camera, &response, ui);

    painter.rect_filled(rect, 0.0, Color32::from_gray(20));

    let view_proj = state
        .viewport
        .camera
        .view_projection(rect.width() / rect.height().max(1.0));

    let stroke = Stroke::new(0.6, Color32::from_rgb(180, 220, 255));
    for mesh in state.meshes.iter().flatten() {
        // Draw each tetrahedron's 6 edges. Visually equivalent to drawing
        // the surface wireframe for the boundary faces, but this also shows
        // interior edges which is useful for sanity-checking the mesh.
        for tet in &mesh.tetrahedra {
            for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                if let (Some(pa), Some(pb)) = (
                    project(&view_proj, rect, mesh.vertices[tet[a]]),
                    project(&view_proj, rect, mesh.vertices[tet[b]]),
                ) {
                    painter.line_segment([pa, pb], stroke);
                }
            }
        }
    }

    let hud = format!(
        "{} tets across {} mesh(es)  |  LMB rotate  RMB pan  scroll zoom",
        state.meshes.iter().flatten().map(|m| m.tet_count()).sum::<usize>(),
        state.meshes.iter().flatten().count()
    );
    painter.text(
        rect.left_top() + Vec2::new(8.0, 8.0),
        egui::Align2::LEFT_TOP,
        hud,
        egui::FontId::monospace(11.0),
        Color32::from_gray(180),
    );
}

fn handle_camera_input(camera: &mut OrbitCamera, response: &egui::Response, ui: &mut Ui) {
    if response.dragged_by(egui::PointerButton::Primary) {
        let d = response.drag_delta();
        camera.rotate(d.x, d.y);
    }
    if response.dragged_by(egui::PointerButton::Secondary) {
        let d = response.drag_delta();
        camera.pan(d.x, d.y);
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.0 {
            camera.zoom(scroll);
        }
    }
}

fn regenerate(state: &mut State, geometry: &Geometry3D) {
    state.meshes.clear();
    state.error = None;
    let n = state.subdivisions;
    let mut unsupported = Vec::new();
    for (idx, (shape, _op)) in geometry.shapes.iter().enumerate() {
        match shape {
            Shape3D::Cube { center, size } => {
                let extents = nalgebra::Vector3::new(*size, *size, *size);
                state.meshes.push(Some(cuboid::generate(*center, extents, n, n, n)));
            }
            Shape3D::Cuboid { center, extents } => {
                state.meshes.push(Some(cuboid::generate(*center, *extents, n, n, n)));
            }
            other => {
                unsupported.push((idx + 1, other.kind().label()));
                state.meshes.push(None);
            }
        }
    }
    if !unsupported.is_empty() {
        let list = unsupported
            .iter()
            .map(|(i, k)| format!("#{i} {k}"))
            .collect::<Vec<_>>()
            .join(", ");
        state.error = Some(format!(
            "Tet meshing not yet implemented for: {list}. Cube and Cuboid are supported in this milestone."
        ));
    }
}

fn add_stats(state: &State, geometry: &Geometry3D, ui: &mut Ui) {
    ui.label(format!("Shapes in scene: {}", geometry.shapes.len()));
    ui.label(format!(
        "Meshed: {}",
        state.meshes.iter().flatten().count()
    ));
    ScrollArea::vertical().show(ui, |ui| {
        for (idx, (shape, _op)) in geometry.shapes.iter().enumerate() {
            let stats = state
                .meshes
                .get(idx)
                .and_then(|m| m.as_ref())
                .map(|m| format!("{} verts, {} tets", m.vertex_count(), m.tet_count()))
                .unwrap_or_else(|| "—".to_string());
            ui.label(format!("{}. {} → {}", idx + 1, shape.kind().label(), stats));
        }
    });
}
