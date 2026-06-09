//! 3D meshing page. Reads the `Geometry3D` from the drawing state, generates
//! a tetrahedral mesh for each supported primitive, and renders the result
//! in the same orbit-camera viewport used by the drawing page.

use egui::{Color32, ScrollArea, Sense, SidePanel, Slider, Ui, Vec2};
use mesh::d3::{cuboid, cylinder, sphere, Mesh3D};
use serde::{Deserialize, Serialize};

use super::drawing::{
    shape::{Geometry3D, Shape3D},
    viewport::{camera::OrbitCamera, scene_mesh, wgpu_scene, ViewportState},
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
            // Cap at 60 so a single cuboid can reach 60^3 ≈ 220k vertices,
            // well above the 10k-particle target users typically want.
            ui.add(Slider::new(&mut state.subdivisions, 1..=60));
            // Quick reference for cuboid vertex counts so users know what
            // they're picking before they hit Generate.
            let n = state.subdivisions as usize;
            ui.label(format!(
                "cuboid: ≈{} verts, ≈{} tets",
                (n + 1).pow(3),
                n.pow(3) * 6
            ));

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

    // Render each generated mesh's boundary surface through the wgpu
    // scene callback (filled, shaded triangles via its own vertex
    // buffer). Previously we drew every tet's 6 edges through
    // egui::Painter, which routes into egui's shared vertex buffer and
    // blows past wgpu's 256 MB max for any mesh over ~30^3 voxels.
    // Surface-only rendering is O(n^2) in subdivisions instead of
    // O(n^3) and scales to the 60-subdivision cap without crashing.
    let mut all_tris: Vec<wgpu_scene::Vertex> = Vec::new();
    for (idx, mesh) in state.meshes.iter().flatten().enumerate() {
        // Per-body tint so adjacent bodies are distinguishable.
        let tint = match idx % 4 {
            0 => [0.70, 0.86, 1.00, 0.95],
            1 => [1.00, 0.78, 0.78, 0.95],
            2 => [0.78, 1.00, 0.84, 0.95],
            _ => [1.00, 0.92, 0.70, 0.95],
        };
        all_tris.extend(scene_mesh::triangles_for_mesh(mesh, |_| tint));
    }
    if !all_tris.is_empty() {
        wgpu_scene::sort_back_to_front(&mut all_tris, &view_proj);
        let cb = wgpu_scene::SceneCallback::from_world_mvp(all_tris, &view_proj);
        painter.add(eframe::egui_wgpu::Callback::new_paint_callback(rect, cb));
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
    for (_idx, (shape, _op)) in geometry.shapes.iter().enumerate() {
        let mesh = match shape {
            Shape3D::Cube { center, size } => {
                let extents = nalgebra::Vector3::new(*size, *size, *size);
                cuboid::generate(*center, extents, n, n, n)
            }
            Shape3D::Cuboid { center, extents } => {
                cuboid::generate(*center, *extents, n, n, n)
            }
            Shape3D::Sphere { center, radius } => sphere::generate(*center, *radius, n),
            Shape3D::Cylinder {
                base_center,
                axis,
                radius,
                height,
            } => {
                // Circumferential resolution scales with the user's
                // subdivisions setting so a single slider controls overall
                // refinement. Clamp at 3 to keep the mesh closed.
                let nt = (n * 4).max(3);
                cylinder::generate(*base_center, *axis, *radius, *height, n, nt, n)
            }
        };
        state.meshes.push(Some(mesh));
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
