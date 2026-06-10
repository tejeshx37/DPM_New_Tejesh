//! 3D meshing page. Reads the `Geometry3D` from the drawing state, generates
//! a tetrahedral mesh for each supported primitive, and renders the result
//! in the same orbit-camera viewport used by the drawing page.

use std::collections::HashMap;

use egui::{Color32, ScrollArea, Sense, SidePanel, Slider, TopBottomPanel, Ui, Vec2};
use mesh::d3::{cuboid, cylinder, sphere, Mesh3D};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use super::boundary_conditions;
use super::drawing::{
    shape::{Geometry3D, Shape3D},
    viewport::{camera::OrbitCamera, scene_mesh, wgpu_scene, ViewportState},
};
use super::simulation::RegionBc;

/// Toggle bar at the top of the viewport, matching the reference UI:
/// regenerate, reset view, particles, wireframe-only, z-slice,
/// auto-rotate, hide mesh, show constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayFlags {
    #[serde(default = "default_true")]
    pub show_constraints: bool,
    #[serde(default = "default_true")]
    pub show_particles: bool,
    #[serde(default)]
    pub show_wireframe_only: bool,
    #[serde(default)]
    pub enable_z_slice: bool,
    /// Z-slice plane offset in normalized [-1, 1] range across the mesh
    /// AABB along the +Z axis. -1 keeps everything; +1 hides everything.
    #[serde(default)]
    pub z_slice_offset: f32,
    #[serde(default)]
    pub auto_rotate: bool,
    #[serde(default)]
    pub hide_mesh: bool,
}

impl Default for DisplayFlags {
    fn default() -> Self {
        Self {
            show_constraints: true,
            show_particles: true,
            show_wireframe_only: false,
            enable_z_slice: false,
            z_slice_offset: -1.0,
            auto_rotate: false,
            hide_mesh: false,
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// User-controlled subdivision count (applied to all axes for cubes;
    /// scaled by extents ratio for cuboids).
    #[serde(default = "default_subdivisions")]
    pub subdivisions: u32,
    /// Desired particle (vertex) count. When the user edits this, subdivisions
    /// are auto-derived via cube root. `0` means manual subdivision mode.
    #[serde(default)]
    pub particle_count: u32,
    /// Generated meshes, one per source shape. Cleared when subdivisions or
    /// geometry change. Indices match `geometry.shapes`.
    #[serde(default)]
    pub meshes: Vec<Option<Mesh3D>>,
    #[serde(default)]
    pub viewport: ViewportState,
    /// Last error message from mesh generation, if any.
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub display: DisplayFlags,
}

fn default_subdivisions() -> u32 {
    4
}

impl Default for State {
    fn default() -> Self {
        Self {
            subdivisions: 4,
            particle_count: 0,
            meshes: Vec::new(),
            viewport: ViewportState::default(),
            error: None,
            display: DisplayFlags::default(),
        }
    }
}

/// Derive subdivisions from a desired particle (vertex) count.
/// Vertex count for a cuboid mesh ≈ (n+1)^3, so n ≈ count^(1/3) - 1.
fn subdivisions_from_particle_count(count: u32) -> u32 {
    if count <= 1 {
        return 1;
    }
    let n = (count as f64).cbrt().round() as u32;
    n.saturating_sub(1).max(1).min(60)
}

pub fn show(
    state: &mut State,
    geometry: &Geometry3D,
    region_bcs: &HashMap<String, RegionBc>,
    ui: &mut Ui,
) {
    // Top toolbar with the feature toggles users expect from comparable
    // mesher UIs (reference image in chat).
    TopBottomPanel::top("d3_meshing_toolbar")
        .show_inside(ui, |ui| add_toolbar(state, geometry, ui));

    SidePanel::right("d3_meshing_side_panel")
        .resizable(true)
        .default_width(240.0)
        .show_inside(ui, |ui| {
            ui.heading("3D Meshing");

            if geometry.shapes.is_empty() {
                ui.colored_label(
                    Color32::YELLOW,
                    "No shapes drawn yet. Add at least one shape on the Drawing page first.",
                );
                return;
            }

            // --- Particle count input ---
            // Auto-regenerates on Enter or focus loss so the user
            // doesn't also have to click "Generate Mesh" after typing
            // a number. Live-drag changes only update the predicted
            // subdivision count to avoid hammering the mesher.
            ui.label("Desired particle count:");
            let mut pc = state.particle_count;
            let pc_response = ui.add(
                egui::DragValue::new(&mut pc)
                    .speed(10.0)
                    .clamp_range(0..=300_000u32),
            );
            if pc_response.changed() {
                state.particle_count = pc;
                if pc > 0 {
                    state.subdivisions = subdivisions_from_particle_count(pc);
                }
            }
            // egui 0.27 calls this drag_released; renamed to
            // dragged_stopped in a later release.
            #[allow(deprecated)]
            let pc_committed = pc_response.lost_focus()
                && ui.input(|i| i.key_pressed(egui::Key::Enter))
                || (pc_response.drag_released() && pc > 0);
            if state.particle_count > 0 {
                ui.label(format!(
                    "→ subdivisions: {}  (≈{} actual particles)",
                    state.subdivisions,
                    (state.subdivisions as usize + 1).pow(3)
                ));
            }
            ui.add_space(2.0);

            // --- Manual subdivisions slider ---
            ui.label("Subdivisions per axis:");
            // Cap at 60 so a single cuboid can reach 60^3 ≈ 220k vertices,
            // well above the 10k-particle target users typically want.
            let sub_response = ui.add(Slider::new(&mut state.subdivisions, 1..=60));
            if sub_response.changed() {
                // When the user drags the slider manually, clear the particle
                // count so it doesn't fight back.
                state.particle_count = 0;
            }
            #[allow(deprecated)]
            let sub_committed = sub_response.drag_released();
            // Quick reference for cuboid vertex counts so users know what
            // they're picking before they hit Generate.
            let n = state.subdivisions as usize;
            ui.label(format!(
                "cuboid: ≈{} verts, ≈{} tets",
                (n + 1).pow(3),
                n.pow(3) * 6
            ));

            ui.add_space(4.0);
            if ui.button("Generate Mesh").clicked() || pc_committed || sub_committed {
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

    // Auto-rotate: increment yaw a fixed amount per frame, ask egui for
    // a continuous repaint. Suppressed while the user is actively
    // dragging the camera.
    if state.display.auto_rotate && !response.dragged() {
        state.viewport.camera.rotate_yaw(0.005);
        ui.ctx().request_repaint();
    }

    painter.rect_filled(rect, 0.0, Color32::from_gray(20));

    let view_proj = state
        .viewport
        .camera
        .view_projection(rect.width() / rect.height().max(1.0));

    // Render wireframe via wgpu: a faint transparent solid surface for depth
    // perception + bright wireframe edge quads on top. Both go into a single
    // wgpu draw call (batched vertex buffer), so this is GPU-accelerated and
    // scales to 60+ subdivisions without freezing the UI.
    let mut all_verts: Vec<wgpu_scene::Vertex> = Vec::new();

    // Edge thickness adapts to camera distance so edges stay visible.
    let cam_dist = state.viewport.camera.distance();
    let edge_thickness = cam_dist * 0.003;
    let particle_size = cam_dist * 0.004;

    // Per-body colors: faint fill + bright wireframe edge + particle dot.
    let body_colors: &[([f32; 4], [f32; 4], [f32; 4])] = &[
        ([0.30, 0.50, 0.70, 0.12], [0.40, 0.75, 1.00, 1.0], [1.00, 0.78, 0.30, 1.0]),
        ([0.70, 0.40, 0.40, 0.12], [1.00, 0.55, 0.55, 1.0], [1.00, 0.85, 0.40, 1.0]),
        ([0.40, 0.65, 0.45, 0.12], [0.55, 1.00, 0.70, 1.0], [1.00, 0.90, 0.55, 1.0]),
        ([0.70, 0.60, 0.35, 0.12], [1.00, 0.85, 0.45, 1.0], [0.90, 0.95, 0.45, 1.0]),
    ];

    let (cam_right, cam_up, _cam_fwd) = state.viewport.camera.basis_world();

    for (idx, mesh) in state.meshes.iter().flatten().enumerate() {
        let (fill_color, edge_color, particle_color) = body_colors[idx % body_colors.len()];

        // Filled surface — skipped if user wants wireframe-only or
        // hid the mesh entirely.
        if !state.display.hide_mesh && !state.display.show_wireframe_only {
            all_verts.extend(scene_mesh::triangles_for_mesh(mesh, |_| fill_color));
        }
        // Wireframe edges — visible unless mesh is fully hidden.
        if !state.display.hide_mesh {
            all_verts.extend(scene_mesh::wireframe_edges_for_mesh(
                mesh,
                edge_color,
                edge_thickness,
            ));
        }
        // Particles — every mesh vertex as a small camera-facing quad.
        if state.display.show_particles {
            all_verts.extend(scene_mesh::particles_for_mesh(
                mesh,
                cam_right,
                cam_up,
                particle_size,
                particle_color,
            ));
        }
    }

    // Z-Slice: clip the assembled triangle stream against a plane along
    // +Z anchored to the mesh AABB.
    if state.display.enable_z_slice && !all_verts.is_empty() {
        let (lo, hi) = combined_aabb(&state.meshes);
        let z_lo = lo.z as f32;
        let z_hi = hi.z as f32;
        let offset = z_lo + (state.display.z_slice_offset * 0.5 + 0.5) * (z_hi - z_lo);
        scene_mesh::clip_triangles_by_plane(
            &mut all_verts,
            Vector3::new(0.0, 0.0, 1.0),
            offset,
        );
    }

    if !all_verts.is_empty() {
        wgpu_scene::sort_back_to_front(&mut all_verts, &view_proj);
        let cb = wgpu_scene::SceneCallback::from_world_mvp(all_verts, &view_proj);
        painter.add(eframe::egui_wgpu::Callback::new_paint_callback(rect, cb));
    }

    // Constraint overlay arrows on top of the wgpu scene.
    if state.display.show_constraints {
        boundary_conditions::paint_constraint_overlays(
            &painter, rect, &view_proj, geometry, region_bcs,
        );
    }

    // Stats HUD matching the reference UI layout: "Elements: N   Points: N".
    let total_tets: usize = state.meshes.iter().flatten().map(|m| m.tet_count()).sum();
    let total_verts: usize = state.meshes.iter().flatten().map(|m| m.vertex_count()).sum();
    let hud = format!(
        "Elements: {}   Points: {}   |   LMB rotate · RMB pan · scroll/+/- zoom",
        total_tets, total_verts
    );
    painter.text(
        rect.left_bottom() + Vec2::new(8.0, -22.0),
        egui::Align2::LEFT_BOTTOM,
        hud,
        egui::FontId::monospace(11.0),
        Color32::from_gray(190),
    );
}

/// AABB of every populated mesh, for the Z-slice plane anchoring.
fn combined_aabb(meshes: &[Option<Mesh3D>]) -> (Vector3<f64>, Vector3<f64>) {
    let mut lo = Vector3::repeat(f64::INFINITY);
    let mut hi = Vector3::repeat(f64::NEG_INFINITY);
    for mesh in meshes.iter().flatten() {
        for v in &mesh.vertices {
            lo = lo.inf(v);
            hi = hi.sup(v);
        }
    }
    if !lo.x.is_finite() {
        lo = Vector3::repeat(0.0);
        hi = Vector3::repeat(1.0);
    }
    (lo, hi)
}

fn add_toolbar(state: &mut State, geometry: &Geometry3D, ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        if ui.button("⟳ Regenerate mesh").clicked() {
            regenerate(state, geometry);
        }
        if ui.button("⌖ Reset view").clicked() {
            state.viewport.auto_frame = true;
        }
        ui.separator();
        ui.checkbox(&mut state.display.show_constraints, "Show constraints");
        ui.checkbox(&mut state.display.show_particles, "Show particles");
        ui.checkbox(&mut state.display.show_wireframe_only, "Show wireframe only");
        ui.checkbox(&mut state.display.enable_z_slice, "Enable Z-Slice");
        ui.checkbox(&mut state.display.auto_rotate, "Auto-Rotate 360°");
        ui.checkbox(&mut state.display.hide_mesh, "Hide mesh");
        if state.display.enable_z_slice {
            ui.label("slice:");
            ui.add(Slider::new(&mut state.display.z_slice_offset, -1.0..=1.0).show_value(false));
        }
    });
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
