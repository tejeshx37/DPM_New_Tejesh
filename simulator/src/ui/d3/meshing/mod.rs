//! 3D meshing page. Reads the `Geometry3D` from the drawing state, generates
//! a tetrahedral mesh for each supported primitive, and renders the result
//! in the same orbit-camera viewport used by the drawing page.

mod selection;

use std::collections::HashMap;

use egui::{Color32, ScrollArea, Sense, SidePanel, Slider, TopBottomPanel, Ui, Vec2};
use mesh::d3::{cuboid, cylinder, sphere, DensityHint, Mesh3D};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use super::boundary_conditions;
use super::drawing::{
    shape::{Geometry3D, Shape3D},
    viewport::{camera::OrbitCamera, scene_mesh, wgpu_scene, ViewportState},
};
use super::simulation::RegionBc;

pub use selection::PendingSelection;

/// Which interaction mode the viewport drag gesture is bound to. Camera is
/// the pre-existing default (rotate/pan); the two Select modes let the user
/// drag out a local refinement region on a shape's surface instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MeshToolMode {
    #[default]
    Camera,
    SelectCircle,
    SelectRectangle,
}

/// A user-drawn local mesh-density region, always stored in world space so
/// it stays put across camera moves and project reloads (the drag gesture
/// that created it is screen-space only, projected once at commit time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementRegion {
    pub id: u64,
    pub shape_index: usize,
    pub shape: RefinementShape,
    /// >1.0 refines (denser mesh), <1.0 coarsens. 1.0 is neutral.
    pub density_multiplier: f32,
    /// Blend margin as a fraction of the region's radius/extent.
    pub falloff: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefinementShape {
    Circle {
        center_world: Vector3<f64>,
        radius_world: f64,
        normal_world: Vector3<f64>,
    },
    Rectangle {
        center_world: Vector3<f64>,
        half_extents_world: [f64; 2],
        u_axis: Vector3<f64>,
        v_axis: Vector3<f64>,
    },
}

impl RefinementRegion {
    /// Collapse to an isotropic world-space density hint for the mesh
    /// generators. Rectangles use their largest half-extent as the radius —
    /// a deliberate v1 simplification (no anisotropic per-axis grading yet).
    pub fn to_density_hint(&self) -> DensityHint {
        let (center_world, radius_world) = match &self.shape {
            RefinementShape::Circle {
                center_world,
                radius_world,
                ..
            } => (*center_world, *radius_world),
            RefinementShape::Rectangle {
                center_world,
                half_extents_world,
                ..
            } => (
                *center_world,
                half_extents_world[0].max(half_extents_world[1]),
            ),
        };
        DensityHint {
            center_world,
            radius_world,
            multiplier: self.density_multiplier,
            falloff: self.falloff,
        }
    }
}

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
    /// Serde default matches `Default::default()` so projects saved
    /// before this field existed still deserialize with the slicer
    /// fully open instead of cutting the mesh in half.
    #[serde(default = "default_z_slice_offset")]
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

fn default_z_slice_offset() -> f32 {
    -1.0
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
    /// User-drawn local density regions (circle/rectangle selections on the
    /// viewport). 3D-only feature — see `RefinementRegion` docs.
    #[serde(default)]
    pub refinement_regions: Vec<RefinementRegion>,
    /// Explicit counter (not `Vec` index) so region ids stay stable across
    /// deletions.
    #[serde(default)]
    pub next_region_id: u64,
    #[serde(default)]
    pub tool_mode: MeshToolMode,
    /// In-progress area-selection drag. Never persisted — a saved/reloaded
    /// project should never resume mid-drag.
    #[serde(skip)]
    pub pending_selection: Option<PendingSelection>,
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
            refinement_regions: Vec::new(),
            next_region_id: 1,
            tool_mode: MeshToolMode::default(),
            pending_selection: None,
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
            // Fire when the user commits the new value, by any of:
            //   - Enter while the DragValue is in text-edit mode
            //   - clicking away (loses focus)
            //   - releasing a drag that moved the value off zero
            // Previously required Enter specifically, which silently
            // dropped the change when users typed and clicked elsewhere.
            // egui 0.27 calls this drag_released; renamed in 0.28+.
            #[allow(deprecated)]
            let pc_committed =
                (pc_response.lost_focus() || pc_response.drag_released()) && pc > 0;
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

            ui.separator();
            add_refinement_region_list(state, geometry, ui);
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

    // Computed before input handling so the area-selection tool can
    // raycast against the *current* camera, matching what the user sees.
    let view_proj = state
        .viewport
        .camera
        .view_projection(rect.width() / rect.height().max(1.0));

    let mut regenerate_requested = false;
    match state.tool_mode {
        MeshToolMode::Camera => {
            handle_camera_input(&mut state.viewport.camera, &response, ui);
        }
        MeshToolMode::SelectCircle | MeshToolMode::SelectRectangle => {
            // Scroll/pinch zoom still works in selection mode; only
            // rotate/pan (which consume the same drag gesture as
            // selection) are suppressed.
            if response.hovered() {
                let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
                if scroll.abs() > 0.0 {
                    state.viewport.camera.zoom(scroll);
                }
                if (pinch - 1.0).abs() > 1e-4 {
                    state.viewport.camera.zoom_by(pinch);
                }
            }
            regenerate_requested |=
                selection::handle_selection_input(state, geometry, &response, rect, &view_proj, ui);
        }
    }
    if regenerate_requested {
        regenerate(state, geometry);
    }

    // Auto-rotate: increment yaw a fixed amount per frame, ask egui for
    // a continuous repaint. Suppressed while the user is actively
    // dragging the camera.
    if state.display.auto_rotate && !response.dragged() {
        state.viewport.camera.rotate_yaw(0.005);
        ui.ctx().request_repaint();
    }

    painter.rect_filled(rect, 0.0, Color32::from_gray(20));

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
    // Particles use a bright pure-yellow palette across all bodies so
    // they pop against the dark background and stay easy to spot
    // regardless of fill/edge tint.
    let body_colors: &[([f32; 4], [f32; 4], [f32; 4])] = &[
        ([0.30, 0.50, 0.70, 0.12], [0.40, 0.75, 1.00, 1.0], [1.0, 1.0, 0.15, 1.0]),
        ([0.70, 0.40, 0.40, 0.12], [1.00, 0.55, 0.55, 1.0], [1.0, 0.95, 0.20, 1.0]),
        ([0.40, 0.65, 0.45, 0.12], [0.55, 1.00, 0.70, 1.0], [1.0, 1.00, 0.30, 1.0]),
        ([0.70, 0.60, 0.35, 0.12], [1.00, 0.85, 0.45, 1.0], [1.0, 0.90, 0.10, 1.0]),
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

    // Refinement-region outlines (committed regions + the in-progress drag).
    selection::paint_overlays(state, &painter, rect, &view_proj);

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
        ui.label("Tool:");
        if ui
            .selectable_label(state.tool_mode == MeshToolMode::Camera, "Camera")
            .clicked()
        {
            state.tool_mode = MeshToolMode::Camera;
            state.pending_selection = None;
        }
        if ui
            .selectable_label(state.tool_mode == MeshToolMode::SelectCircle, "◯ Select circle")
            .clicked()
        {
            state.tool_mode = MeshToolMode::SelectCircle;
            state.pending_selection = None;
        }
        if ui
            .selectable_label(state.tool_mode == MeshToolMode::SelectRectangle, "▭ Select rectangle")
            .clicked()
        {
            state.tool_mode = MeshToolMode::SelectRectangle;
            state.pending_selection = None;
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
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        if scroll.abs() > 0.0 {
            camera.zoom(scroll);
        }
        if (pinch - 1.0).abs() > 1e-4 {
            camera.zoom_by(pinch);
        }
    }
}

fn regenerate(state: &mut State, geometry: &Geometry3D) {
    state.meshes.clear();
    state.error = None;
    let n = state.subdivisions;
    for (idx, (shape, _op)) in geometry.shapes.iter().enumerate() {
        let hints: Vec<DensityHint> = state
            .refinement_regions
            .iter()
            .filter(|r| r.shape_index == idx)
            .map(RefinementRegion::to_density_hint)
            .collect();
        let generate_with = |hints: &[DensityHint]| -> Mesh3D {
            match shape {
                Shape3D::Cube { center, size } => {
                    let extents = nalgebra::Vector3::new(*size, *size, *size);
                    cuboid::generate(*center, extents, n, n, n, hints)
                }
                Shape3D::Cuboid { center, extents } => {
                    cuboid::generate(*center, *extents, n, n, n, hints)
                }
                Shape3D::Sphere { center, radius } => sphere::generate(*center, *radius, n, hints),
                Shape3D::Cylinder {
                    base_center,
                    axis,
                    radius,
                    height,
                } => {
                    // Circumferential resolution scales with the user's
                    // subdivisions setting so a single slider controls
                    // overall refinement. Clamp at 3 to keep the mesh closed.
                    let nt = (n * 4).max(3);
                    cylinder::generate(*base_center, *axis, *radius, *height, n, nt, n, hints)
                }
            }
        };

        let mut mesh = generate_with(&hints);
        // Defensive check: extreme multiplier/falloff combos could in
        // principle push the graded spacing into a degenerate (inverted)
        // tetrahedron even with the grading module's clamping. Cheap O(n)
        // scan catches it and falls back to the uniform mesh for this shape
        // rather than handing the solver a corrupted mesh.
        if !hints.is_empty() && has_degenerate_tet(&mesh) {
            mesh = generate_with(&[]);
            state.error = Some(format!(
                "Shape {}: refinement regions produced an invalid mesh and were ignored for it.",
                idx + 1
            ));
        }
        state.meshes.push(Some(mesh));
    }
}

/// Scan for tetrahedra with non-positive signed volume (inverted or
/// degenerate elements), which would corrupt downstream DPM solver
/// consumption.
fn has_degenerate_tet(mesh: &Mesh3D) -> bool {
    mesh.tetrahedra.iter().any(|&[a, b, c, d]| {
        let p0 = mesh.vertices[a];
        let p1 = mesh.vertices[b];
        let p2 = mesh.vertices[c];
        let p3 = mesh.vertices[d];
        let vol = (p1 - p0).cross(&(p2 - p0)).dot(&(p3 - p0));
        vol.abs() < 1e-15
    })
}

/// Region list panel: one row per drawn refinement region, with a density
/// slider and delete button. Slider commit and delete both trigger a
/// regeneration, same as editing the global subdivisions slider.
fn add_refinement_region_list(state: &mut State, geometry: &Geometry3D, ui: &mut Ui) {
    ui.heading("Refinement Regions");
    if state.refinement_regions.is_empty() {
        ui.label("Draw a circle/rectangle on a shape to add one.");
        return;
    }

    let mut to_delete: Option<usize> = None;
    let mut needs_regenerate = false;
    ScrollArea::vertical()
        .max_height(220.0)
        .show(ui, |ui| {
            for (i, region) in state.refinement_regions.iter_mut().enumerate() {
                ui.group(|ui| {
                    let shape_label = geometry
                        .shapes
                        .get(region.shape_index)
                        .map(|(s, _)| s.kind().label())
                        .unwrap_or("—");
                    let kind = match region.shape {
                        RefinementShape::Circle { .. } => "circle",
                        RefinementShape::Rectangle { .. } => "rectangle",
                    };
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "R{}: {} on shape {} ({})",
                            region.id,
                            kind,
                            region.shape_index + 1,
                            shape_label
                        ));
                        if ui.small_button("✕").clicked() {
                            to_delete = Some(i);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("density:");
                        let resp = ui.add(
                            Slider::new(&mut region.density_multiplier, 0.1..=5.0).show_value(true),
                        );
                        #[allow(deprecated)]
                        if resp.drag_released() || resp.lost_focus() {
                            needs_regenerate = true;
                        }
                    });
                });
            }
        });

    if let Some(i) = to_delete {
        state.refinement_regions.remove(i);
        needs_regenerate = true;
    }
    if needs_regenerate {
        regenerate(state, geometry);
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
