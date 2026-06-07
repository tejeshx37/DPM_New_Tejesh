//! 3D simulation page.
//!
//! Workflow: user picks a mesh from the meshing page, assigns a boundary
//! condition to each named boundary region (Free / Pinned / Constant
//! Force / Constant Displacement), tunes material + integration
//! parameters, and runs the solver. The viewport renders the deformed
//! mesh with per-tet edges colored by Von Mises stress.

use std::collections::HashMap;

use cpd::d3::{Axis, BoundaryCondition3D, Computer3D, Config3D, IsotropicProps3D, StressStats};
use egui::{Color32, ComboBox, DragValue, ScrollArea, Sense, SidePanel, Slider, Stroke, Ui, Vec2};
use mesh::d3::Mesh3D;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use super::drawing::viewport::{camera::OrbitCamera, project, ViewportState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Index into the meshing state's `meshes` Vec selecting which mesh to
    /// simulate. Single-mesh for this milestone; multi-body scenes are a
    /// future iteration.
    #[serde(default)]
    pub selected_mesh: usize,
    /// Per-region BC choice, keyed by region name (e.g. "x_min").
    #[serde(default)]
    pub region_bcs: HashMap<String, RegionBc>,
    #[serde(default)]
    pub material: IsotropicProps3D,
    #[serde(default = "default_time_delta")]
    pub time_delta: f32,
    #[serde(default = "default_duration")]
    pub duration: f32,
    #[serde(default = "default_steps_per_frame")]
    pub steps_per_frame: u32,
    #[serde(default)]
    pub running: bool,
    #[serde(default)]
    pub viewport: ViewportState,
    /// Live solver state. Not persisted; rebuilt on demand.
    #[serde(skip)]
    pub computer: Option<Computer3D>,
    /// Latest stress stats from the running solver.
    #[serde(skip)]
    pub stats: StressStats,
}

fn default_time_delta() -> f32 {
    1.0e-5
}
fn default_duration() -> f32 {
    1.0e-2
}
fn default_steps_per_frame() -> u32 {
    50
}

impl Default for State {
    fn default() -> Self {
        Self {
            selected_mesh: 0,
            region_bcs: HashMap::new(),
            material: IsotropicProps3D::default(),
            time_delta: default_time_delta(),
            duration: default_duration(),
            steps_per_frame: default_steps_per_frame(),
            running: false,
            viewport: ViewportState::default(),
            computer: None,
            stats: StressStats::default(),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RegionBc {
    pub kind: BcKind,
    pub axes: Axis,
    pub force: [f32; 3],
    pub displacement: [f32; 3],
    pub ramp_seconds: f32,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BcKind {
    #[default]
    Free,
    Pinned,
    ConstantForce,
    ConstantDisplacement,
}

impl BcKind {
    fn label(self) -> &'static str {
        match self {
            BcKind::Free => "Free",
            BcKind::Pinned => "Pinned",
            BcKind::ConstantForce => "Constant Force",
            BcKind::ConstantDisplacement => "Constant Displacement",
        }
    }
}

impl RegionBc {
    fn to_bc(&self) -> BoundaryCondition3D {
        match self.kind {
            BcKind::Free => BoundaryCondition3D::Free,
            BcKind::Pinned => BoundaryCondition3D::Pinned { axes: self.axes },
            BcKind::ConstantForce => BoundaryCondition3D::ConstantForce { force: self.force },
            BcKind::ConstantDisplacement => BoundaryCondition3D::ConstantDisplacement {
                axes: self.axes,
                displacement: self.displacement,
                ramp_seconds: self.ramp_seconds,
            },
        }
    }
}

pub fn show(state: &mut State, meshes: &[Option<Mesh3D>], ui: &mut Ui) {
    SidePanel::right("d3_sim_side_panel")
        .resizable(true)
        .default_width(300.0)
        .show_inside(ui, |ui| {
            ui.heading("3D Simulation");
            ui.separator();
            add_mesh_picker(state, meshes, ui);

            let active_mesh = meshes
                .get(state.selected_mesh)
                .and_then(|m| m.as_ref());

            if active_mesh.is_none() {
                ui.colored_label(
                    Color32::YELLOW,
                    "No mesh available. Generate one on the Meshing page first.",
                );
                return;
            }
            let mesh = active_mesh.unwrap();

            ui.collapsing("Material", |ui| add_material_controls(state, ui));
            ui.collapsing("Integration", |ui| add_integration_controls(state, ui));
            ui.collapsing("Boundary Conditions", |ui| {
                add_bc_controls(state, mesh, ui)
            });

            ui.separator();
            add_run_controls(state, mesh, ui);
            add_stats(state, ui);
        });

    if state.running {
        if let Some(c) = state.computer.as_mut() {
            let steps = state.steps_per_frame as u64;
            cpd::d3::run_steps(c, steps);
            state.stats = c.stress_stats();
            if c.time() >= state.duration {
                state.running = false;
            }
        }
        ui.ctx().request_repaint();
    }

    show_viewport(state, meshes, ui);
}

fn add_mesh_picker(state: &mut State, meshes: &[Option<Mesh3D>], ui: &mut Ui) {
    let available: Vec<usize> = meshes
        .iter()
        .enumerate()
        .filter_map(|(i, m)| m.as_ref().map(|_| i))
        .collect();
    if available.is_empty() {
        return;
    }
    if !available.contains(&state.selected_mesh) {
        state.selected_mesh = available[0];
    }
    ComboBox::from_label("Mesh to simulate")
        .selected_text(format!("Mesh #{}", state.selected_mesh + 1))
        .show_ui(ui, |ui| {
            for i in &available {
                ui.selectable_value(&mut state.selected_mesh, *i, format!("Mesh #{}", i + 1));
            }
        });
}

fn add_material_controls(state: &mut State, ui: &mut Ui) {
    let m = &mut state.material;
    ui.horizontal(|ui| {
        ui.label("Young's modulus (E)");
        ui.add(DragValue::new(&mut m.elasticity_modulus).speed(1000.0).clamp_range(1.0..=f32::MAX));
    });
    ui.horizontal(|ui| {
        ui.label("Poisson's ratio (ν)");
        ui.add(DragValue::new(&mut m.poissons_ratio).speed(0.01).clamp_range(0.0..=0.49));
    });
    ui.horizontal(|ui| {
        ui.label("Density (ρ)");
        ui.add(DragValue::new(&mut m.density).speed(10.0).clamp_range(1e-6..=f32::MAX));
    });
    ui.horizontal(|ui| {
        ui.label("Damping (c)");
        ui.add(DragValue::new(&mut m.damping).speed(0.1).clamp_range(0.0..=f32::MAX));
    });
}

fn add_integration_controls(state: &mut State, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.label("Time step (s)");
        ui.add(
            DragValue::new(&mut state.time_delta)
                .speed(1e-6)
                .clamp_range(1e-9..=1.0)
                .min_decimals(6)
                .max_decimals(9),
        );
    });
    ui.horizontal(|ui| {
        ui.label("Duration (s)");
        ui.add(DragValue::new(&mut state.duration).speed(0.001).clamp_range(1e-6..=f32::MAX));
    });
    ui.horizontal(|ui| {
        ui.label("Steps/frame");
        ui.add(Slider::new(&mut state.steps_per_frame, 1..=1000));
    });
}

fn add_bc_controls(state: &mut State, mesh: &Mesh3D, ui: &mut Ui) {
    for region in &mesh.boundary_faces.regions {
        let entry = state.region_bcs.entry(region.name.clone()).or_default();
        ui.group(|ui| {
            ui.strong(format!(
                "{} ({} verts)",
                region.name,
                region.vertices.len()
            ));
            ComboBox::from_id_source(format!("bc_kind_{}", region.name))
                .selected_text(entry.kind.label())
                .show_ui(ui, |ui| {
                    for k in [
                        BcKind::Free,
                        BcKind::Pinned,
                        BcKind::ConstantForce,
                        BcKind::ConstantDisplacement,
                    ] {
                        ui.selectable_value(&mut entry.kind, k, k.label());
                    }
                });
            match entry.kind {
                BcKind::Free => {}
                BcKind::Pinned => {
                    axes_row(ui, &mut entry.axes);
                }
                BcKind::ConstantForce => {
                    vec_row(ui, "Force", &mut entry.force);
                }
                BcKind::ConstantDisplacement => {
                    axes_row(ui, &mut entry.axes);
                    vec_row(ui, "Target", &mut entry.displacement);
                    ui.horizontal(|ui| {
                        ui.label("Ramp (s)");
                        ui.add(
                            DragValue::new(&mut entry.ramp_seconds)
                                .speed(0.001)
                                .clamp_range(0.0..=f32::MAX),
                        );
                    });
                }
            }
        });
    }
}

fn axes_row(ui: &mut Ui, axes: &mut Axis) {
    ui.horizontal(|ui| {
        ui.label("Axes:");
        ui.checkbox(&mut axes.x, "X");
        ui.checkbox(&mut axes.y, "Y");
        ui.checkbox(&mut axes.z, "Z");
    });
}

fn vec_row(ui: &mut Ui, label: &str, v: &mut [f32; 3]) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(DragValue::new(&mut v[0]).speed(0.1));
        ui.add(DragValue::new(&mut v[1]).speed(0.1));
        ui.add(DragValue::new(&mut v[2]).speed(0.1));
    });
}

fn add_run_controls(state: &mut State, mesh: &Mesh3D, ui: &mut Ui) {
    ui.horizontal(|ui| {
        if ui.button(if state.running { "Pause" } else { "Run" }).clicked() {
            if state.computer.is_none() {
                rebuild_computer(state, mesh);
            }
            state.running = !state.running;
        }
        if ui.button("Reset").clicked() {
            state.running = false;
            if let Some(c) = state.computer.as_mut() {
                c.reset();
                state.stats = StressStats::default();
            }
        }
        if ui.button("Rebuild").clicked() {
            state.running = false;
            rebuild_computer(state, mesh);
        }
    });
}

fn rebuild_computer(state: &mut State, mesh: &Mesh3D) {
    let verts_f32: Vec<Vector3<f32>> = mesh
        .vertices
        .iter()
        .map(|v| Vector3::new(v.x as f32, v.y as f32, v.z as f32))
        .collect();
    let cfg = Config3D {
        material: state.material,
        time_delta_seconds: state.time_delta,
        duration_seconds: state.duration,
    };
    let Some(mut c) = Computer3D::from_mesh(&verts_f32, &mesh.tetrahedra, cfg) else {
        return;
    };
    // Apply BCs: for each boundary region in the mesh, look up the user's
    // selection and apply to those node indices.
    for region in &mesh.boundary_faces.regions {
        let Some(entry) = state.region_bcs.get(&region.name) else {
            continue;
        };
        let bc = entry.to_bc();
        c.set_bc(&region.vertices, bc);
    }
    state.computer = Some(c);
    state.stats = StressStats::default();
}

fn add_stats(state: &State, ui: &mut Ui) {
    ui.separator();
    if let Some(c) = state.computer.as_ref() {
        ui.label(format!("Time: {:.5} s", c.time()));
        ui.label(format!("Iterations: {}", c.iterations));
        ui.label(format!(
            "Von Mises stress  min/mean/max:  {:.3e} / {:.3e} / {:.3e}",
            state.stats.min_von_mises, state.stats.mean_von_mises, state.stats.max_von_mises
        ));
    } else {
        ui.label("Solver not built. Click Run to initialize.");
    }
}

fn show_viewport(state: &mut State, meshes: &[Option<Mesh3D>], ui: &mut Ui) {
    let available = ui.available_size();
    let size = Vec2::new(available.x.max(100.0), available.y.max(100.0));
    let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
    let rect = response.rect;

    // Auto-frame on first render.
    let aabb = meshes
        .get(state.selected_mesh)
        .and_then(|m| m.as_ref())
        .map(|m| {
            let lo = m
                .vertices
                .iter()
                .fold(Vector3::repeat(f64::INFINITY), |acc, v| acc.inf(v));
            let hi = m
                .vertices
                .iter()
                .fold(Vector3::repeat(f64::NEG_INFINITY), |acc, v| acc.sup(v));
            (lo, hi)
        });

    if state.viewport.auto_frame {
        if let Some((lo, hi)) = aabb {
            state.viewport.camera.frame_aabb(lo, hi);
            state.viewport.auto_frame = false;
        }
    }

    handle_camera_input(&mut state.viewport.camera, &response, ui);

    painter.rect_filled(rect, 0.0, Color32::from_gray(20));

    let view_proj = state
        .viewport
        .camera
        .view_projection(rect.width() / rect.height().max(1.0));

    // If we have a running computer, draw the deformed mesh colored by
    // per-tet Von Mises stress. Otherwise fall back to the reference mesh
    // wireframe.
    if let Some(c) = state.computer.as_ref() {
        let s_min = state.stats.min_von_mises;
        let s_max = state.stats.max_von_mises.max(s_min + 1e-12);
        for e in &c.elements {
            let vm = von_mises_of(&e.stress);
            let t = ((vm - s_min) / (s_max - s_min)).clamp(0.0, 1.0);
            let color = heatmap(t);
            let stroke = Stroke::new(0.6, color);
            let p: [Vector3<f64>; 4] = [
                vec3_f64(&c.nodes[e.indices[0]].position),
                vec3_f64(&c.nodes[e.indices[1]].position),
                vec3_f64(&c.nodes[e.indices[2]].position),
                vec3_f64(&c.nodes[e.indices[3]].position),
            ];
            for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                if let (Some(pa), Some(pb)) = (
                    project(&view_proj, rect, p[a]),
                    project(&view_proj, rect, p[b]),
                ) {
                    painter.line_segment([pa, pb], stroke);
                }
            }
        }
    } else if let Some(mesh) = meshes.get(state.selected_mesh).and_then(|m| m.as_ref()) {
        let stroke = Stroke::new(0.5, Color32::from_gray(140));
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
}

fn vec3_f64(v: &Vector3<f32>) -> Vector3<f64> {
    Vector3::new(v.x as f64, v.y as f64, v.z as f64)
}

fn von_mises_of(s: &nalgebra::Matrix3<f32>) -> f32 {
    let d12 = s.m11 - s.m22;
    let d23 = s.m22 - s.m33;
    let d31 = s.m33 - s.m11;
    let sh = s.m12 * s.m12 + s.m23 * s.m23 + s.m13 * s.m13;
    (0.5 * (d12 * d12 + d23 * d23 + d31 * d31 + 6.0 * sh)).sqrt()
}

/// Simple blue→cyan→green→yellow→red heatmap.
fn heatmap(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
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
