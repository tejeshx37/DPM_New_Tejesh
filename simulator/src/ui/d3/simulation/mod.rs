//! 3D simulation page.
//!
//! Workflow: user picks a mesh from the meshing page, assigns a boundary
//! condition to each named boundary region (Free / Pinned / Constant
//! Force / Constant Displacement), tunes material + integration
//! parameters, and runs the solver. The viewport renders the deformed
//! mesh with per-tet edges colored by Von Mises stress.

use std::collections::HashMap;
use std::path::PathBuf;

pub use cpd::d3::Axis;
use cpd::d3::{
    AxisTimeSeries, BoundaryCondition3D, Computer3D, Config3D, FailureCriteria3D,
    IsotropicProps3D, MaterialProps3D, OrthotropicProps3D, RegionAverages, StressStats,
};
use egui::{Color32, ComboBox, DragValue, Sense, SidePanel, Slider, Stroke, TopBottomPanel, Ui, Vec2};
use egui_plot::{Line, Plot, PlotPoints};
use mesh::d3::Mesh3D;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

pub mod engine_config;
pub mod export;
pub mod gpu;

use super::drawing::viewport::{
    camera::OrbitCamera, project, scene_mesh, wgpu_scene, ViewportState,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct State {
    /// Legacy field from the single-mesh era — kept so old `.simproj`
    /// files still deserialize. Multi-body scenes (B11) now always
    /// combine every populated mesh in the scene into a single solver.
    #[serde(default)]
    #[allow(dead_code)]
    pub selected_mesh: usize,
    /// Per-region BC choice, keyed by region name (e.g. "x_min").
    #[serde(default)]
    pub region_bcs: HashMap<String, RegionBc>,
    #[serde(default)]
    pub material: MaterialProps3D,
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
    /// Live solver state. Persisted across sessions (D16) so a paused
    /// simulation resumes mid-run on reopen; defaulted on first load.
    #[serde(default)]
    pub computer: Option<Computer3D>,
    /// Latest stress stats from the running solver.
    #[serde(default)]
    pub stats: StressStats,
    /// Time-series captured during simulation runs (cleared on Reset /
    /// Rebuild). Persisted alongside the solver.
    #[serde(default)]
    pub history: History,
    /// Plots panel visibility + tunables.
    #[serde(default)]
    pub plots: PlotConfig,
    /// Per-element / per-vertex inspection picks (A5). One-based index in
    /// the UI for ergonomics; converted to 0-based when accessed.
    #[serde(default)]
    pub inspect: InspectConfig,
    /// Last directory the user exported CSV into (B10). Used as the
    /// default for the next export so they don't pick repeatedly.
    #[serde(default)]
    pub last_export_dir: Option<PathBuf>,
    /// Last export status message (#files written or error string).
    #[serde(skip)]
    pub last_export_status: Option<String>,
    /// Opt-in GPU strain/stress kernel (E18). Off by default; the kernel
    /// is built lazily on first use and only supports isotropic material.
    #[serde(default)]
    pub use_gpu_stresses: bool,
    /// Lazy-initialised GPU compute kernel. Not persisted; rebuilt
    /// on demand.
    #[serde(skip)]
    pub gpu_kernel: Option<gpu::GpuStressKernel>,
    /// Last GPU init status — error message if init failed, "ready"
    /// if it succeeded.
    #[serde(skip)]
    pub gpu_status: Option<String>,
    /// User-pinned GPU adapter override. When `Some`, the kernel tries
    /// to bind to this specific adapter on next build instead of using
    /// wgpu's HighPerformance auto-pick. Persisted so the choice
    /// survives a restart.
    #[serde(default)]
    pub gpu_preferred_adapter: Option<gpu::AdapterDisplay>,
    /// Cached list of adapters wgpu can see on the host. Populated on
    /// first open of the GPU panel; not persisted.
    #[serde(skip)]
    pub gpu_available_adapters: Vec<gpu::AdapterDisplay>,
    /// Whether the Engine Config modal is currently open. Lets users
    /// see every simulation parameter in one place instead of clicking
    /// through individual side-panel collapsibles.
    #[serde(skip)]
    pub engine_config_open: bool,
    /// Constant body acceleration (gravity-like). Applied per step as
    /// `force += mass * body_force`. Stored as f32 triple for the
    /// Engine Config dialog binding.
    #[serde(default)]
    pub body_force: [f32; 3],
    /// Auto-export at end-of-run flag. The CSV button on the side
    /// panel still does an on-demand export; this turns it into an
    /// automatic action when the duration is reached.
    #[serde(default)]
    pub auto_export: bool,
    /// Render every solver node as a billboard quad in the Simulation
    /// viewport. Lets users see particles moving and the crack
    /// propagating through the body, not just on the surface.
    #[serde(default = "default_true")]
    pub sim_show_particles: bool,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct InspectConfig {
    /// 1-based index, 0 = disabled.
    #[serde(default)]
    pub element_index: u32,
    /// 1-based index, 0 = disabled.
    #[serde(default)]
    pub vertex_index: u32,
}

/// History buffer for time-series plotting. Capped at `MAX_SAMPLES` with
/// drop-oldest behavior so long runs don't grow unbounded. Serializable
/// so a paused simulation can be saved and resumed across sessions (D16).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct History {
    pub stress: Vec<(f32, StressStats)>,
    /// Per-region (mean_displacement, mean_force) over time. Key is the
    /// boundary region name from the mesh.
    pub regions: std::collections::HashMap<String, Vec<(f32, RegionAverages)>>,
    /// Inspected element Von Mises stress history (A5).
    pub element: Vec<(f32, f32)>,
    /// Inspected vertex (displacement magnitude, force magnitude) history (A5).
    pub vertex: Vec<(f32, f32, f32)>,
}

const MAX_SAMPLES: usize = 5000;

impl History {
    pub fn clear(&mut self) {
        self.stress.clear();
        self.regions.clear();
        self.element.clear();
        self.vertex.clear();
    }

    fn push_stress(&mut self, t: f32, s: StressStats) {
        self.stress.push((t, s));
        if self.stress.len() > MAX_SAMPLES {
            self.stress.drain(0..MAX_SAMPLES / 2);
        }
    }

    fn push_region(&mut self, name: &str, t: f32, a: RegionAverages) {
        let v = self.regions.entry(name.to_string()).or_default();
        v.push((t, a));
        if v.len() > MAX_SAMPLES {
            v.drain(0..MAX_SAMPLES / 2);
        }
    }

    fn push_element(&mut self, t: f32, vm: f32) {
        self.element.push((t, vm));
        if self.element.len() > MAX_SAMPLES {
            self.element.drain(0..MAX_SAMPLES / 2);
        }
    }

    fn push_vertex(&mut self, t: f32, disp_mag: f32, force_mag: f32) {
        self.vertex.push((t, disp_mag, force_mag));
        if self.vertex.len() > MAX_SAMPLES {
            self.vertex.drain(0..MAX_SAMPLES / 2);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotConfig {
    /// Show the plots panel below the viewport.
    #[serde(default = "default_true")]
    pub visible: bool,
    /// How often to sample (every Nth solver step).
    #[serde(default = "default_sample_stride")]
    pub sample_stride: u32,
    /// Which curves to draw.
    #[serde(default = "default_true")]
    pub show_vm_min: bool,
    #[serde(default = "default_true")]
    pub show_vm_mean: bool,
    #[serde(default = "default_true")]
    pub show_vm_max: bool,
    /// Per-region displacement magnitude on the displacement plot.
    #[serde(default = "default_true")]
    pub show_region_displacement: bool,
    /// Per-region force magnitude on the force plot.
    #[serde(default = "default_true")]
    pub show_region_force: bool,
    /// Inspect-element / inspect-vertex plot (A5).
    #[serde(default = "default_true")]
    pub show_inspect: bool,
}

fn default_true() -> bool {
    true
}
fn default_sample_stride() -> u32 {
    10
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            visible: true,
            sample_stride: default_sample_stride(),
            show_vm_min: true,
            show_vm_mean: true,
            show_vm_max: true,
            show_region_displacement: true,
            show_region_force: true,
            show_inspect: true,
        }
    }
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
            material: MaterialProps3D::default(),
            time_delta: default_time_delta(),
            duration: default_duration(),
            steps_per_frame: default_steps_per_frame(),
            running: false,
            viewport: ViewportState::default(),
            computer: None,
            stats: StressStats::default(),
            history: History::default(),
            plots: PlotConfig::default(),
            inspect: InspectConfig::default(),
            last_export_dir: None,
            last_export_status: None,
            use_gpu_stresses: false,
            gpu_kernel: None,
            gpu_status: None,
            gpu_preferred_adapter: None,
            gpu_available_adapters: Vec::new(),
            engine_config_open: false,
            body_force: [0.0; 3],
            auto_export: false,
            sim_show_particles: true,
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
    /// Per-axis keyframes for TimeForce / TimeDisplacement BCs. Shared
    /// between the two so users can swap kinds without retyping.
    #[serde(default)]
    pub time_profile: AxisTimeSeries,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BcKind {
    #[default]
    Free,
    Pinned,
    ConstantForce,
    ConstantDisplacement,
    TimeForce,
    TimeDisplacement,
}

impl BcKind {
    pub fn label(self) -> &'static str {
        match self {
            BcKind::Free => "Free",
            BcKind::Pinned => "Pinned",
            BcKind::ConstantForce => "Constant Force",
            BcKind::ConstantDisplacement => "Constant Displacement",
            BcKind::TimeForce => "Time Force",
            BcKind::TimeDisplacement => "Time Displacement",
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
            BcKind::TimeForce => BoundaryCondition3D::TimeForce {
                profile: self.time_profile.clone(),
            },
            BcKind::TimeDisplacement => BoundaryCondition3D::TimeDisplacement {
                axes: self.axes,
                profile: self.time_profile.clone(),
            },
        }
    }
}

/// Stitch every populated mesh in the scene into a single `Mesh3D` so
/// the solver sees one combined system (B11). Returns `None` if no
/// mesh is available. Body numbering is by **slice position**, not
/// populated index — see `Mesh3D::combine` for why this matters for the
/// Boundary Conditions phase.
pub fn combine_active(meshes: &[Option<Mesh3D>]) -> Option<Mesh3D> {
    if meshes.iter().all(|m| m.is_none()) {
        return None;
    }
    let refs: Vec<Option<&Mesh3D>> = meshes.iter().map(|m| m.as_ref()).collect();
    Some(Mesh3D::combine(&refs))
}

pub fn show(
    state: &mut State,
    geometry: &super::drawing::shape::Geometry3D,
    meshes: &[Option<Mesh3D>],
    ui: &mut Ui,
) {
    let combined = combine_active(meshes);
    SidePanel::right("d3_sim_side_panel")
        .resizable(true)
        .default_width(300.0)
        .show_inside(ui, |ui| {
            ui.heading("3D Simulation");
            // "X of N shapes meshed" status — amber when unmeshed
            // shapes remain in the scene, so users notice before they
            // hit Run.
            let populated = meshes.iter().filter(|m| m.is_some()).count();
            let total = geometry.shapes.len();
            let header_color = if total == 0 {
                Color32::YELLOW
            } else if populated < total {
                Color32::from_rgb(255, 160, 100)
            } else {
                Color32::from_gray(200)
            };
            ui.colored_label(
                header_color,
                format!("Bodies: {populated} meshed / {total} drawn"),
            );
            ui.separator();
            add_mesh_summary(meshes, &combined, ui);

            let Some(mesh) = combined.as_ref() else {
                ui.colored_label(
                    Color32::YELLOW,
                    "No mesh available. Generate one on the Meshing page first.",
                );
                return;
            };

            if ui.button("⚙ Engine config…").clicked() {
                state.engine_config_open = true;
            }
            ui.collapsing("Material", |ui| add_material_controls(state, ui));
            ui.collapsing("Integration", |ui| add_integration_controls(state, ui));
            ui.label("Boundary conditions are set on the Boundary Conditions page.");
            ui.collapsing("Inspect", |ui| add_inspect_controls(state, ui));

            ui.separator();
            add_run_controls(state, mesh, ui);
            add_stats(state, ui);
        });

    if state.running {
        if let Some(c) = state.computer.as_mut() {
            // Step one at a time so we can sample at the user's stride
            // rather than just at frame boundaries (smoother curves).
            let steps = state.steps_per_frame as u64;
            let stride = state.plots.sample_stride.max(1) as u64;
            let active_mesh = combined.as_ref();
            // GPU strain/stress path (E18). The kernel is built lazily on
            // first use; failures fall back to CPU and surface a status
            // message instead of panicking.
            let gpu_active = state.use_gpu_stresses
                && gpu::GpuStressKernel::supports(&c.config.material);
            if gpu_active && state.gpu_kernel.is_none() {
                match gpu::GpuStressKernel::new(state.gpu_preferred_adapter.as_ref()) {
                    Ok(k) => {
                        state.gpu_status = Some(format!("GPU ready: {}", k.adapter.label()));
                        state.gpu_kernel = Some(k);
                    }
                    Err(e) => {
                        state.gpu_status = Some(format!("GPU init failed: {e} — using CPU"));
                        state.use_gpu_stresses = false;
                    }
                }
            }
            let mut gpu_active = gpu_active && state.gpu_kernel.is_some();
            for _ in 0..steps {
                if gpu_active {
                    let kernel = state.gpu_kernel.as_mut().unwrap();
                    match kernel.compute_stresses(c) {
                        Ok(stresses) => {
                            c.apply_external_stresses(&stresses);
                            c.assemble_forces_and_integrate();
                        }
                        Err(e) => {
                            // Persistent runtime failure: drop the kernel
                            // and turn the toggle off so we don't re-fail
                            // every frame. User can re-enable explicitly
                            // after addressing the cause.
                            state.gpu_status = Some(format!(
                                "GPU compute failed: {e} — GPU disabled, using CPU"
                            ));
                            state.gpu_kernel = None;
                            state.use_gpu_stresses = false;
                            gpu_active = false;
                            c.step();
                        }
                    }
                } else {
                    c.step();
                }
                if c.iterations % stride == 0 {
                    let t = c.time();
                    let stats = c.stress_stats();
                    state.history.push_stress(t, stats);
                    if let Some(mesh) = active_mesh {
                        for region in &mesh.boundary_faces.regions {
                            let avg = c.region_averages(&region.vertices);
                            state.history.push_region(&region.name, t, avg);
                        }
                    }
                    if state.inspect.element_index > 0 {
                        let idx = (state.inspect.element_index - 1) as usize;
                        if let Some(e) = c.elements.get(idx) {
                            state.history.push_element(t, von_mises_of(&e.stress));
                        }
                    }
                    if state.inspect.vertex_index > 0 {
                        let idx = (state.inspect.vertex_index - 1) as usize;
                        if let Some(n) = c.nodes.get(idx) {
                            let disp = (n.position - n.initial_position).norm();
                            let force = n.force.norm();
                            state.history.push_vertex(t, disp, force);
                        }
                    }
                }
                if c.time() >= state.duration {
                    break;
                }
            }
            state.stats = c.stress_stats();
            if c.time() >= state.duration {
                state.running = false;
            }
        }
        ui.ctx().request_repaint();
    }

    if state.plots.visible {
        TopBottomPanel::bottom("d3_sim_plots_panel")
            .resizable(true)
            .default_height(200.0)
            .show_inside(ui, |ui| add_plots(state, ui));
    }
    show_viewport(state, combined.as_ref(), ui);
    engine_config::show_modal(state, ui.ctx());
}

fn add_mesh_summary(meshes: &[Option<Mesh3D>], combined: &Option<Mesh3D>, ui: &mut Ui) {
    let body_count = meshes.iter().filter(|m| m.is_some()).count();
    let (verts, tets) = combined
        .as_ref()
        .map(|m| (m.vertex_count(), m.tet_count()))
        .unwrap_or((0, 0));
    ui.horizontal(|ui| {
        ui.label(format!(
            "{} bod{} combined → {} verts, {} tets",
            body_count,
            if body_count == 1 { "y" } else { "ies" },
            verts,
            tets
        ));
    });
}

fn add_material_controls(state: &mut State, ui: &mut Ui) {
    // Kind selector: swap between Isotropic and Orthotropic, preserving
    // the bulk (density / damping / failure) section across the switch.
    let current_kind = match state.material {
        MaterialProps3D::Isotropic(_) => "Isotropic",
        MaterialProps3D::Orthotropic(_) => "Orthotropic",
    };
    ComboBox::from_label("Kind")
        .selected_text(current_kind)
        .show_ui(ui, |ui| {
            let mut iso_clicked = matches!(state.material, MaterialProps3D::Isotropic(_));
            let mut ortho_clicked = matches!(state.material, MaterialProps3D::Orthotropic(_));
            if ui.selectable_label(iso_clicked, "Isotropic").clicked() {
                iso_clicked = true;
            }
            if ui.selectable_label(ortho_clicked, "Orthotropic").clicked() {
                ortho_clicked = true;
            }
            if iso_clicked && !matches!(state.material, MaterialProps3D::Isotropic(_)) {
                let bulk = state.material.bulk().clone();
                state.material = MaterialProps3D::Isotropic(IsotropicProps3D {
                    bulk,
                    ..IsotropicProps3D::default()
                });
            }
            if ortho_clicked && !matches!(state.material, MaterialProps3D::Orthotropic(_)) {
                let bulk = state.material.bulk().clone();
                state.material = MaterialProps3D::Orthotropic(OrthotropicProps3D {
                    bulk,
                    ..OrthotropicProps3D::default()
                });
            }
        });

    match &mut state.material {
        MaterialProps3D::Isotropic(p) => add_isotropic_fields(p, ui),
        MaterialProps3D::Orthotropic(p) => add_orthotropic_fields(p, ui),
    }

    ui.separator();
    ui.label("Bulk");
    let bulk = match &mut state.material {
        MaterialProps3D::Isotropic(p) => &mut p.bulk,
        MaterialProps3D::Orthotropic(p) => &mut p.bulk,
    };
    ui.horizontal(|ui| {
        ui.label("Density (ρ)");
        ui.add(DragValue::new(&mut bulk.density).speed(10.0).clamp_range(1e-6..=f32::MAX));
    });
    ui.horizontal(|ui| {
        ui.label("Damping (c)");
        ui.add(DragValue::new(&mut bulk.damping).speed(0.1).clamp_range(0.0..=f32::MAX));
    });

    ui.collapsing("Failure criteria", |ui| {
        add_failure_criteria_controls(&mut bulk.failure_criteria, ui);
    });
}

fn add_isotropic_fields(p: &mut IsotropicProps3D, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.label("Young's modulus (E)");
        ui.add(DragValue::new(&mut p.elasticity_modulus).speed(1000.0).clamp_range(1.0..=f32::MAX));
    });
    ui.horizontal(|ui| {
        ui.label("Poisson's ratio (ν)");
        ui.add(DragValue::new(&mut p.poissons_ratio).speed(0.01).clamp_range(0.0..=0.49));
    });
}

fn add_orthotropic_fields(p: &mut OrthotropicProps3D, ui: &mut Ui) {
    ui.label("Young's moduli (E_x, E_y, E_z)");
    ui.horizontal(|ui| {
        ui.add(DragValue::new(&mut p.elasticity_modulus_x).speed(1000.0).clamp_range(1.0..=f32::MAX));
        ui.add(DragValue::new(&mut p.elasticity_modulus_y).speed(1000.0).clamp_range(1.0..=f32::MAX));
        ui.add(DragValue::new(&mut p.elasticity_modulus_z).speed(1000.0).clamp_range(1.0..=f32::MAX));
    });
    ui.label("Poisson's ratios (ν_xy, ν_xz, ν_yz)");
    ui.horizontal(|ui| {
        ui.add(DragValue::new(&mut p.poissons_ratio_xy).speed(0.01).clamp_range(0.0..=0.49));
        ui.add(DragValue::new(&mut p.poissons_ratio_xz).speed(0.01).clamp_range(0.0..=0.49));
        ui.add(DragValue::new(&mut p.poissons_ratio_yz).speed(0.01).clamp_range(0.0..=0.49));
    });
    ui.label("Shear moduli (G_xy, G_xz, G_yz)");
    ui.horizontal(|ui| {
        ui.add(DragValue::new(&mut p.shear_modulus_xy).speed(1000.0).clamp_range(1.0..=f32::MAX));
        ui.add(DragValue::new(&mut p.shear_modulus_xz).speed(1000.0).clamp_range(1.0..=f32::MAX));
        ui.add(DragValue::new(&mut p.shear_modulus_yz).speed(1000.0).clamp_range(1.0..=f32::MAX));
    });
}

fn add_failure_criteria_controls(c: &mut FailureCriteria3D, ui: &mut Ui) {
    optional_field(ui, "Strain energy density (W)", &mut c.strain_energy, 1.0, 0.0);
    optional_field(ui, "Tensile principal stress", &mut c.tensional_stress, 1000.0, 0.0);
    optional_field(ui, "Compressive principal stress", &mut c.compressional_stress, 1000.0, 0.0);
}

fn optional_field(ui: &mut Ui, label: &str, slot: &mut Option<f32>, speed: f32, min: f32) {
    let mut enabled = slot.is_some();
    ui.horizontal(|ui| {
        ui.checkbox(&mut enabled, label);
        if enabled {
            let mut v = slot.unwrap_or(min.max(1.0));
            ui.add(DragValue::new(&mut v).speed(speed).clamp_range(min..=f32::MAX));
            *slot = Some(v);
        } else {
            *slot = None;
        }
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
    let supports_gpu = gpu::GpuStressKernel::supports(&state.material);
    ui.horizontal(|ui| {
        ui.add_enabled(
            supports_gpu,
            egui::Checkbox::new(&mut state.use_gpu_stresses, "GPU strain/stress (E18)"),
        );
        if !supports_gpu {
            state.use_gpu_stresses = false;
            let reason = if !matches!(state.material, MaterialProps3D::Isotropic(_)) {
                "(isotropic only for now)"
            } else {
                "(disabled while failure criteria are configured)"
            };
            ui.label(reason);
        }
    });
    if state.use_gpu_stresses {
        add_gpu_controls(state, ui);
    }
}

fn add_gpu_controls(state: &mut State, ui: &mut Ui) {
    ui.indent("gpu_controls_indent", |ui| {
        // Picked GPU label.
        if let Some(k) = state.gpu_kernel.as_ref() {
            ui.label(format!("Active GPU: {}", k.adapter.label()));
        } else if let Some(msg) = state.gpu_status.as_deref() {
            ui.label(msg);
        } else {
            ui.label("GPU will be initialised on first run.");
        }

        // Lazy populate the adapter list the first time the panel is
        // shown; cheap enough to re-poll on demand via a button if the
        // user docks / undocks an eGPU.
        if state.gpu_available_adapters.is_empty() {
            state.gpu_available_adapters = gpu::list_adapters();
        }
        ui.horizontal(|ui| {
            ui.label("Override:");
            let mut selection = state.gpu_preferred_adapter.clone();
            let current_label = selection
                .as_ref()
                .map(|a| a.label())
                .unwrap_or_else(|| "Auto (HighPerformance)".to_string());
            egui::ComboBox::from_id_source("gpu_override")
                .selected_text(current_label)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut selection, None, "Auto (HighPerformance)");
                    for a in &state.gpu_available_adapters {
                        ui.selectable_value(&mut selection, Some(a.clone()), a.label());
                    }
                });
            if selection != state.gpu_preferred_adapter {
                state.gpu_preferred_adapter = selection;
                // Drop the kernel so the new preference is honored on
                // the next step. Keep the toggle on.
                state.gpu_kernel = None;
                state.gpu_status = Some("override changed — rebuilding on next step".to_string());
            }
            if ui.small_button("rescan").on_hover_text("Re-enumerate adapters").clicked() {
                state.gpu_available_adapters = gpu::list_adapters();
            }
        });

        if let Some(k) = state.gpu_kernel.as_ref() {
            ui.collapsing("Init log", |ui| {
                for line in &k.init_log {
                    ui.monospace(line);
                }
            });
        }
    });
}

/// Mesh-driven BC editor. The new pipeline order (Drawing → BC →
/// Meshing → Simulation) means the BC phase now reads geometry shapes
/// directly via `super::boundary_conditions::add_geometry_bc_controls`,
/// so this function is currently unused. Kept around in case a debug
/// "edit BCs against the meshed body" view returns later.
#[allow(dead_code)]
pub fn add_bc_controls(state: &mut State, mesh: &Mesh3D, ui: &mut Ui) {
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
                        BcKind::TimeForce,
                        BcKind::TimeDisplacement,
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
                BcKind::TimeForce => {
                    time_profile_editor(&region.name, &mut entry.time_profile, ui);
                }
                BcKind::TimeDisplacement => {
                    axes_row(ui, &mut entry.axes);
                    time_profile_editor(&region.name, &mut entry.time_profile, ui);
                }
            }
        });
    }
}

#[allow(dead_code)]
fn time_profile_editor(region_name: &str, profile: &mut AxisTimeSeries, ui: &mut Ui) {
    ui.label("Keyframes (time, value) per axis");
    for (axis_label, series) in [
        ("X", &mut profile.x),
        ("Y", &mut profile.y),
        ("Z", &mut profile.z),
    ] {
        ui.collapsing(
            format!("{axis_label} — {} pts", series.points.len()),
            |ui| {
                let id_suffix = format!("{region_name}_{axis_label}");
                let mut remove: Option<usize> = None;
                for (idx, (t, v)) in series.points.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("{idx}:"));
                        ui.label("t");
                        ui.add(DragValue::new(t).speed(0.001).clamp_range(0.0..=f32::MAX));
                        ui.label("v");
                        ui.add(DragValue::new(v).speed(0.01));
                        if ui.small_button("x").clicked() {
                            remove = Some(idx);
                        }
                    });
                    let _ = &id_suffix;
                }
                if let Some(i) = remove {
                    series.points.remove(i);
                }
                if ui.button("+ keyframe").clicked() {
                    let next_t = series.points.last().map(|(t, _)| *t + 0.001).unwrap_or(0.0);
                    let next_v = series.points.last().map(|(_, v)| *v).unwrap_or(0.0);
                    series.push_keyframe(next_t, next_v);
                }
            },
        );
    }
}

fn add_inspect_controls(state: &mut State, ui: &mut Ui) {
    ui.label("Per-element / per-vertex history (A5).");
    ui.label("Enter a 1-based index, or 0 to disable. Reset clears history.");
    ui.horizontal(|ui| {
        ui.label("Element");
        ui.add(
            DragValue::new(&mut state.inspect.element_index)
                .speed(1.0)
                .clamp_range(0..=u32::MAX),
        );
    });
    if let Some(c) = state.computer.as_ref() {
        ui.label(format!("(of {} elements)", c.elements.len()));
    }
    ui.horizontal(|ui| {
        ui.label("Vertex");
        ui.add(
            DragValue::new(&mut state.inspect.vertex_index)
                .speed(1.0)
                .clamp_range(0..=u32::MAX),
        );
    });
    if let Some(c) = state.computer.as_ref() {
        ui.label(format!("(of {} nodes)", c.nodes.len()));
    }
}

#[allow(dead_code)]
fn axes_row(ui: &mut Ui, axes: &mut Axis) {
    ui.horizontal(|ui| {
        ui.label("Axes:");
        ui.checkbox(&mut axes.x, "X");
        ui.checkbox(&mut axes.y, "Y");
        ui.checkbox(&mut axes.z, "Z");
    });
}

#[allow(dead_code)]
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
                state.history.clear();
            }
        }
        if ui.button("Rebuild").clicked() {
            state.running = false;
            state.history.clear();
            rebuild_computer(state, mesh);
        }
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut state.plots.visible, "Show plots");
        ui.label("Sample every");
        ui.add(
            DragValue::new(&mut state.plots.sample_stride)
                .speed(1.0)
                .clamp_range(1..=1000u32),
        );
        ui.label("steps");
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut state.sim_show_particles, "Show particles");
    });
    ui.horizontal(|ui| {
        if ui.button("Export CSV…").clicked() {
            let start = state
                .last_export_dir
                .clone()
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
            if let Some(dir) = rfd::FileDialog::new().set_directory(start).pick_folder() {
                let report = export::export(state, Some(mesh), &dir);
                if let Some(err) = report.error {
                    state.last_export_status = Some(format!("Export failed: {err}"));
                } else {
                    state.last_export_status =
                        Some(format!("Wrote {} files to {}", report.files.len(), dir.display()));
                    state.last_export_dir = Some(dir);
                }
            }
        }
        if let Some(msg) = state.last_export_status.as_deref() {
            ui.label(msg);
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
        body_force: state.body_force,
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

fn add_plots(state: &mut State, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.label("Plot curves:");
        ui.checkbox(&mut state.plots.show_vm_min, "VM min");
        ui.checkbox(&mut state.plots.show_vm_mean, "VM mean");
        ui.checkbox(&mut state.plots.show_vm_max, "VM max");
        ui.separator();
        ui.checkbox(&mut state.plots.show_region_displacement, "Region |u|");
        ui.checkbox(&mut state.plots.show_region_force, "Region |F|");
        ui.separator();
        ui.checkbox(&mut state.plots.show_inspect, "Inspect");
    });
    let cols = if state.plots.show_inspect { 4.0 } else { 3.0 };
    let total = ui.available_width().max(300.0);
    let panel_w = (total / cols).max(150.0);
    let panel_h = ui.available_height().max(120.0);

    ui.horizontal(|ui| {
        ui.allocate_ui(Vec2::new(panel_w, panel_h), |ui| {
            ui.label("Von Mises stress (Pa)");
            Plot::new("d3_vm_plot")
                .height(panel_h - 20.0)
                .show(ui, |plot_ui| {
                    if state.plots.show_vm_min {
                        plot_ui.line(
                            Line::new(PlotPoints::from_iter(
                                state.history.stress.iter().map(|(t, s)| [*t as f64, s.min_von_mises as f64]),
                            ))
                            .color(Color32::from_rgb(80, 140, 240))
                            .name("min"),
                        );
                    }
                    if state.plots.show_vm_mean {
                        plot_ui.line(
                            Line::new(PlotPoints::from_iter(
                                state.history.stress.iter().map(|(t, s)| [*t as f64, s.mean_von_mises as f64]),
                            ))
                            .color(Color32::from_rgb(220, 220, 80))
                            .name("mean"),
                        );
                    }
                    if state.plots.show_vm_max {
                        plot_ui.line(
                            Line::new(PlotPoints::from_iter(
                                state.history.stress.iter().map(|(t, s)| [*t as f64, s.max_von_mises as f64]),
                            ))
                            .color(Color32::from_rgb(240, 100, 100))
                            .name("max"),
                        );
                    }
                });
        });
        ui.allocate_ui(Vec2::new(panel_w, panel_h), |ui| {
            ui.label("Region displacement magnitude (m)");
            Plot::new("d3_disp_plot")
                .height(panel_h - 20.0)
                .show(ui, |plot_ui| {
                    if !state.plots.show_region_displacement {
                        return;
                    }
                    for (i, (name, series)) in sorted_regions(&state.history.regions).enumerate() {
                        plot_ui.line(
                            Line::new(PlotPoints::from_iter(
                                series.iter().map(|(t, a)| {
                                    [*t as f64, a.mean_displacement.norm() as f64]
                                }),
                            ))
                            .color(region_color(i))
                            .name(name),
                        );
                    }
                });
        });
        ui.allocate_ui(Vec2::new(panel_w, panel_h), |ui| {
            ui.label("Region force magnitude (N)");
            Plot::new("d3_force_plot")
                .height(panel_h - 20.0)
                .show(ui, |plot_ui| {
                    if !state.plots.show_region_force {
                        return;
                    }
                    for (i, (name, series)) in sorted_regions(&state.history.regions).enumerate() {
                        plot_ui.line(
                            Line::new(PlotPoints::from_iter(
                                series.iter().map(|(t, a)| {
                                    [*t as f64, a.mean_force.norm() as f64]
                                }),
                            ))
                            .color(region_color(i))
                            .name(name),
                        );
                    }
                });
        });
        if state.plots.show_inspect {
            ui.allocate_ui(Vec2::new(panel_w, panel_h), |ui| {
                ui.label(format!(
                    "Inspect (elem #{}, vert #{})",
                    state.inspect.element_index, state.inspect.vertex_index
                ));
                Plot::new("d3_inspect_plot")
                    .height(panel_h - 20.0)
                    .show(ui, |plot_ui| {
                        if !state.history.element.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::from_iter(
                                    state
                                        .history
                                        .element
                                        .iter()
                                        .map(|(t, v)| [*t as f64, *v as f64]),
                                ))
                                .color(Color32::from_rgb(240, 100, 100))
                                .name("element VM"),
                            );
                        }
                        if !state.history.vertex.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::from_iter(
                                    state
                                        .history
                                        .vertex
                                        .iter()
                                        .map(|(t, d, _)| [*t as f64, *d as f64]),
                                ))
                                .color(Color32::from_rgb(120, 220, 120))
                                .name("vertex |u|"),
                            );
                            plot_ui.line(
                                Line::new(PlotPoints::from_iter(
                                    state
                                        .history
                                        .vertex
                                        .iter()
                                        .map(|(t, _, f)| [*t as f64, *f as f64]),
                                ))
                                .color(Color32::from_rgb(100, 160, 240))
                                .name("vertex |F|"),
                            );
                        }
                    });
            });
        }
    });
}

fn sorted_regions(
    regions: &std::collections::HashMap<String, Vec<(f32, RegionAverages)>>,
) -> impl Iterator<Item = (&String, &Vec<(f32, RegionAverages)>)> {
    let mut names: Vec<&String> = regions.keys().collect();
    names.sort();
    names.into_iter().map(move |n| (n, &regions[n]))
}

fn region_color(i: usize) -> Color32 {
    const PALETTE: [Color32; 8] = [
        Color32::from_rgb(120, 200, 255),
        Color32::from_rgb(255, 140, 140),
        Color32::from_rgb(150, 230, 150),
        Color32::from_rgb(255, 200, 100),
        Color32::from_rgb(200, 150, 255),
        Color32::from_rgb(100, 220, 220),
        Color32::from_rgb(255, 180, 220),
        Color32::from_rgb(220, 220, 100),
    ];
    PALETTE[i % PALETTE.len()]
}

pub fn show_viewport(state: &mut State, active_mesh: Option<&Mesh3D>, ui: &mut Ui) {
    // Stale-solver guard: if the mesh has been regenerated since the
    // Computer3D was built (different vertex / element counts), the
    // node positions no longer match the mesh's boundary face indices
    // and rendering would index out of bounds. Drop the stale solver
    // so the next Run rebuilds it against the current mesh.
    if let (Some(c), Some(mesh)) = (state.computer.as_ref(), active_mesh) {
        if c.nodes.len() != mesh.vertices.len() || c.elements.len() != mesh.tetrahedra.len() {
            state.running = false;
            state.computer = None;
            state.stats = StressStats::default();
            state.history.clear();
            state.gpu_status = Some(
                "Mesh changed — solver discarded. Click Run to rebuild.".to_string(),
            );
        }
    }

    let available = ui.available_size();
    let size = Vec2::new(available.x.max(100.0), available.y.max(100.0));
    let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
    let rect = response.rect;

    // Auto-frame on first render.
    let aabb = active_mesh.map(|m| {
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

    // If we have a running computer, draw the deformed mesh surface
    // colored by Von Mises stress averaged over each vertex's incident
    // tets. Otherwise show the reference mesh's surface in solid grey.
    if let (Some(c), Some(mesh)) = (state.computer.as_ref(), active_mesh) {
        let s_min = state.stats.min_von_mises;
        let s_max = state.stats.max_von_mises.max(s_min + 1e-12);

        // Per-vertex average Von Mises plus a broken-tet flag. Vertices
        // touching any broken tet render as cracked red so users can
        // see fracture forming on the surface.
        let mut vm_sum = vec![0.0_f32; c.nodes.len()];
        let mut vm_count = vec![0_u32; c.nodes.len()];
        let mut touches_broken = vec![false; c.nodes.len()];
        for e in &c.elements {
            let vm = von_mises_of(&e.stress);
            for &i in &e.indices {
                vm_sum[i] += vm;
                vm_count[i] += 1;
                if e.is_broken {
                    touches_broken[i] = true;
                }
            }
        }
        let positions_f32: Vec<Vector3<f32>> =
            c.nodes.iter().map(|n| n.position).collect();
        let color_for = |i: usize| -> [f32; 4] {
            if touches_broken[i] {
                // Cracked: deep red, used as the "broken" visual marker.
                return [0.85, 0.10, 0.10, 1.0];
            }
            let avg = if vm_count[i] > 0 {
                vm_sum[i] / vm_count[i] as f32
            } else {
                0.0
            };
            let t = ((avg - s_min) / (s_max - s_min)).clamp(0.0, 1.0);
            color32_to_f32_rgba(heatmap(t))
        };
        let mut tris = scene_mesh::triangles_for_deformed(mesh, &positions_f32, color_for);
        if state.sim_show_particles {
            // Per-vertex tint: cracked-red on broken-adjacent nodes,
            // bright yellow elsewhere. Reuses the touches_broken array
            // already built for the surface color.
            let particle_color_for = |i: usize| -> [f32; 4] {
                if touches_broken.get(i).copied().unwrap_or(false) {
                    [0.85, 0.10, 0.10, 1.0]
                } else {
                    [1.0, 1.0, 0.15, 1.0]
                }
            };
            let (cam_right, cam_up, _) = state.viewport.camera.basis_world();
            let particle_size = state.viewport.camera.distance() * 0.004;
            tris.extend(scene_mesh::particles_for_mesh_colored(
                mesh,
                &positions_f32,
                cam_right,
                cam_up,
                particle_size,
                particle_color_for,
            ));
        }
        if !tris.is_empty() {
            wgpu_scene::sort_back_to_front(&mut tris, &view_proj);
            let cb = wgpu_scene::SceneCallback::from_world_mvp(tris, &view_proj);
            painter.add(eframe::egui_wgpu::Callback::new_paint_callback(rect, cb));
        }
    } else if let Some(mesh) = active_mesh {
        let grey = [0.65, 0.65, 0.68, 1.0];
        let mut tris = scene_mesh::triangles_for_mesh(mesh, |_| grey);
        if !tris.is_empty() {
            wgpu_scene::sort_back_to_front(&mut tris, &view_proj);
            let cb = wgpu_scene::SceneCallback::from_world_mvp(tris, &view_proj);
            painter.add(eframe::egui_wgpu::Callback::new_paint_callback(rect, cb));
        }
    }

    // Boundary-condition overlay arrows / pinned markers, drawn on top
    // of the wgpu scene as a 2D HUD. Uses solver positions when a Computer
    // exists, otherwise the mesh's reference vertices.
    if let Some(mesh) = active_mesh {
        let positions: Option<Vec<Vector3<f64>>> = state.computer.as_ref().map(|c| {
            c.nodes
                .iter()
                .map(|n| Vector3::new(n.position.x as f64, n.position.y as f64, n.position.z as f64))
                .collect()
        });
        paint_bc_overlays(
            &painter,
            rect,
            &view_proj,
            &state.region_bcs,
            mesh,
            positions.as_deref(),
        );
    }

    // Color-bar legend, only meaningful when the heatmap is showing.
    if state.computer.is_some() {
        paint_color_bar(
            &painter,
            rect,
            state.stats.min_von_mises,
            state.stats.max_von_mises,
        );
    }

    paint_zoom_controls(&mut state.viewport, rect, ui);
}

/// On-screen zoom in/out + reset buttons in the bottom-left corner of
/// the viewport. Useful on trackpads and on large displays where users
/// want a precise step. Also bind +/- on the keyboard while the
/// viewport area is hovered. Reset triggers auto-framing on the next
/// paint.
fn paint_zoom_controls(viewport: &mut ViewportState, rect: egui::Rect, ui: &mut Ui) {
    let pos = rect.left_bottom() + egui::vec2(8.0, -8.0 - 28.0);
    let btn_size = egui::vec2(28.0, 28.0);
    let btn_rect = |i: usize| {
        egui::Rect::from_min_size(
            egui::pos2(pos.x + i as f32 * (btn_size.x + 4.0), pos.y),
            btn_size,
        )
    };

    let mut zoom_in = ui
        .put(btn_rect(0), egui::Button::new("+").min_size(btn_size))
        .on_hover_text("Zoom in")
        .clicked();
    let mut zoom_out = ui
        .put(btn_rect(1), egui::Button::new("−").min_size(btn_size))
        .on_hover_text("Zoom out")
        .clicked();
    let reset_view = ui
        .put(btn_rect(2), egui::Button::new("⟳").min_size(btn_size))
        .on_hover_text("Reset view (auto-frame on next paint)")
        .clicked();

    // Keyboard +/- when the pointer is over the viewport area.
    if ui.rect_contains_pointer(rect) {
        let (kin, kout) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals),
                i.key_pressed(egui::Key::Minus),
            )
        });
        zoom_in |= kin;
        zoom_out |= kout;
    }

    // Camera::zoom takes scroll-delta semantics: positive = zoom in.
    // 50 ≈ a chunky discrete step that feels right for button clicks.
    if zoom_in {
        viewport.camera.zoom(50.0);
    }
    if zoom_out {
        viewport.camera.zoom(-50.0);
    }
    if reset_view {
        viewport.auto_frame = true;
    }
}

fn paint_bc_overlays(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &nalgebra::Matrix4<f64>,
    region_bcs: &HashMap<String, RegionBc>,
    mesh: &Mesh3D,
    deformed_positions: Option<&[Vector3<f64>]>,
) {
    // Scene-size scale: use the AABB diagonal of the active vertex set so
    // arrow lengths stay proportional to the mesh, not to physical force
    // magnitudes (which vary wildly).
    // Same stale-solver guard as in show_viewport: if the deformed
    // positions slice doesn't cover every mesh vertex, fall back to
    // the reference vertex instead of panicking on the indexing.
    let vertex_pos = |i: usize| -> Vector3<f64> {
        match deformed_positions {
            Some(p) => *p.get(i).unwrap_or(&mesh.vertices[i]),
            None => mesh.vertices[i],
        }
    };
    let mut lo = Vector3::repeat(f64::INFINITY);
    let mut hi = Vector3::repeat(f64::NEG_INFINITY);
    for i in 0..mesh.vertices.len() {
        let v = vertex_pos(i);
        lo = lo.inf(&v);
        hi = hi.sup(&v);
    }
    let scene_scale = (hi - lo).norm().max(1e-6);
    let arrow_len_world = scene_scale * 0.18;

    for region in &mesh.boundary_faces.regions {
        let Some(bc) = region_bcs.get(&region.name) else {
            continue;
        };
        if matches!(bc.kind, BcKind::Free) || region.vertices.is_empty() {
            continue;
        }

        // Region centroid.
        let mut centroid = Vector3::<f64>::zeros();
        for &i in &region.vertices {
            centroid += vertex_pos(i);
        }
        centroid /= region.vertices.len() as f64;

        // Outward normal: sum of face normals (without normalizing each
        // first) lets larger faces dominate, then normalize.
        let mut normal = Vector3::<f64>::zeros();
        for face in &region.faces {
            let a = vertex_pos(face[0]);
            let b = vertex_pos(face[1]);
            let c = vertex_pos(face[2]);
            normal += (b - a).cross(&(c - a));
        }
        if normal.norm() > 1e-12 {
            normal = normal.normalize();
        } else {
            normal = Vector3::new(0.0, 1.0, 0.0);
        }

        // Anchor the arrow base just off the surface so it's not hidden
        // by the mesh; tail offset = 5% of scene scale outward.
        let base = centroid + normal * scene_scale * 0.02;

        match bc.kind {
            BcKind::Free => {}
            BcKind::Pinned => {
                draw_pinned_marker(
                    painter,
                    rect,
                    view_proj,
                    base,
                    normal,
                    bc.axes,
                    scene_scale * 0.04,
                );
                draw_label(
                    painter,
                    rect,
                    view_proj,
                    base + normal * scene_scale * 0.05,
                    &format!("{}: pinned {}", region.name, axes_label(bc.axes)),
                    Color32::from_rgb(160, 200, 240),
                );
            }
            BcKind::ConstantForce => {
                let f = Vector3::new(bc.force[0] as f64, bc.force[1] as f64, bc.force[2] as f64);
                let dir = if f.norm() > 1e-12 { f.normalize() } else { normal };
                let tip = base + dir * arrow_len_world;
                draw_arrow(
                    painter,
                    rect,
                    view_proj,
                    base,
                    tip,
                    Color32::from_rgb(255, 200, 100),
                    2.0,
                );
                draw_label(
                    painter,
                    rect,
                    view_proj,
                    tip,
                    &format!("{}: F |{:.2e}| N", region.name, f.norm()),
                    Color32::from_rgb(255, 200, 100),
                );
            }
            BcKind::ConstantDisplacement => {
                let d = Vector3::new(
                    bc.displacement[0] as f64,
                    bc.displacement[1] as f64,
                    bc.displacement[2] as f64,
                );
                let dir = if d.norm() > 1e-12 { d.normalize() } else { normal };
                let tip = base + dir * arrow_len_world;
                draw_arrow_dashed(
                    painter,
                    rect,
                    view_proj,
                    base,
                    tip,
                    Color32::from_rgb(160, 240, 160),
                    2.0,
                );
                draw_label(
                    painter,
                    rect,
                    view_proj,
                    tip,
                    &format!(
                        "{}: u {} |{:.2e}|",
                        region.name,
                        axes_label(bc.axes),
                        d.norm()
                    ),
                    Color32::from_rgb(160, 240, 160),
                );
            }
            BcKind::TimeForce => {
                // Indicate a time-varying force using a chevron-style label
                // anchored at the region centroid. Its direction varies
                // with time so a static arrow would be misleading.
                draw_label(
                    painter,
                    rect,
                    view_proj,
                    base + normal * scene_scale * 0.05,
                    &format!(
                        "{}: F(t) ({} keyframes)",
                        region.name,
                        bc.time_profile.x.points.len()
                            + bc.time_profile.y.points.len()
                            + bc.time_profile.z.points.len()
                    ),
                    Color32::from_rgb(255, 200, 100),
                );
            }
            BcKind::TimeDisplacement => {
                draw_label(
                    painter,
                    rect,
                    view_proj,
                    base + normal * scene_scale * 0.05,
                    &format!(
                        "{}: u(t) {} ({} keyframes)",
                        region.name,
                        axes_label(bc.axes),
                        bc.time_profile.x.points.len()
                            + bc.time_profile.y.points.len()
                            + bc.time_profile.z.points.len()
                    ),
                    Color32::from_rgb(160, 240, 160),
                );
            }
        }
    }
}

fn axes_label(axes: Axis) -> String {
    let mut s = String::new();
    if axes.x {
        s.push('X');
    }
    if axes.y {
        s.push('Y');
    }
    if axes.z {
        s.push('Z');
    }
    if s.is_empty() {
        "—".to_string()
    } else {
        s
    }
}

fn draw_arrow(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &nalgebra::Matrix4<f64>,
    base: Vector3<f64>,
    tip: Vector3<f64>,
    color: Color32,
    width: f32,
) {
    let (Some(p0), Some(p1)) = (project(view_proj, rect, base), project(view_proj, rect, tip)) else {
        return;
    };
    let stroke = Stroke::new(width, color);
    painter.line_segment([p0, p1], stroke);
    draw_arrowhead(painter, p0, p1, color, width);
}

fn draw_arrow_dashed(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &nalgebra::Matrix4<f64>,
    base: Vector3<f64>,
    tip: Vector3<f64>,
    color: Color32,
    width: f32,
) {
    let (Some(p0), Some(p1)) = (project(view_proj, rect, base), project(view_proj, rect, tip)) else {
        return;
    };
    let dash_count = 6;
    let stroke = Stroke::new(width, color);
    for i in 0..dash_count {
        if i % 2 == 1 {
            continue;
        }
        let t0 = i as f32 / dash_count as f32;
        let t1 = (i + 1) as f32 / dash_count as f32;
        let a = egui::pos2(p0.x + (p1.x - p0.x) * t0, p0.y + (p1.y - p0.y) * t0);
        let b = egui::pos2(p0.x + (p1.x - p0.x) * t1, p0.y + (p1.y - p0.y) * t1);
        painter.line_segment([a, b], stroke);
    }
    draw_arrowhead(painter, p0, p1, color, width);
}

fn draw_arrowhead(
    painter: &egui::Painter,
    base: egui::Pos2,
    tip: egui::Pos2,
    color: Color32,
    width: f32,
) {
    let dx = tip.x - base.x;
    let dy = tip.y - base.y;
    let len = (dx * dx + dy * dy).sqrt().max(1e-3);
    let ux = dx / len;
    let uy = dy / len;
    let head = 9.0_f32.min(len * 0.4);
    // Perpendicular for the wings.
    let px = -uy;
    let py = ux;
    let a = egui::pos2(
        tip.x - ux * head + px * head * 0.5,
        tip.y - uy * head + py * head * 0.5,
    );
    let b = egui::pos2(
        tip.x - ux * head - px * head * 0.5,
        tip.y - uy * head - py * head * 0.5,
    );
    let stroke = Stroke::new(width, color);
    painter.line_segment([tip, a], stroke);
    painter.line_segment([tip, b], stroke);
}

fn draw_pinned_marker(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &nalgebra::Matrix4<f64>,
    center: Vector3<f64>,
    normal: Vector3<f64>,
    axes: Axis,
    world_size: f64,
) {
    // Render as three short bars pointing along each axis the constraint
    // covers. A bar means "this axis is locked"; missing bar means free.
    let bars = [
        (axes.x, Vector3::new(1.0, 0.0, 0.0), Color32::from_rgb(240, 110, 110)),
        (axes.y, Vector3::new(0.0, 1.0, 0.0), Color32::from_rgb(110, 220, 110)),
        (axes.z, Vector3::new(0.0, 0.0, 1.0), Color32::from_rgb(110, 160, 250)),
    ];
    for (on, dir, color) in bars {
        if !on {
            continue;
        }
        let a = center - dir * world_size * 0.5;
        let b = center + dir * world_size * 0.5;
        if let (Some(pa), Some(pb)) = (project(view_proj, rect, a), project(view_proj, rect, b)) {
            painter.line_segment([pa, pb], Stroke::new(2.5, color));
        }
    }
    // Also draw a small ring oriented along the surface normal to indicate
    // the anchor point.
    let _ = normal;
    if let Some(p) = project(view_proj, rect, center) {
        painter.circle_stroke(
            p,
            4.0,
            Stroke::new(1.2, Color32::from_rgb(200, 220, 240)),
        );
    }
}

fn draw_label(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &nalgebra::Matrix4<f64>,
    anchor: Vector3<f64>,
    text: &str,
    color: Color32,
) {
    if let Some(p) = project(view_proj, rect, anchor) {
        painter.text(
            p + egui::vec2(6.0, -6.0),
            egui::Align2::LEFT_BOTTOM,
            text,
            egui::FontId::monospace(10.0),
            color,
        );
    }
}

fn paint_color_bar(painter: &egui::Painter, rect: egui::Rect, vmin: f32, vmax: f32) {
    let bar_w = 14.0;
    let margin = 12.0;
    let bar_h = (rect.height() - margin * 4.0).max(60.0);
    let x = rect.right() - margin - bar_w;
    let y_top = rect.top() + margin * 2.0;
    let bar_rect = egui::Rect::from_min_size(
        egui::pos2(x, y_top),
        egui::vec2(bar_w, bar_h),
    );

    // Paint a stack of thin colored rectangles. 64 stops is smooth enough
    // at this size without flooding the egui shape buffer.
    const STOPS: usize = 64;
    let stop_h = bar_h / STOPS as f32;
    for i in 0..STOPS {
        // Top of the bar = max stress (t=1), bottom = min (t=0).
        let t = 1.0 - (i as f32 + 0.5) / STOPS as f32;
        let color = heatmap(t);
        let r = egui::Rect::from_min_size(
            egui::pos2(x, y_top + i as f32 * stop_h),
            egui::vec2(bar_w, stop_h + 0.5),
        );
        painter.rect_filled(r, 0.0, color);
    }
    painter.rect_stroke(bar_rect, 0.0, Stroke::new(1.0, Color32::from_gray(120)));

    let font = egui::FontId::monospace(10.0);
    let label_color = Color32::from_gray(220);
    let label = |val: f32, y: f32| {
        painter.text(
            egui::pos2(x - 6.0, y),
            egui::Align2::RIGHT_CENTER,
            format!("{:.2e}", val),
            font.clone(),
            label_color,
        );
    };
    label(vmax, y_top);
    label((vmin + vmax) * 0.5, y_top + bar_h * 0.5);
    label(vmin, y_top + bar_h);
    painter.text(
        egui::pos2(x + bar_w * 0.5, y_top - margin),
        egui::Align2::CENTER_BOTTOM,
        "Von Mises (Pa)",
        font,
        label_color,
    );
}

fn color32_to_f32_rgba(c: Color32) -> [f32; 4] {
    let [r, g, b, a] = c.to_array();
    [
        r as f32 / 255.0,
        g as f32 / 255.0,
        b as f32 / 255.0,
        a as f32 / 255.0,
    ]
}

#[allow(dead_code)]
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
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        if scroll.abs() > 0.0 {
            camera.zoom(scroll);
        }
        if (pinch - 1.0).abs() > 1e-4 {
            camera.zoom_by(pinch);
        }
    }
}
