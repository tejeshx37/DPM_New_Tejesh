//! GPU compute foundation for 3D strain/stress evaluation (E18).
//!
//! Owns a self-contained wgpu Instance + Device + Queue (separate from
//! the eframe rendering device for isolation; per-step buffer reuse
//! amortises the extra device cost). A single compute pipeline runs the
//! deformation-gradient → Green-Lagrange strain → isotropic stress
//! pipeline per element in parallel on the GPU and reads the resulting
//! stresses back into the simulator.
//!
//! Scope this iteration: isotropic material only, no failure criteria
//! (those rely on per-element `strain_energy` updates that the CPU path
//! tracks). Orthotropic + failure on GPU are future work; the UI
//! disables the toggle while non-isotropic materials are selected.
//!
//! Integration with `Computer3D`: the simulator's run loop calls
//! `kernel.compute_stresses(c)`, then `c.apply_external_stresses(...)`,
//! then `c.assemble_forces_and_integrate()` in place of `c.step()`.
//! Force assembly and integration stay on the CPU; this commit's perf
//! contribution is the strain/stress kernel alone.

use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use cpd::d3::{Computer3D, MaterialProps3D};
use nalgebra::Matrix3;
use wgpu::util::DeviceExt;

/// Default workgroup size literally baked into the [`SHADER_SRC`] string
/// (`@workgroup_size(64)`); the kernel rewrites this at pipeline build
/// to the backend-specific value from [`pick_workgroup_size`]. The
/// constant exists only so future tooling that lints the WGSL source
/// in isolation sees a valid number.
#[allow(dead_code)]
const DEFAULT_WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct ElementInputs {
    indices: [u32; 4],
    /// `mat3` in WGSL pads each column to vec4; we pre-flatten in Rust.
    ref_inv_c0: [f32; 4],
    ref_inv_c1: [f32; 4],
    ref_inv_c2: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct StressOut {
    c0: [f32; 4],
    c1: [f32; 4],
    c2: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct MaterialUniform {
    lambda: f32,
    mu: f32,
    _pad: [f32; 2],
}

/// Workgroup size baked into the shader at pipeline-build time. Picked
/// per backend in [`pick_workgroup_size`]: Metal threadgroups perform
/// best around 32 threads, while Vulkan/DX12 favour 64.
fn pick_workgroup_size(backend: wgpu::Backend) -> u32 {
    match backend {
        wgpu::Backend::Metal => 32,
        _ => 64,
    }
}

/// Lightweight, owned snapshot of `wgpu::AdapterInfo` for UI display and
/// for matching the user's manual override selection against enumerated
/// adapters. Cloneable / Eq so the egui ComboBox can use it as state.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct AdapterDisplay {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub vendor: u32,
    pub device: u32,
}

impl AdapterDisplay {
    fn from_info(info: &wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            vendor: info.vendor,
            device: info.device,
        }
    }

    pub fn label(&self) -> String {
        format!("{} ({}, {})", self.name, self.device_type, self.backend)
    }
}

/// Enumerate every adapter wgpu can see on the host. Used by the UI to
/// populate the manual-override dropdown. Creates a throwaway Instance;
/// cheap relative to kernel init.
pub fn list_adapters() -> Vec<AdapterDisplay> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    instance
        .enumerate_adapters(wgpu::Backends::PRIMARY)
        .iter()
        .map(|a| AdapterDisplay::from_info(&a.get_info()))
        .collect()
}

/// Resident GPU compute kernel for element strain/stress. Buffers are
/// resized as element / node counts grow, never shrunk.
#[derive(Debug)]
pub struct GpuStressKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,

    positions: wgpu::Buffer,
    positions_cap: u64,
    elements: wgpu::Buffer,
    elements_cap: u64,
    stresses: wgpu::Buffer,
    stresses_readback: wgpu::Buffer,
    stresses_cap: u64,
    material: wgpu::Buffer,
    /// Workgroup size baked into this pipeline (32 on Metal, 64 elsewhere).
    workgroup_size: u32,
    /// Snapshot of the adapter info this kernel was built against, for
    /// UI display.
    pub adapter: AdapterDisplay,
    /// Chronological log of init attempts ("tried Adapter X → ok",
    /// "tried Adapter Y → failed because Z"). Surfaced in the UI so the
    /// user can see why a particular GPU was or wasn't picked.
    pub init_log: Vec<String>,

    last_positions: usize,
    last_elements: usize,
}

const INITIAL_NODES: u64 = 256;
const INITIAL_ELEMENTS: u64 = 256;

impl GpuStressKernel {
    /// Initialise an independent wgpu instance + device + queue and
    /// build the compute pipeline.
    ///
    /// `preferred` lets the user override wgpu's automatic adapter
    /// pick (PowerPreference::HighPerformance, which picks the
    /// discrete GPU on hybrid systems). When `Some`, this scans the
    /// enumerated adapter list for a matching (name, backend) pair.
    /// Falls back to the auto pick if the preferred adapter is missing
    /// at runtime; logs each attempt in [`Self::init_log`] so the UI
    /// can show why a given GPU was picked.
    pub fn new(preferred: Option<&AdapterDisplay>) -> Result<Self, String> {
        let mut init_log: Vec<String> = Vec::new();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = if let Some(pref) = preferred {
            let mut found = None;
            for cand in instance.enumerate_adapters(wgpu::Backends::PRIMARY) {
                let info = cand.get_info();
                let display = AdapterDisplay::from_info(&info);
                if display.name == pref.name && display.backend == pref.backend {
                    init_log.push(format!("matched user override: {}", display.label()));
                    found = Some(cand);
                    break;
                }
            }
            match found {
                Some(a) => a,
                None => {
                    init_log.push(format!(
                        "user override {} not found at runtime; falling back to auto-pick",
                        pref.label()
                    ));
                    request_default_adapter(&instance, &mut init_log)?
                }
            }
        } else {
            request_default_adapter(&instance, &mut init_log)?
        };

        let adapter_display = AdapterDisplay::from_info(&adapter.get_info());

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("dpm_gpu_kernel_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|e| format!("request_device failed: {e}"))?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let workgroup_size = pick_workgroup_size(adapter.get_info().backend);
        init_log.push(format!(
            "workgroup_size={workgroup_size} for backend {}",
            adapter_display.backend
        ));

        let shader_src = SHADER_SRC.replace("@workgroup_size(64)", &format!("@workgroup_size({workgroup_size})"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dpm_stress_kernel_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_src)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dpm_stress_bgl"),
            entries: &[
                storage_binding(0, true),  // positions (read)
                storage_binding(1, true),  // elements (read)
                storage_binding(2, false), // stresses (write)
                uniform_binding(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dpm_stress_pll"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dpm_stress_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let positions =
            device.create_buffer(&buffer_desc("positions", INITIAL_NODES * 16, true, true, false));
        let elements = device.create_buffer(&buffer_desc(
            "elements",
            INITIAL_ELEMENTS * std::mem::size_of::<ElementInputs>() as u64,
            true,
            true,
            false,
        ));
        let stresses = device.create_buffer(&buffer_desc(
            "stresses",
            INITIAL_ELEMENTS * std::mem::size_of::<StressOut>() as u64,
            true,
            false,
            true,
        ));
        let stresses_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dpm_stresses_readback"),
            size: INITIAL_ELEMENTS * std::mem::size_of::<StressOut>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let material_uniform = MaterialUniform::default();
        let material = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dpm_material"),
            contents: bytemuck::bytes_of(&material_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            bind_group: None,
            positions,
            positions_cap: INITIAL_NODES,
            elements,
            elements_cap: INITIAL_ELEMENTS,
            stresses,
            stresses_readback,
            stresses_cap: INITIAL_ELEMENTS,
            material,
            workgroup_size,
            adapter: adapter_display,
            init_log,
            last_positions: 0,
            last_elements: 0,
        })
    }

    /// True when the GPU kernel can safely replace the CPU strain/stress
    /// phase for this material. The GPU shader currently:
    ///   * handles isotropic only (no 6×6 Voigt path yet)
    ///   * does not update `Element3D::strain_energy` or `is_broken`, so
    ///     any active failure threshold would silently be inert
    /// Both gates are checked here; the UI uses the result to enable or
    /// disable the toggle.
    pub fn supports(material: &MaterialProps3D) -> bool {
        if !matches!(material, MaterialProps3D::Isotropic(_)) {
            return false;
        }
        let fc = material.failure_criteria();
        fc.strain_energy.is_none()
            && fc.tensional_stress.is_none()
            && fc.compressional_stress.is_none()
    }

    /// Compute per-element stresses on the GPU from `computer`'s current
    /// node positions and element reference inverses. The returned `Vec`
    /// is one stress matrix per element, in `computer.elements` order.
    pub fn compute_stresses(
        &mut self,
        computer: &Computer3D,
    ) -> Result<Vec<Matrix3<f32>>, String> {
        let material = match computer.config.material {
            MaterialProps3D::Isotropic(p) => p,
            MaterialProps3D::Orthotropic(_) => {
                return Err("orthotropic material not supported on GPU yet".to_string())
            }
        };
        let n_positions = computer.nodes.len();
        let n_elements = computer.elements.len();
        if n_elements == 0 {
            return Ok(Vec::new());
        }

        // Grow buffers if needed; rebuild the bind group whenever buffer
        // identities change.
        let mut rebind = self.bind_group.is_none();
        rebind |= self.ensure_positions_capacity(n_positions as u64);
        rebind |= self.ensure_elements_capacity(n_elements as u64);
        rebind |= self.ensure_stresses_capacity(n_elements as u64);
        if rebind {
            self.bind_group = Some(self.create_bind_group());
        }

        // Upload positions.
        let positions_data: Vec<[f32; 4]> = computer
            .nodes
            .iter()
            .map(|n| [n.position.x, n.position.y, n.position.z, 0.0])
            .collect();
        self.queue
            .write_buffer(&self.positions, 0, bytemuck::cast_slice(&positions_data));

        // Upload elements (only re-uploading the full set is fine here;
        // for very large meshes we could detect changes and incremental
        // upload, but the per-step bottleneck is positions, not this).
        let elements_data: Vec<ElementInputs> = computer
            .elements
            .iter()
            .map(|e| ElementInputs {
                indices: [
                    e.indices[0] as u32,
                    e.indices[1] as u32,
                    e.indices[2] as u32,
                    e.indices[3] as u32,
                ],
                ref_inv_c0: [e.ref_inv.m11, e.ref_inv.m21, e.ref_inv.m31, 0.0],
                ref_inv_c1: [e.ref_inv.m12, e.ref_inv.m22, e.ref_inv.m32, 0.0],
                ref_inv_c2: [e.ref_inv.m13, e.ref_inv.m23, e.ref_inv.m33, 0.0],
            })
            .collect();
        self.queue
            .write_buffer(&self.elements, 0, bytemuck::cast_slice(&elements_data));

        // Material uniform.
        let mat = MaterialUniform {
            lambda: material.lame_lambda(),
            mu: material.lame_mu(),
            _pad: [0.0; 2],
        };
        self.queue
            .write_buffer(&self.material, 0, bytemuck::bytes_of(&mat));

        // Dispatch. Workgroup size matches what was baked into the shader
        // at pipeline build (32 on Metal, 64 elsewhere).
        let n_groups = (n_elements as u32 + self.workgroup_size - 1) / self.workgroup_size;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dpm_stress_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dpm_stress_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(n_groups, 1, 1);
        }

        // Copy stresses → readback buffer.
        let stress_bytes =
            (n_elements as u64) * std::mem::size_of::<StressOut>() as u64;
        encoder.copy_buffer_to_buffer(
            &self.stresses,
            0,
            &self.stresses_readback,
            0,
            stress_bytes,
        );
        self.queue.submit([encoder.finish()]);

        // Synchronous readback.
        let slice = self.stresses_readback.slice(0..stress_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("readback channel closed: {e}"))?
            .map_err(|e| format!("map_async failed: {e}"))?;

        let data = slice.get_mapped_range();
        let stresses: &[StressOut] = bytemuck::cast_slice(&data);
        let result: Vec<Matrix3<f32>> = stresses[..n_elements]
            .iter()
            .map(|s| {
                Matrix3::new(
                    s.c0[0], s.c1[0], s.c2[0],
                    s.c0[1], s.c1[1], s.c2[1],
                    s.c0[2], s.c1[2], s.c2[2],
                )
            })
            .collect();
        drop(data);
        self.stresses_readback.unmap();

        self.last_positions = n_positions;
        self.last_elements = n_elements;
        Ok(result)
    }

    fn ensure_positions_capacity(&mut self, needed: u64) -> bool {
        if needed <= self.positions_cap {
            return false;
        }
        let new_cap = needed.next_power_of_two().max(INITIAL_NODES);
        self.positions =
            self.device
                .create_buffer(&buffer_desc("positions", new_cap * 16, true, true, false));
        self.positions_cap = new_cap;
        true
    }

    fn ensure_elements_capacity(&mut self, needed: u64) -> bool {
        if needed <= self.elements_cap {
            return false;
        }
        let new_cap = needed.next_power_of_two().max(INITIAL_ELEMENTS);
        self.elements = self.device.create_buffer(&buffer_desc(
            "elements",
            new_cap * std::mem::size_of::<ElementInputs>() as u64,
            true,
            true,
            false,
        ));
        self.elements_cap = new_cap;
        true
    }

    fn ensure_stresses_capacity(&mut self, needed: u64) -> bool {
        if needed <= self.stresses_cap {
            return false;
        }
        let new_cap = needed.next_power_of_two().max(INITIAL_ELEMENTS);
        let bytes = new_cap * std::mem::size_of::<StressOut>() as u64;
        self.stresses = self
            .device
            .create_buffer(&buffer_desc("stresses", bytes, true, false, true));
        self.stresses_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dpm_stresses_readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.stresses_cap = new_cap;
        true
    }

    fn create_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dpm_stress_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.elements.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.stresses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.material.as_entire_binding(),
                },
            ],
        })
    }
}

/// Default-pick path: HighPerformance preference (picks the discrete
/// GPU on hybrid systems). Logged into `init_log` so the UI can show
/// what was tried.
fn request_default_adapter(
    instance: &wgpu::Instance,
    init_log: &mut Vec<String>,
) -> Result<wgpu::Adapter, String> {
    init_log.push("auto-pick: PowerPreference::HighPerformance".to_string());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "no wgpu adapter available".to_string())?;
    let info = adapter.get_info();
    init_log.push(format!(
        "picked {} ({:?}, {:?})",
        info.name, info.device_type, info.backend
    ));
    Ok(adapter)
}

fn storage_binding(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_desc(
    name: &'static str,
    size: u64,
    storage: bool,
    copy_dst: bool,
    copy_src: bool,
) -> wgpu::BufferDescriptor<'static> {
    let mut usage = wgpu::BufferUsages::empty();
    if storage {
        usage |= wgpu::BufferUsages::STORAGE;
    }
    if copy_dst {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    if copy_src {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    wgpu::BufferDescriptor {
        label: Some(name),
        size,
        usage,
        mapped_at_creation: false,
    }
}

const SHADER_SRC: &str = r#"
struct Position {
    p: vec4<f32>,
};

struct Element {
    indices: vec4<u32>,
    ref_inv_c0: vec4<f32>,
    ref_inv_c1: vec4<f32>,
    ref_inv_c2: vec4<f32>,
};

struct Stress {
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
};

struct Material {
    lambda: f32,
    mu: f32,
};

@group(0) @binding(0) var<storage, read>       positions: array<Position>;
@group(0) @binding(1) var<storage, read>       elements:  array<Element>;
@group(0) @binding(2) var<storage, read_write> stresses:  array<Stress>;
@group(0) @binding(3) var<uniform>             material:  Material;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&elements)) {
        return;
    }
    let e = elements[i];
    let p0 = positions[e.indices.x].p.xyz;
    let p1 = positions[e.indices.y].p.xyz;
    let p2 = positions[e.indices.z].p.xyz;
    let p3 = positions[e.indices.w].p.xyz;

    // D = [p1-p0 | p2-p0 | p3-p0]   (column-major 3x3)
    let d0 = p1 - p0;
    let d1 = p2 - p0;
    let d2 = p3 - p0;

    // F = D * R^{-1}, where R^{-1} columns are ref_inv_c0..c2.
    let r0 = e.ref_inv_c0.xyz;
    let r1 = e.ref_inv_c1.xyz;
    let r2 = e.ref_inv_c2.xyz;
    let f0 = d0 * r0.x + d1 * r0.y + d2 * r0.z;
    let f1 = d0 * r1.x + d1 * r1.y + d2 * r1.z;
    let f2 = d0 * r2.x + d1 * r2.y + d2 * r2.z;

    // Green-Lagrange strain E = 0.5 (F^T F - I)
    let ftf00 = dot(f0, f0);
    let ftf01 = dot(f0, f1);
    let ftf02 = dot(f0, f2);
    let ftf11 = dot(f1, f1);
    let ftf12 = dot(f1, f2);
    let ftf22 = dot(f2, f2);
    let e00 = 0.5 * (ftf00 - 1.0);
    let e11 = 0.5 * (ftf11 - 1.0);
    let e22 = 0.5 * (ftf22 - 1.0);
    let e01 = 0.5 * ftf01;
    let e02 = 0.5 * ftf02;
    let e12 = 0.5 * ftf12;

    // Isotropic stress sigma = lambda tr(E) I + 2 mu E.
    let tr = e00 + e11 + e22;
    let lt = material.lambda * tr;
    let m2 = 2.0 * material.mu;
    let s00 = lt + m2 * e00;
    let s11 = lt + m2 * e11;
    let s22 = lt + m2 * e22;
    let s01 = m2 * e01;
    let s02 = m2 * e02;
    let s12 = m2 * e12;

    var out: Stress;
    out.c0 = vec4<f32>(s00, s01, s02, 0.0);
    out.c1 = vec4<f32>(s01, s11, s12, 0.0);
    out.c2 = vec4<f32>(s02, s12, s22, 0.0);
    stresses[i] = out;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use cpd::d3::{Config3D, IsotropicProps3D, MaterialProps3D};
    use nalgebra::Vector3;

    fn single_tet_stretched(strain: f32) -> Computer3D {
        let v = vec![
            Vector3::new(0.0_f32, 0.0, 0.0),
            Vector3::new(1.0 + strain, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let t = vec![[0, 1, 2, 3]];
        let mut cfg = Config3D::default();
        cfg.material = MaterialProps3D::Isotropic(IsotropicProps3D::default());
        let mut c = Computer3D::from_mesh(&v, &t, cfg).unwrap();
        let rest = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        for (n, r) in c.nodes.iter_mut().zip(rest.iter()) {
            n.initial_position = *r;
        }
        c.elements[0] = cpd::d3::Element3D::from_reference([0, 1, 2, 3], rest).unwrap();
        c
    }

    /// CPU and GPU strain/stress should match within float rounding.
    /// Skipped at runtime if the host doesn't expose a wgpu adapter
    /// (CI containers, etc).
    #[test]
    fn gpu_matches_cpu_on_uniaxial_stretch() {
        let Ok(mut kernel) = GpuStressKernel::new(None) else {
            eprintln!("no GPU adapter; skipping E18 parity test");
            return;
        };
        let mut c = single_tet_stretched(0.01);
        // CPU pass.
        c.update_strain_stress_cpu();
        let cpu_stress = c.elements[0].stress;
        // GPU pass on the same current configuration.
        let gpu_stresses = kernel.compute_stresses(&c).expect("compute_stresses");
        assert_eq!(gpu_stresses.len(), 1);
        let gpu_stress = gpu_stresses[0];
        for i in 0..3 {
            for j in 0..3 {
                let diff = (cpu_stress[(i, j)] - gpu_stress[(i, j)]).abs();
                assert!(
                    diff < 1.0,
                    "mismatch at ({i},{j}): cpu={} gpu={} diff={diff}",
                    cpu_stress[(i, j)],
                    gpu_stress[(i, j)],
                );
            }
        }
    }
}
