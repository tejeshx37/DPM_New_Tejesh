//! wgpu pipeline for the 3D viewports.
//!
//! Holds a single render pipeline, vertex buffer (grown on demand), and
//! uniform buffer. Stored once in `egui_wgpu::Renderer::callback_resources`
//! at app start so paint callbacks across the drawing and simulation
//! viewports share the same GPU resources.
//!
//! Rendering strategy for this iteration: no depth attachment (egui's
//! render pass does not expose one). Triangles are sorted back-to-front
//! on the CPU by clip-space centroid Z before being uploaded. Lambert
//! shading + ambient against a single fixed light. A real depth buffer
//! and PBR materials are planned follow-ups.

use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu;
use egui_wgpu::CallbackTrait;
use nalgebra::Matrix4;
use wgpu::util::DeviceExt;

/// Per-vertex POD layout. Color is RGBA with linear-space components in
/// `[0, 1]`; the fragment shader treats alpha as straight transparency
/// (no premultiplication needed because the only translucent shapes are
/// Difference primitives, which we paint without face culling).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    light_dir: [f32; 4],
    ambient: [f32; 4],
}

pub struct Scene {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: u64,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

const INITIAL_VERTEX_CAPACITY: u64 = 4096;

impl Scene {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dpm_scene_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SRC)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dpm_scene_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dpm_scene_pll"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dpm_scene_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Disable culling: Difference shapes are translucent and
                // depth-sorted, and the user's CSG-Difference visual is
                // clearer when back faces also render.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dpm_scene_vbuf"),
            size: INITIAL_VERTEX_CAPACITY * std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms = Uniforms {
            view_proj: Matrix4::identity().data.0.map(|c| [c[0], c[1], c[2], c[3]]),
            light_dir: [0.4, 0.75, 0.5, 0.0],
            ambient: [0.25, 0.25, 0.30, 1.0],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dpm_scene_ubuf"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dpm_scene_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            bind_group_layout,
            vertex_buffer,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
            uniform_buffer,
            bind_group,
        }
    }

    fn ensure_capacity(&mut self, device: &wgpu::Device, needed: u64) {
        if needed <= self.vertex_capacity {
            return;
        }
        let new_cap = needed.next_power_of_two().max(INITIAL_VERTEX_CAPACITY);
        self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dpm_scene_vbuf"),
            size: new_cap * std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.vertex_capacity = new_cap;
        // Bind group references the uniform buffer, not vertex buffer; no
        // rebind needed when growing the vertex buffer.
        let _ = &self.bind_group_layout;
    }
}

/// Convert an `f64` view-projection matrix (column-major from nalgebra) to
/// the column-major `[[f32; 4]; 4]` layout wgpu expects, applying the
/// OpenGL → wgpu NDC z-remap (z' = (z + w) / 2) at the same time so the
/// existing `OrbitCamera::view_projection` (OpenGL convention) renders
/// correctly under wgpu without changing the camera math.
fn mvp_to_uniform(view_proj: &Matrix4<f64>) -> [[f32; 4]; 4] {
    // Row-major remap M' = R * M where R fixes z to [0, 1]:
    //   R = | 1 0   0   0 |
    //       | 0 1   0   0 |
    //       | 0 0 0.5 0.5 |
    //       | 0 0   0   1 |
    let mut out = [[0.0_f32; 4]; 4];
    for col in 0..4 {
        let m0 = view_proj[(0, col)] as f32;
        let m1 = view_proj[(1, col)] as f32;
        let m2 = view_proj[(2, col)] as f32;
        let m3 = view_proj[(3, col)] as f32;
        out[col] = [m0, m1, 0.5 * (m2 + m3), m3];
    }
    out
}

/// One frame's worth of triangles + camera, sent through `painter.add`.
/// Vertex data is sorted on the CPU before construction so the GPU just
/// renders in supplied order (back-to-front).
pub struct SceneCallback {
    pub vertices: Arc<Vec<Vertex>>,
    pub mvp: [[f32; 4]; 4],
}

impl SceneCallback {
    pub fn from_world_mvp(vertices: Vec<Vertex>, view_proj: &Matrix4<f64>) -> Self {
        Self {
            vertices: Arc::new(vertices),
            mvp: mvp_to_uniform(view_proj),
        }
    }
}

impl CallbackTrait for SceneCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(scene) = callback_resources.get_mut::<Scene>() else {
            return Vec::new();
        };
        if self.vertices.is_empty() {
            return Vec::new();
        }
        scene.ensure_capacity(device, self.vertices.len() as u64);
        queue.write_buffer(&scene.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        let uniforms = Uniforms {
            view_proj: self.mvp,
            light_dir: [0.4, 0.75, 0.5, 0.0],
            ambient: [0.18, 0.18, 0.22, 1.0],
        };
        queue.write_buffer(&scene.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        Vec::new()
    }

    fn paint<'a>(
        &'a self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'a>,
        callback_resources: &'a egui_wgpu::CallbackResources,
    ) {
        let Some(scene) = callback_resources.get::<Scene>() else {
            return;
        };
        if self.vertices.is_empty() {
            return;
        }
        render_pass.set_pipeline(&scene.pipeline);
        render_pass.set_bind_group(0, &scene.bind_group, &[]);
        render_pass.set_vertex_buffer(0, scene.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}

/// Sort triangles back-to-front by clip-space centroid Z. `verts.len()`
/// must be a multiple of 3. Cheap for the scene sizes we expect
/// (≤ ~10k triangles).
pub fn sort_back_to_front(verts: &mut Vec<Vertex>, view_proj: &Matrix4<f64>) {
    if verts.len() < 3 {
        return;
    }
    let n_tris = verts.len() / 3;
    let mut keys: Vec<(usize, f64)> = (0..n_tris)
        .map(|i| {
            let p0 = verts[i * 3].position;
            let p1 = verts[i * 3 + 1].position;
            let p2 = verts[i * 3 + 2].position;
            let cx = (p0[0] + p1[0] + p2[0]) as f64 / 3.0;
            let cy = (p0[1] + p1[1] + p2[1]) as f64 / 3.0;
            let cz = (p0[2] + p1[2] + p2[2]) as f64 / 3.0;
            let clip = view_proj * nalgebra::Vector4::new(cx, cy, cz, 1.0);
            // Larger z/w = further from camera in OpenGL convention.
            (i, clip.z / clip.w)
        })
        .collect();
    keys.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let original = verts.clone();
    for (out_idx, (in_idx, _)) in keys.into_iter().enumerate() {
        verts[out_idx * 3] = original[in_idx * 3];
        verts[out_idx * 3 + 1] = original[in_idx * 3 + 1];
        verts[out_idx * 3 + 2] = original[in_idx * 3 + 2];
    }
}

const SHADER_SRC: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    ambient: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_normal = in.normal;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(u.light_dir.xyz);
    let lambert = max(dot(n, l), 0.0);
    let lit = in.color.rgb * (u.ambient.rgb + vec3<f32>(lambert));
    return vec4<f32>(lit, in.color.a);
}
"#;
