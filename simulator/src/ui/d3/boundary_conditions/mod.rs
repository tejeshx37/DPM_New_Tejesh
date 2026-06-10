//! Dedicated 3D Boundary Conditions phase.
//!
//! Sits between Drawing and Meshing in the new pipeline order so users
//! can assign BCs against shape regions *before* a mesh exists. The
//! page reads `Geometry3D` directly, enumerates each shape's canonical
//! boundary regions via `Shape3D::region_names()`, and writes BC
//! choices into `simulation::State::region_bcs` keyed by
//! `body{shape_index+1}/{region}` — the same naming scheme
//! `Mesh3D::combine` produces once meshes are generated, so the keys
//! survive intact through the Meshing phase.
//!
//! Preview viewport renders the bare shapes (no mesh needed) through
//! the same wgpu scene callback used by the Drawing page, with the
//! existing simulation BC-overlay arrows drawn on top so the user can
//! see what their settings will do.

use egui::{Color32, ScrollArea, Sense, SidePanel, Stroke, Ui, Vec2};
use nalgebra::Vector3;

use super::drawing::{
    shape::{CsgOp3D, Geometry3D},
    viewport::{camera::OrbitCamera, project, scene_mesh, wgpu_scene},
};
use super::simulation::{self, Axis, BcKind, RegionBc};

/// Key under which BC settings are stored. Matches the post-mesh region
/// name (`body{i+1}/{region}`) so transitioning to the Meshing /
/// Simulation phases doesn't lose the choices the user made here.
fn region_key(shape_index: usize, region: &str) -> String {
    format!("body{}/{}", shape_index + 1, region)
}

/// Paint BC marker arrows / pinned bars for every region in the
/// geometry that has a configured BC. Shared by the BC page (which
/// passes its own viewport rect / view-projection) and the Meshing
/// page's "Show constraints" toggle.
pub fn paint_constraint_overlays(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &nalgebra::Matrix4<f64>,
    geometry: &Geometry3D,
    region_bcs: &std::collections::HashMap<String, RegionBc>,
) {
    let (lo, hi) = geometry.aabb();
    let scene_scale = (hi - lo).norm().max(1e-6);
    let arrow_len = scene_scale * 0.18;
    for (idx, (shape, _op)) in geometry.shapes.iter().enumerate() {
        let (s_lo, s_hi) = shape.aabb();
        for region_name in shape.region_names() {
            let key = region_key(idx, region_name);
            let Some(bc) = region_bcs.get(&key) else {
                continue;
            };
            if matches!(bc.kind, BcKind::Free) {
                continue;
            }
            let normal = region_outward_normal(region_name);
            let centroid = region_centroid(s_lo, s_hi, region_name);
            let base = centroid + normal * scene_scale * 0.02;
            paint_bc_marker(painter, rect, view_proj, bc, base, normal, arrow_len, &key);
        }
    }
}

pub fn show(sim_state: &mut simulation::State, geometry: &Geometry3D, ui: &mut Ui) {
    SidePanel::right("d3_bc_side_panel")
        .resizable(true)
        .default_width(340.0)
        .show_inside(ui, |ui| {
            ui.heading("Boundary Conditions");
            ui.separator();
            if geometry.shapes.is_empty() {
                ui.colored_label(
                    Color32::YELLOW,
                    "No shapes drawn yet. Add at least one shape on the Drawing page first.",
                );
                return;
            }
            let total_regions: usize = geometry
                .shapes
                .iter()
                .map(|(s, _)| s.region_names().len())
                .sum();
            ui.label(format!(
                "{} bod{}, {} boundary region{}",
                geometry.shapes.len(),
                if geometry.shapes.len() == 1 { "y" } else { "ies" },
                total_regions,
                if total_regions == 1 { "" } else { "s" }
            ));
            ui.label("Assigned now; applied after meshing.");
            ui.separator();
            ScrollArea::vertical().show(ui, |ui| {
                add_geometry_bc_controls(sim_state, geometry, ui);
            });
        });
    show_preview_viewport(sim_state, geometry, ui);
}

/// Iterate every shape's canonical boundary regions and surface a BC
/// editor for each. Mirrors `simulation::add_bc_controls` but is keyed
/// off the geometry instead of an existing mesh.
fn add_geometry_bc_controls(sim_state: &mut simulation::State, geometry: &Geometry3D, ui: &mut Ui) {
    for (idx, (shape, _op)) in geometry.shapes.iter().enumerate() {
        ui.collapsing(
            format!("Body {} ({})", idx + 1, shape.kind().label()),
            |ui| {
                for (region_idx, region_name) in shape.region_names().iter().enumerate() {
                    let key = region_key(idx, region_name);
                    let entry = sim_state.region_bcs.entry(key.clone()).or_default();
                    ui.group(|ui| {
                        // Lead with the B# label that's painted on the
                        // shape's face in the preview viewport so users
                        // can map between the panel and the 3D scene at
                        // a glance.
                        ui.strong(format!(
                            "B{} — {}/{}",
                            region_idx + 1,
                            body_label(idx + 1),
                            region_name
                        ));
                        bc_kind_combo(&key, entry, ui);
                        bc_parameter_fields(entry, ui);
                    });
                }
            },
        );
    }
}

fn body_label(i: usize) -> String {
    format!("body{i}")
}

fn bc_kind_combo(key: &str, entry: &mut RegionBc, ui: &mut Ui) {
    egui::ComboBox::from_id_source(format!("bc_kind_{key}"))
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
}

fn bc_parameter_fields(entry: &mut RegionBc, ui: &mut Ui) {
    match entry.kind {
        BcKind::Free => {}
        BcKind::Pinned => axes_row(ui, &mut entry.axes),
        BcKind::ConstantForce => vec_row(ui, "Force", &mut entry.force),
        BcKind::ConstantDisplacement => {
            axes_row(ui, &mut entry.axes);
            vec_row(ui, "Target", &mut entry.displacement);
            ui.horizontal(|ui| {
                ui.label("Ramp (s)");
                ui.add(
                    egui::DragValue::new(&mut entry.ramp_seconds)
                        .speed(0.001)
                        .clamp_range(0.0..=f32::MAX),
                );
            });
        }
        BcKind::TimeForce => {
            ui.label("Edit keyframes on the Simulation page (one editor instance).");
        }
        BcKind::TimeDisplacement => {
            axes_row(ui, &mut entry.axes);
            ui.label("Edit keyframes on the Simulation page (one editor instance).");
        }
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
        ui.add(egui::DragValue::new(&mut v[0]).speed(0.1));
        ui.add(egui::DragValue::new(&mut v[1]).speed(0.1));
        ui.add(egui::DragValue::new(&mut v[2]).speed(0.1));
    });
}

fn show_preview_viewport(sim_state: &mut simulation::State, geometry: &Geometry3D, ui: &mut Ui) {
    let available = ui.available_size();
    let size = Vec2::new(available.x.max(100.0), available.y.max(100.0));
    let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
    let rect = response.rect;

    if sim_state.viewport.auto_frame && !geometry.shapes.is_empty() {
        let (lo, hi) = geometry.aabb();
        sim_state.viewport.camera.frame_aabb(lo, hi);
        sim_state.viewport.auto_frame = false;
    }

    handle_camera_input(&mut sim_state.viewport.camera, &response, ui);
    painter.rect_filled(rect, 0.0, Color32::from_gray(20));

    let view_proj = sim_state
        .viewport
        .camera
        .view_projection(rect.width() / rect.height().max(1.0));

    // Tessellate the bare shapes; same look as the Drawing viewport.
    let mut tris = Vec::new();
    for (shape, op) in &geometry.shapes {
        let color = match op {
            CsgOp3D::Union => [0.47, 0.78, 1.0, 0.95],
            CsgOp3D::Difference => [1.0, 0.55, 0.55, 0.35],
        };
        tris.extend(scene_mesh::triangles_for(shape, color));
    }
    if !tris.is_empty() {
        wgpu_scene::sort_back_to_front(&mut tris, &view_proj);
        let cb = wgpu_scene::SceneCallback::from_world_mvp(tris, &view_proj);
        painter.add(eframe::egui_wgpu::Callback::new_paint_callback(rect, cb));
    }

    // BC marker overlays: pinned bars + force/displacement arrows on each
    // shape, sized to the geometry AABB. Centroid is the shape's AABB
    // center; outward normals are perpendicular to whichever face we're
    // labelling and computed from the canonical region name.
    let (lo, hi) = geometry.aabb();
    let scene_scale = (hi - lo).norm().max(1e-6);
    let arrow_len = scene_scale * 0.18;
    for (idx, (shape, _op)) in geometry.shapes.iter().enumerate() {
        let (s_lo, s_hi) = shape.aabb();
        for region_name in shape.region_names() {
            let key = region_key(idx, region_name);
            let Some(bc) = sim_state.region_bcs.get(&key) else {
                continue;
            };
            if matches!(bc.kind, BcKind::Free) {
                continue;
            }
            let normal = region_outward_normal(region_name);
            let centroid = region_centroid(s_lo, s_hi, region_name);
            let base = centroid + normal * scene_scale * 0.02;
            paint_bc_marker(
                &painter, rect, &view_proj, bc, base, normal, arrow_len, &key,
            );
        }
    }

    // B-labels on each face, matching the BC panel's "B1 — body1/x_min"
    // headings. Red when the region has a configured (non-Free) BC,
    // white otherwise — same convention as the reference 2D mesher.
    paint_region_labels(&painter, rect, &view_proj, geometry, &sim_state.region_bcs);
}

fn paint_region_labels(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &nalgebra::Matrix4<f64>,
    geometry: &Geometry3D,
    region_bcs: &std::collections::HashMap<String, RegionBc>,
) {
    let (lo, hi) = geometry.aabb();
    let scene_scale = (hi - lo).norm().max(1e-6);
    let font = egui::FontId::monospace(14.0);
    let configured_color = Color32::from_rgb(255, 140, 140);
    let free_color = Color32::from_gray(230);

    for (idx, (shape, _op)) in geometry.shapes.iter().enumerate() {
        let (s_lo, s_hi) = shape.aabb();
        for (region_idx, region_name) in shape.region_names().iter().enumerate() {
            let key = region_key(idx, region_name);
            let configured = region_bcs
                .get(&key)
                .map(|b| !matches!(b.kind, BcKind::Free))
                .unwrap_or(false);
            let normal = region_outward_normal(region_name);
            let centroid = region_centroid(s_lo, s_hi, region_name);
            // Push slightly off the surface so the text doesn't z-fight
            // with the body and is readable from any angle.
            let anchor = centroid + normal * scene_scale * 0.04;
            if let Some(p) = project(view_proj, rect, anchor) {
                painter.text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    format!("B{}", region_idx + 1),
                    font.clone(),
                    if configured { configured_color } else { free_color },
                );
            }
        }
    }
}

fn paint_bc_marker(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &nalgebra::Matrix4<f64>,
    bc: &RegionBc,
    base: Vector3<f64>,
    normal: Vector3<f64>,
    arrow_len: f64,
    key: &str,
) {
    match bc.kind {
        BcKind::Free => {}
        BcKind::Pinned => {
            // Three short bars colored per locked axis.
            let bars = [
                (bc.axes.x, Vector3::new(1.0, 0.0, 0.0), Color32::from_rgb(240, 110, 110)),
                (bc.axes.y, Vector3::new(0.0, 1.0, 0.0), Color32::from_rgb(110, 220, 110)),
                (bc.axes.z, Vector3::new(0.0, 0.0, 1.0), Color32::from_rgb(110, 160, 250)),
            ];
            let size = arrow_len * 0.25;
            for (on, dir, color) in bars {
                if !on {
                    continue;
                }
                let a = base - dir * size * 0.5;
                let b = base + dir * size * 0.5;
                if let (Some(pa), Some(pb)) =
                    (project(view_proj, rect, a), project(view_proj, rect, b))
                {
                    painter.line_segment([pa, pb], Stroke::new(2.5, color));
                }
            }
            if let Some(p) = project(view_proj, rect, base) {
                painter.circle_stroke(p, 4.0, Stroke::new(1.2, Color32::from_rgb(200, 220, 240)));
                painter.text(
                    p + egui::vec2(6.0, -6.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{key}: pinned"),
                    egui::FontId::monospace(10.0),
                    Color32::from_rgb(160, 200, 240),
                );
            }
        }
        BcKind::ConstantForce | BcKind::TimeForce => {
            let f = match bc.kind {
                BcKind::ConstantForce => Vector3::new(bc.force[0] as f64, bc.force[1] as f64, bc.force[2] as f64),
                _ => normal, // direction varies with time; show along surface normal
            };
            let dir = if f.norm() > 1e-12 { f.normalize() } else { normal };
            let tip = base + dir * arrow_len;
            paint_arrow(painter, rect, view_proj, base, tip, Color32::from_rgb(255, 200, 100));
            if let Some(p) = project(view_proj, rect, tip) {
                painter.text(
                    p + egui::vec2(6.0, -6.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{key}: F"),
                    egui::FontId::monospace(10.0),
                    Color32::from_rgb(255, 200, 100),
                );
            }
        }
        BcKind::ConstantDisplacement | BcKind::TimeDisplacement => {
            let d = match bc.kind {
                BcKind::ConstantDisplacement => Vector3::new(bc.displacement[0] as f64, bc.displacement[1] as f64, bc.displacement[2] as f64),
                _ => normal,
            };
            let dir = if d.norm() > 1e-12 { d.normalize() } else { normal };
            let tip = base + dir * arrow_len;
            paint_arrow(painter, rect, view_proj, base, tip, Color32::from_rgb(160, 240, 160));
            if let Some(p) = project(view_proj, rect, tip) {
                painter.text(
                    p + egui::vec2(6.0, -6.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{key}: u"),
                    egui::FontId::monospace(10.0),
                    Color32::from_rgb(160, 240, 160),
                );
            }
        }
    }
}

fn paint_arrow(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &nalgebra::Matrix4<f64>,
    base: Vector3<f64>,
    tip: Vector3<f64>,
    color: Color32,
) {
    let (Some(p0), Some(p1)) = (project(view_proj, rect, base), project(view_proj, rect, tip)) else {
        return;
    };
    let stroke = Stroke::new(2.0, color);
    painter.line_segment([p0, p1], stroke);
    // Simple arrowhead.
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let len = (dx * dx + dy * dy).sqrt().max(1e-3);
    let ux = dx / len;
    let uy = dy / len;
    let head = 9.0_f32.min(len * 0.4);
    let px = -uy;
    let py = ux;
    let a = egui::pos2(p1.x - ux * head + px * head * 0.5, p1.y - uy * head + py * head * 0.5);
    let b = egui::pos2(p1.x - ux * head - px * head * 0.5, p1.y - uy * head - py * head * 0.5);
    painter.line_segment([p1, a], stroke);
    painter.line_segment([p1, b], stroke);
}

/// Outward face normal for a canonical region name. Sphere and cylinder
/// "side" use radial / axis-aligned fallbacks since the BC overlay is a
/// 2D marker; the actual mesher provides per-vertex normals at runtime.
fn region_outward_normal(name: &str) -> Vector3<f64> {
    match name {
        "x_min" => Vector3::new(-1.0, 0.0, 0.0),
        "x_max" => Vector3::new(1.0, 0.0, 0.0),
        "y_min" => Vector3::new(0.0, -1.0, 0.0),
        "y_max" => Vector3::new(0.0, 1.0, 0.0),
        "z_min" => Vector3::new(0.0, 0.0, -1.0),
        "z_max" => Vector3::new(0.0, 0.0, 1.0),
        "bottom" => Vector3::new(0.0, -1.0, 0.0),
        "top" => Vector3::new(0.0, 1.0, 0.0),
        "side" => Vector3::new(1.0, 0.0, 0.0),
        "surface" => Vector3::new(0.0, 1.0, 0.0),
        _ => Vector3::new(0.0, 1.0, 0.0),
    }
}

/// Centroid of the canonical region on a shape's AABB. Approximate for
/// curved regions (sphere "surface", cylinder "side") — good enough to
/// anchor a marker.
fn region_centroid(lo: Vector3<f64>, hi: Vector3<f64>, name: &str) -> Vector3<f64> {
    let c = (lo + hi) * 0.5;
    match name {
        "x_min" => Vector3::new(lo.x, c.y, c.z),
        "x_max" => Vector3::new(hi.x, c.y, c.z),
        "y_min" | "bottom" => Vector3::new(c.x, lo.y, c.z),
        "y_max" | "top" => Vector3::new(c.x, hi.y, c.z),
        "z_min" => Vector3::new(c.x, c.y, lo.z),
        "z_max" => Vector3::new(c.x, c.y, hi.z),
        "side" => Vector3::new(hi.x, c.y, c.z),
        "surface" => Vector3::new(c.x, hi.y, c.z),
        _ => c,
    }
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
